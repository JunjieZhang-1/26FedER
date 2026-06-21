#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PFedRN Net2 Bootstrap Oracle
============================

Purpose:
  Diagnose whether a local personalized model (net2) has an early clean-sample
  window that can bootstrap FedRN faster than the original 100-round warmup.

Main idea:
  1. Split each client's original local data before noise injection:
       80% noisy train for net1/net2 training, 20% clean holdout only for
       diagnosing net2 and selecting its best early checkpoint.
  2. Stage 1, net2 search:
       train each selected client's net2 on its noisy 80% train data, evaluate
       on its 20% clean holdout, and keep the best checkpoint before
       net2_search_epochs.
  3. Stage 2, net2 bootstrap:
       use the best net2 checkpoint to run GMM on the local noisy train data.
       The clean set selected by net2 is used directly to train net1.
  4. Stage 3, FedRN stable:
       after fedrn_start_epoch, switch back to original FedRN selection
       (target client + reliable neighbors + one-pass head fine-tuning).

Important:
  This is an oracle/upper-bound diagnostic if best net2 is selected by clean
  holdout labels. If it works, the next step is replacing holdout selection
  with an unsupervised criterion.
"""

import os
import sys
import copy
import time
import random
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.options_pfedrn_net2_bootstrap_oracle import args_parser
from utils import load_dataset
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label
from models.fed import LocalModelWeights
from models.nets_original_fedrn import get_model as get_fedrn_model
from models.test import test_img
from models.update_original_fedrn import DatasetSplit
from sklearn.mixture import GaussianMixture

# ================================================================
# Windows 环境兼容
# ================================================================
conda_path = r"C:\Users\25839\.conda\envs\improve-FedRN-main\Library\bin"
if os.path.exists(conda_path):
    os.environ['PATH'] = conda_path + os.pathsep + os.environ['PATH']


def get_logits(output, head='global'):
    """返回模型logits；当前Speedup中net1/net2均为原版FedRN单头模型。"""
    if isinstance(output, (tuple, list)):
        if head == 'local' and len(output) > 1:
            return output[1]
        return output[0]
    return output


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def safe_minmax(values):
    values = list(values)
    v_min, v_max = min(values), max(values)
    denom = v_max - v_min
    if abs(float(denom)) < 1e-12:
        return [0.0 for _ in values]
    return [(v - v_min) / denom for v in values]


def is_local_head_key(key):
    return False


def is_head_key(key):
    return ('linear' in key) or key.startswith('fc2')


def fedavg(weights, weight_list):
    """FedAvg 聚合。当前net1是原版FedRN单头模型，因此聚合全部参数。"""
    if len(weights) == 0:
        return None
    total = float(sum(weight_list))
    w_avg = copy.deepcopy(weights[0])
    for k in w_avg.keys():
        if is_local_head_key(k):
            continue
        w_avg[k] = w_avg[k] * weight_list[0]
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k] * weight_list[i]
        w_avg[k] = torch.div(w_avg[k], total)
    return w_avg


def sync_global_to_client(local_model, global_state, mode='global'):
    """
    mode='global': 同步全部参数
    mode='personal': 只同步特征层，保留原版模型的 linear 分类头
    """
    state = local_model.state_dict()
    for k, v in global_state.items():
        if k not in state:
            continue
        if mode == 'global':
            if is_local_head_key(k):
                continue
            state[k] = v.clone()
        elif mode == 'personal':
            if is_head_key(k):
                continue
            state[k] = v.clone()
    local_model.load_state_dict(state)


def split_client_indices(idxs, train_ratio, seed):
    idxs = np.asarray(list(idxs), dtype=int)
    rng = np.random.RandomState(seed)
    shuffled = idxs.copy()
    rng.shuffle(shuffled)
    train_len = int(len(shuffled) * train_ratio)
    train_idx = shuffled[:train_len]
    holdout_idx = shuffled[train_len:]
    return train_idx.astype(int).tolist(), holdout_idx.astype(int).tolist()


# ================================================================
# 客户端本地更新对象（直接继承 update.py 中的 LocalUpdateFedRN 并魔改）
# ================================================================

class LocalUpdateNet1Net2:
    """
    双模型本地更新器：
      - net1: 全局模型，使用原版FedRN单头模型
      - net2: 本地个性化模型，也使用原版FedRN单头模型
      - 判断是否干净：原版FedRN概率 + net2辅助概率
    """

    def __init__(self, args, dataset=None, user_idx=None, train_idxs=None,
                 holdout_idxs=None, gaussian_noise=None, clean_label_mask=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = list(train_idxs)
        self.data_indices = np.array(train_idxs)
        self.holdout_indices = np.array(holdout_idxs, dtype=int)
        self.gaussian_noise = gaussian_noise
        self.clean_label_mask = clean_label_mask

        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.loss_func = nn.CrossEntropyLoss()

        # net1/net2都使用原版FedRN单头模型；net2只是本地保留，不参与聚合。
        self.net1 = get_fedrn_model(args).to(args.device)
        self.net2 = get_fedrn_model(args).to(args.device)

        # DataLoader for GMM on this client's noisy train split.
        self.ldr_eval = DataLoader(
            DatasetSplit(dataset, self.data_indices, real_idx_return=True),
            batch_size=args.local_bs, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        self.ldr_holdout = DataLoader(
            DatasetSplit(dataset, self.holdout_indices, real_idx_return=True),
            batch_size=args.local_bs, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        # FedRN 需要的属性
        self.expertise = 0.5
        self.arbitrary_output = torch.rand((1, args.num_classes))

        # 统计
        self.last_selection_stats = {}
        self.last_scorea_stats = {}
        self.last_fedrn_stats = {}
        self.last_net2_stats = {}
        self.last_disagreement_stats = {}
        self.last_class_counts = {}
        self.net2_ready = False
        self.net2_best_state = None
        self.net2_best_epoch = -1
        self.net2_best_holdout_acc = -1.0
        self.net2_last_holdout_acc = -1.0
        self.net2_last_holdout_loss = -1.0
        self.net2_best_updated = False

    # ---------- 辅助：获取 logits ----------
    def _select_logits(self, outputs, head='global'):
        if isinstance(outputs, (tuple, list)):
            if head == 'local' and len(outputs) > 1:
                return outputs[1]
            return outputs[0]
        return outputs

    def _forward_logits(self, net, inputs, head='global'):
        outputs = net(inputs)
        return self._select_logits(outputs, head=head)

    # ---------- 真实 clean mask 统计 ----------
    def _true_clean_flags(self, indices):
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return np.asarray([], dtype=bool)
        if self.clean_label_mask is None:
            return np.ones(len(indices), dtype=bool)
        if isinstance(self.clean_label_mask, dict):
            return np.asarray(
                [bool(self.clean_label_mask.get(int(idx), True)) for idx in indices],
                dtype=bool,
            )
        return np.asarray(self.clean_label_mask, dtype=bool)[indices]

    def _build_selection_stats(self, pred_clean_idx, pred_noisy_idx):
        total_indices = self.data_indices
        pc = np.asarray(pred_clean_idx, dtype=int)
        pn = np.asarray(pred_noisy_idx, dtype=int)
        total_count = len(total_indices)
        pred_clean_count = len(pc)
        pred_noisy_count = len(pn)
        true_clean_all = self._true_clean_flags(total_indices)
        true_clean_in_pred_clean = self._true_clean_flags(pc)
        true_clean_in_pred_noisy = self._true_clean_flags(pn)

        actual_train_clean_count = int(true_clean_in_pred_clean.sum())
        false_clean_count = int(pred_clean_count - actual_train_clean_count)
        missed_clean_count = int(true_clean_in_pred_noisy.sum())

        return {
            'total_count': total_count,
            'true_clean_count': int(true_clean_all.sum()),
            'pred_clean_count': pred_clean_count,
            'pred_noisy_count': pred_noisy_count,
            'actual_train_clean_count': actual_train_clean_count,
            'actual_train_clean_ratio': float(actual_train_clean_count) / max(pred_clean_count, 1),
            'clean_precision': float(actual_train_clean_count) / max(pred_clean_count, 1),
            'clean_recall': float(actual_train_clean_count) / max(true_clean_all.sum(), 1),
            'missed_clean_count': missed_clean_count,
            'false_clean_count': false_clean_count,
        }

    def _complement_idx(self, clean_idx):
        clean_set = set(int(i) for i in list(clean_idx))
        return np.asarray(
            [int(i) for i in self.data_indices if int(i) not in clean_set],
            dtype=int,
        )

    def _subset_stats(self, indices):
        idx = np.asarray(list(indices), dtype=int)
        count = int(len(idx))
        true_clean = int(self._true_clean_flags(idx).sum()) if count > 0 else 0
        return {
            'count': count,
            'true_clean': true_clean,
            'precision': float(true_clean) / max(count, 1),
        }

    def _build_disagreement_stats(self, fedrn_clean_idx, net2_clean_idx, final_clean_idx):
        fedrn_set = set(int(i) for i in np.asarray(fedrn_clean_idx, dtype=int))
        net2_set = set(int(i) for i in np.asarray(net2_clean_idx, dtype=int))
        final_set = set(int(i) for i in np.asarray(final_clean_idx, dtype=int))

        fedrn_stats = self._build_selection_stats(
            np.asarray(sorted(fedrn_set), dtype=int),
            self._complement_idx(fedrn_set),
        )
        net2_stats = self._build_selection_stats(
            np.asarray(sorted(net2_set), dtype=int),
            self._complement_idx(net2_set),
        )
        final_stats = self._build_selection_stats(
            np.asarray(sorted(final_set), dtype=int),
            self._complement_idx(final_set),
        )

        fedrn_only = self._subset_stats(fedrn_set - net2_set)
        net2_only = self._subset_stats(net2_set - fedrn_set)
        both = self._subset_stats(fedrn_set & net2_set)

        return {
            'fedrn_clean_count': fedrn_stats['pred_clean_count'],
            'net2_clean_count': net2_stats['pred_clean_count'],
            'final_clean_count': final_stats['pred_clean_count'],
            'fedrn_only_clean_count': fedrn_only['count'],
            'net2_only_clean_count': net2_only['count'],
            'both_clean_count': both['count'],
            'fedrn_only_precision': fedrn_only['precision'],
            'net2_only_precision': net2_only['precision'],
            'both_precision': both['precision'],
            'fedrn_precision': fedrn_stats['clean_precision'],
            'net2_precision': net2_stats['clean_precision'],
            'final_precision': final_stats['clean_precision'],
            'fedrn_recall': fedrn_stats['clean_recall'],
            'net2_recall': net2_stats['clean_recall'],
            'final_recall': final_stats['clean_recall'],
        }

    def _class_counts(self, indices):
        labels = (
            self.dataset.targets if hasattr(self.dataset, 'targets')
            else self.dataset.train_labels
        )
        counts = [0 for _ in range(self.args.num_classes)]
        for idx in np.asarray(indices, dtype=int):
            label = int(labels[int(idx)])
            if 0 <= label < self.args.num_classes:
                counts[label] += 1
        return counts

    # ---------- FedRN expertise / arbitrary output ----------
    def set_expertise(self):
        self.net1.eval()
        correct = 0
        n_total = len(self.ldr_eval.dataset)
        with torch.no_grad():
            for inputs, targets, items, idxs in self.ldr_eval:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self._forward_logits(self.net1, inputs, head='global')
                pred = outputs.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.data.view_as(pred)).float().sum().item()
        self.expertise = correct / max(n_total, 1)

    def set_arbitrary_output(self):
        self.net1.eval()
        with torch.no_grad():
            outputs = self._forward_logits(
                self.net1, self.gaussian_noise.to(self.args.device), head='global')
        self.arbitrary_output = outputs.detach()

    # ---------- GMM ----------
    def fit_gmm(self, net, head='global'):
        losses = []
        net.eval()
        with torch.no_grad():
            for inputs, targets, items, idxs in self.ldr_eval:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self._forward_logits(net, inputs, head=head)
                loss = self.CE(outputs, targets)
                losses.append(loss)
        losses = torch.cat(losses).cpu().numpy()
        loss_min, loss_max = losses.min(), losses.max()
        denom = loss_max - loss_min
        if abs(float(denom)) < 1e-12:
            losses = np.zeros_like(losses, dtype=np.float64)
        else:
            losses = (losses - loss_min) / denom
        input_loss = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return prob

    def get_clean_idx(self, prob):
        threshold = self.args.p_threshold
        pred = (prob > threshold)
        pred_clean_idx = pred.nonzero()[0]
        pred_clean_idx = self.data_indices[pred_clean_idx]
        pred_noisy_idx = (~pred).nonzero()[0]
        pred_noisy_idx = self.data_indices[pred_noisy_idx]
        if len(pred_clean_idx) == 0:
            pred_clean_idx = pred_noisy_idx.copy()
            pred_noisy_idx = np.array([], dtype=int)
        return pred_clean_idx, pred_noisy_idx

    def evaluate_net2_holdout(self, net=None):
        if net is None:
            net = self.net2
        net.eval()
        correct, total = 0, 0
        losses = []
        with torch.no_grad():
            for inputs, targets, items, idxs in self.ldr_holdout:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                logits = self._forward_logits(net, inputs)
                loss = self.loss_func(logits, targets)
                pred = logits.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.data.view_as(pred)).float().sum().item()
                total += targets.size(0)
                losses.append(float(loss.item()))
        acc = 100.0 * correct / max(total, 1)
        loss_avg = float(np.mean(losses)) if losses else 0.0
        self.net2_last_holdout_acc = acc
        self.net2_last_holdout_loss = loss_avg
        return acc, loss_avg

    def update_best_net2(self, epoch):
        acc, loss = self.evaluate_net2_holdout(self.net2)
        self.net2_best_updated = False
        if acc > self.net2_best_holdout_acc:
            self.net2_best_holdout_acc = acc
            self.net2_best_epoch = int(epoch)
            self.net2_best_state = copy.deepcopy(self.net2.state_dict())
            self.net2_best_updated = True
        return acc, loss, self.net2_best_updated

    def get_best_net2(self):
        net = copy.deepcopy(self.net2).to(self.args.device)
        if self.net2_best_state is not None:
            net.load_state_dict(self.net2_best_state)
        return net

    def get_net2_bootstrap_clean(self):
        best_net2 = self.get_best_net2()
        prob = self.fit_gmm(best_net2)
        clean_idx, noisy_idx = self.get_clean_idx(prob)
        self.last_net2_stats = self._build_selection_stats(clean_idx, noisy_idx)
        self.last_selection_stats = self.last_net2_stats
        self.last_class_counts = {'net2_bootstrap': self._class_counts(clean_idx)}
        return clean_idx, noisy_idx

    # ---------- 邻居微调 ----------
    def finetune_head(self, neighbor_list, pred_clean_idx, head='global', ft_ep=None):
        """在目标客户端 A 的 scoreA 上临时微调邻居模型的分类头。

        原版 FedRN 只微调邻居 netB1 的分类头。
        微调后的模型只用于本轮 A 的样本判断，不写回邻居本体。
        """
        loader = DataLoader(
            DatasetSplit(self.dataset, pred_clean_idx, real_idx_return=True),
            batch_size=self.args.local_bs, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True,
        )
        if ft_ep is None:
            ft_ep = getattr(self.args, 'neighbor_ft_ep', 1)
        ft_ep = max(int(ft_ep), 1)
        optimizer_list = []
        for neighbor_net in neighbor_list:
            neighbor_net = neighbor_net.to(self.args.device)
            neighbor_net.train()
            head_params = []
            body_params = []
            for name, p in neighbor_net.named_parameters():
                if 'linear' in name:
                    head_params.append(p)
                else:
                    body_params.append(p)
            optimizer = torch.optim.SGD([
                {
                    'params': head_params,
                    'lr': self.args.lr,
                    'momentum': self.args.momentum,
                    'weight_decay': self.args.weight_decay,
                },
                {'params': body_params, 'lr': 0.0},
            ])
            optimizer_list.append(optimizer)

        for _ in range(ft_ep):
            for inputs, targets, items, idxs in loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                for neighbor_net, optimizer in zip(neighbor_list, optimizer_list):
                    neighbor_net.zero_grad()
                    outputs = self._forward_logits(neighbor_net, inputs, head=head)
                    loss = self.loss_func(outputs, targets)
                    loss.backward()
                    optimizer.step()
        return neighbor_list

    # ---------- 训练：在指定子集上训练 net1 ----------
    def train_global_model(self, net_global, train_indices):
        """只训练 net1/global model，用于 warmup 和最终 clean set 更新。"""
        train_loader = DataLoader(
            DatasetSplit(self.dataset, train_indices, real_idx_return=True),
            batch_size=self.args.local_bs, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True,
        )
        net_global.train()
        opt_g = torch.optim.SGD(
            net_global.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        loss_g_list = []
        for epoch in range(self.args.local_ep):
            for inputs, targets, items, idxs in train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                net_global.zero_grad()
                logits_g = self._forward_logits(net_global, inputs, head='global')
                loss_g = self.loss_func(logits_g, targets)
                loss_g.backward()
                opt_g.step()
                loss_g_list.append(loss_g.item())

        self.net1.load_state_dict(net_global.state_dict())
        loss_g_avg = float(np.mean(loss_g_list)) if loss_g_list else 0.0
        return net_global.state_dict(), loss_g_avg

    # ---------- 训练：在指定子集上训练 net2 ----------
    def train_personal_model(self, net_personal, train_indices):
        """只训练 net2/personal model。Speedup 中它本地保留，不上传聚合。"""
        if len(train_indices) == 0:
            return net_personal.state_dict(), 0.0

        train_loader = DataLoader(
            DatasetSplit(self.dataset, train_indices, real_idx_return=True),
            batch_size=self.args.local_bs, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True,
        )
        net_personal.train()
        opt_p = torch.optim.SGD(
            net_personal.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        loss_p_list = []
        local_ep_p = getattr(self.args, 'net2_local_ep', self.args.local_ep)
        for epoch in range(local_ep_p):
            for inputs, targets, items, idxs in train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                net_personal.zero_grad()
                logits_p = self._forward_logits(net_personal, inputs)
                loss_p = self.loss_func(logits_p, targets)
                loss_p.backward()
                opt_p.step()
                loss_p_list.append(loss_p.item())

        self.net2.load_state_dict(net_personal.state_dict())
        self.net2_ready = True
        loss_p_avg = float(np.mean(loss_p_list)) if loss_p_list else 0.0
        return net_personal.state_dict(), loss_p_avg

    # ---------- 训练：在指定子集上同时训练 net1 和 net2（保留备用） ----------
    def train_both_models(self, net_global, net_personal, train_indices):
        """
        在 train_indices（筛选出的干净样本）上同时训练 net1 和 net2。
        net1/net2都使用原版FedRN单头模型。
        """
        train_loader = DataLoader(
            DatasetSplit(self.dataset, train_indices, real_idx_return=True),
            batch_size=self.args.local_bs, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True,
        )
        net_global.train()
        net_personal.train()

        opt_g = torch.optim.SGD(net_global.parameters(),
                                lr=self.args.lr, momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        opt_p = torch.optim.SGD(net_personal.parameters(),
                                lr=self.args.lr, momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)

        loss_g_list, loss_p_list = [], []
        for epoch in range(self.args.local_ep):
            for inputs, targets, items, idxs in train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                # net1 训练
                net_global.zero_grad()
                logits_g = self._forward_logits(net_global, inputs, head='global')
                loss_g = self.loss_func(logits_g, targets)
                loss_g.backward()
                opt_g.step()
                loss_g_list.append(loss_g.item())

                # net2 训练
                net_personal.zero_grad()
                logits_p = self._forward_logits(net_personal, inputs)
                loss_p = self.loss_func(logits_p, targets)
                loss_p.backward()
                opt_p.step()
                loss_p_list.append(loss_p.item())

        # 更新本地副本
        self.net1.load_state_dict(net_global.state_dict())
        self.net2.load_state_dict(net_personal.state_dict())

        loss_g_avg = float(np.mean(loss_g_list)) if loss_g_list else 0.0
        loss_p_avg = float(np.mean(loss_p_list)) if loss_p_list else 0.0

        return net_global.state_dict(), loss_g_avg, net_personal.state_dict(), loss_p_avg

    # ---------- Phase 1: net2 search + net1 short warmup ----------
    def train_phase1(self, net_global, net_personal):
        """Short warmup: net1 trains on noisy train split, net2 searches best checkpoint."""
        self.last_selection_stats = {}
        self.last_scorea_stats = {}
        self.last_fedrn_stats = {}
        self.last_net2_stats = {}
        self.last_class_counts = {}
        w_g, loss_g = self.train_global_model(net_global, self.data_indices)
        rng_state = capture_rng_state()
        try:
            w_p, loss_p = self.train_personal_model(net_personal, self.data_indices)
            self.update_best_net2(getattr(self.args, 'g_epoch', -1))
        finally:
            restore_rng_state(rng_state)
        self.set_expertise()
        self.set_arbitrary_output()
        return w_g, loss_g, w_p, loss_p

    # ---------- Phase 2: use best net2 clean set to bootstrap net1 ----------
    def train_phase_bootstrap(self, net_global, net_personal):
        """Use best net2 checkpoint to select clean samples, then train net1 on them."""
        self.last_scorea_stats = {}
        self.last_fedrn_stats = {}
        if self.net2_best_state is None:
            self.train_personal_model(net_personal, self.data_indices)
            self.update_best_net2(getattr(self.args, 'g_epoch', -1))
        bootstrap_clean_idx, bootstrap_noisy_idx = self.get_net2_bootstrap_clean()
        clean_ratio = self.last_selection_stats['pred_clean_count'] / max(len(self.data_indices), 1)

        w_g, loss_g = self.train_global_model(net_global, bootstrap_clean_idx)

        self.set_expertise()
        self.set_arbitrary_output()

        rng_state = capture_rng_state()
        try:
            if getattr(self.args, 'continue_train_net2_after_search', 0):
                w_p, loss_p = self.train_personal_model(net_personal, bootstrap_clean_idx)
                self.evaluate_net2_holdout(self.net2)
            else:
                w_p = copy.deepcopy(net_personal.state_dict())
                loss_p = 0.0
        finally:
            restore_rng_state(rng_state)

        return w_g, loss_g, w_p, loss_p, clean_ratio

    # ---------- Phase 3: original FedRN selection ----------
    def train_phase2(self, net_global, net_personal, prev_score,
                     neighbor_net1_list, neighbor_score_list):
        """
        Original FedRN phase:
        1) target netA1 uses GMM to get scoreA.
        2) reliable neighbor netB1 is fine-tuned on scoreA.
        3) final clean set is pure FedRN clean set.
        """
        self.last_net2_stats = {}
        # ---- 1. 目标客户端 A 的初始 clean probability，即 scoreA ----
        prob_global = self.fit_gmm(self.net1, head='global')
        pred_clean_idx_1, pred_noisy_idx_1 = self.get_clean_idx(prob_global)
        self.last_scorea_stats = self._build_selection_stats(
            pred_clean_idx_1, pred_noisy_idx_1)

        # ---- 2. 原版 FedRN：邻居 netB1 在 scoreA 上临时微调，再做 GMM ----
        neighbor_net1_list = self.finetune_head(
            neighbor_net1_list, pred_clean_idx_1, head='global',
            ft_ep=getattr(self.args, 'neighbor_ft_ep', 1))

        fedrn_prob_list = [prob_global]
        for neighbor_net1 in neighbor_net1_list:
            prob_b1 = self.fit_gmm(neighbor_net1, head='global')
            fedrn_prob_list.append(prob_b1)

        score_list = [prev_score] + neighbor_score_list
        score_sum = sum(score_list)
        score_list = [s / score_sum for s in score_list]

        fedrn_prob = np.zeros(len(prob_global))
        for prob, sc in zip(fedrn_prob_list, score_list):
            fedrn_prob = np.add(fedrn_prob, np.multiply(prob, sc))

        # ---- 3. Pure FedRN selection ----
        fedrn_clean_idx, fedrn_noisy_idx = self.get_clean_idx(fedrn_prob)
        final_clean_idx = fedrn_clean_idx
        final_noisy_idx = fedrn_noisy_idx

        self.last_fedrn_stats = self._build_selection_stats(
            fedrn_clean_idx, fedrn_noisy_idx)
        self.last_selection_stats = self._build_selection_stats(
            final_clean_idx, final_noisy_idx)
        self.last_disagreement_stats = {}
        self.last_class_counts = {
            'scoreA': self._class_counts(pred_clean_idx_1),
            'fedrn': self._class_counts(fedrn_clean_idx),
            'final': self._class_counts(final_clean_idx),
        }
        clean_ratio = self.last_selection_stats['pred_clean_count'] / max(len(self.data_indices), 1)

        # ---- 4. 用最终 clean set 训练 net1 ----
        w_g, loss_g = self.train_global_model(net_global, final_clean_idx)

        self.set_expertise()
        self.set_arbitrary_output()

        w_p = copy.deepcopy(net_personal.state_dict())
        loss_p = 0.0

        return w_g, loss_g, w_p, loss_p, clean_ratio


# ================================================================
# 主函数
# ================================================================

def main():
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    # 时间和日志
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    noise_str = '_'.join([f'{a}-{b}' for a, b in zip(
        args.group_noise_rate[0::2], args.group_noise_rate[1::2])])
    exp_dir = os.path.join(
        args.save_dir,
        'PFedRN-Net2BootstrapOracle')
    os.makedirs(exp_dir, exist_ok=True)

    log_filename = os.path.join(exp_dir, f'detail_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.txt')
    metrics_filename = os.path.join(exp_dir, f'metrics_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.csv')
    scorea_filename = os.path.join(exp_dir, f'scoreA_stats_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.csv')
    net2_holdout_filename = os.path.join(exp_dir, f'net2_holdout_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.csv')
    selected_net2_summary_filename = os.path.join(exp_dir, f'selected_net2_summary_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.csv')
    net2_bootstrap_filename = os.path.join(exp_dir, f'net2_bootstrap_stats_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.csv')
    class_dist_filename = os.path.join(exp_dir, f'class_distribution_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.csv')

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 与原版FedRN保持一致：先生成相似度用的固定噪声，再做数据和模型初始化。
    gaussian_noise = torch.randn(
        1, args.num_channels, args.img_size, args.img_size
    )

    # 加载数据
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    args.img_size = int(dataset_train[0][0].shape[1])
    train_labels_ref = (
        dataset_train.targets if hasattr(dataset_train, 'targets')
        else dataset_train.train_labels
    )
    original_train_labels = np.array(copy.deepcopy(list(train_labels_ref)), dtype=np.int64)
    labels = np.array(train_labels_ref, dtype=np.int64)

    # non-IID 划分
    if args.iid:
        dict_users_full = sample_iid(labels, args.num_users)
    elif args.partition == 'shard':
        dict_users_full = sample_noniid_shard(
            labels=labels, num_users=args.num_users,
            num_shards=args.num_shards,
            )
    elif args.partition == 'dirichlet':
        dict_users_full = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)
    else:
        raise ValueError(f'Unsupported partition: {args.partition}')

    dict_users, dict_holdout = {}, {}
    for user in range(args.num_users):
        train_idx, holdout_idx = split_client_indices(
            dict_users_full[user],
            train_ratio=args.train_ratio,
            seed=args.seed + user,
        )
        dict_users[user] = train_idx
        dict_holdout[user] = holdout_idx

    # Inject noise only into the 80% train split. The 20% holdout keeps clean labels.
    if sum(args.noise_group_num) != args.num_users:
        raise ValueError('sum(noise_group_num) must equal num_users')

    if len(args.group_noise_rate) == 1:
        args.group_noise_rate = args.group_noise_rate * 2
    group_noise_rate = [
        (args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
        for i in range(len(args.group_noise_rate) // 2)
    ]
    if not (
        len(args.noise_group_num) == len(args.noise_type_lst) == len(group_noise_rate)
    ):
        raise ValueError(
            'noise_group_num, noise_type_lst, and group_noise_rate groups must have the same length'
        )

    user_noise_type_rates = []
    for num_users_in_group, noise_type, (min_r, max_r) in zip(
            args.noise_group_num, args.noise_type_lst, group_noise_rate):
        step = (max_r - min_r) / max(num_users_in_group, 1)
        rates = np.array(range(num_users_in_group)) * step + min_r
        user_noise_type_rates += list(zip([noise_type] * num_users_in_group, rates))

    for user, (noise_type, noise_rate) in enumerate(user_noise_type_rates):
        if noise_type == 'clean':
            continue
        data_indices = list(copy.deepcopy(dict_users[user]))
        if args.noise_seed_mode == 'fedrn':
            random.seed(args.seed)
        elif args.noise_seed_mode in ('per_user', 'user', 'client'):
            random.seed(args.seed + user)
        else:
            raise ValueError(f'Unsupported noise_seed_mode: {args.noise_seed_mode}')
        random.shuffle(data_indices)
        noise_num = int(len(data_indices) * noise_rate)
        for d_idx in data_indices[:noise_num]:
            if hasattr(dataset_train, 'targets'):
                orig = dataset_train.targets[d_idx]
                dataset_train.targets[d_idx] = noisify_label(orig, num_classes=args.num_classes, noise_type=noise_type)
            else:
                orig = dataset_train.train_labels[d_idx]
                dataset_train.train_labels[d_idx] = noisify_label(orig, num_classes=args.num_classes, noise_type=noise_type)

    current_train_labels = np.array(
        dataset_train.targets if hasattr(dataset_train, 'targets')
        else dataset_train.train_labels,
        dtype=np.int64,
    )
    clean_label_mask = (original_train_labels == current_train_labels)

    # Noise statistics only for train split.
    train_indices_all = [idx for user in range(args.num_users) for idx in dict_users[user]]
    clean_count = int(clean_label_mask[train_indices_all].sum())
    noisy_count = int(len(train_indices_all) - clean_count)
    total_count = clean_count + noisy_count
    holdout_count = sum(len(dict_holdout[user]) for user in range(args.num_users))
    print(f'Noisy train split: clean={clean_count}/{total_count} ({clean_count/max(total_count,1):.4f}); '
          f'noisy={noisy_count} ({noisy_count/max(total_count,1):.4f}); '
          f'clean holdout={holdout_count}')
    print(f'Per-client split example: train={len(dict_users[0])}, holdout={len(dict_holdout[0])}')

    # 创建全局模型
    net_glob = get_fedrn_model(args).to(args.device)
    local_weights = LocalModelWeights(
        all_clients=args.all_clients,
        net_glob=net_glob,
        num_users=args.num_users,
        method=args.fed_method,
        dict_users=dict_users,
    )

    # 创建客户端本地对象
    local_objects = []
    for i in range(args.num_users):
        local = LocalUpdateNet1Net2(
            args=args,
            user_idx=i,
            dataset=dataset_train,
            train_idxs=dict_users[i],
            holdout_idxs=dict_holdout[i],
            gaussian_noise=gaussian_noise,
            clean_label_mask=clean_label_mask,
        )
        local_objects.append(local)

    # 测试集 DataLoader
    log_train_loader = DataLoader(
        dataset_train, batch_size=args.bs,
        num_workers=args.num_workers, pin_memory=True)
    log_test_loader = DataLoader(
        dataset_test, batch_size=args.bs,
        num_workers=args.num_workers, pin_memory=True)

    # 日志文件头
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(f'Experiment Start: {timestamp}\n')
        f.write(f'Args: {args}\n')
        f.write('=' * 100 + '\n')
    with open(metrics_filename, 'w', encoding='utf-8') as f:
        f.write(
            'epoch,global_test_acc,global_test_loss,train_loss,clean_ratio,'
            'scoreA_clean_ratio,scoreA_precision,scoreA_recall,'
            'fedrn_clean_ratio,fedrn_precision,fedrn_recall,'
            'net2_bootstrap_clean_ratio,net2_bootstrap_precision,net2_bootstrap_recall,'
            'selection_precision,selection_recall,false_clean,missed_clean,'
            'phase,selected_net2_best_epoch_avg,selected_net2_best_holdout_acc_avg,'
            'selected_net2_current_holdout_acc_avg\n'
        )
    with open(scorea_filename, 'w', encoding='utf-8') as f:
        f.write(
            'epoch,client_id,scoreA_count,scoreA_precision,scoreA_recall,'
            'scoreA_false_clean,scoreA_missed_clean,true_clean_count,total_count\n'
        )
    with open(net2_holdout_filename, 'w', encoding='utf-8') as f:
        f.write(
            'epoch,client_id,holdout_acc,holdout_loss,best_epoch,best_holdout_acc,best_updated,'
            'train_count,holdout_count\n'
        )
    with open(selected_net2_summary_filename, 'w', encoding='utf-8') as f:
        f.write(
            'epoch,phase,selected_client_count,selected_net2_current_holdout_acc_avg,'
            'selected_net2_best_holdout_acc_avg,selected_net2_best_epoch_avg\n'
        )
    with open(net2_bootstrap_filename, 'w', encoding='utf-8') as f:
        f.write(
            'epoch,client_id,net2_clean_count,net2_precision,net2_recall,'
            'net2_false_clean,net2_missed_clean,true_clean_count,total_count,'
            'best_epoch,best_holdout_acc\n'
        )
    with open(class_dist_filename, 'w', encoding='utf-8') as f:
        f.write('epoch,client_id,set_name,' + ','.join([f'class_{i}' for i in range(args.num_classes)]) + '\n')

    all_client_ids = list(range(args.num_users))
    print(f'\n=== PFedRN Net2 Bootstrap Oracle ===')
    print(
        f'net2_search_epochs={args.net2_search_epochs}, '
        f'fedrn_start_epoch={args.fedrn_start_epoch}, train_ratio={args.train_ratio}\n'
    )

    start_time = time.time()

    for epoch in range(args.epochs):
        # 学习率衰减
        if (epoch + 1) in args.schedule:
            args.lr *= args.lr_decay
        args.g_epoch = epoch
        if epoch < args.net2_search_epochs:
            phase_name = 'net2_search'
        elif epoch < args.fedrn_start_epoch:
            phase_name = 'net2_bootstrap'
        else:
            phase_name = 'fedrn_stable'

        # 选客户端
        m = max(int(args.frac * args.num_users), 1)
        selected_clients = np.random.choice(range(args.num_users), m, replace=False)

        client_losses = []
        client_clean = []
        client_best_epochs = []
        client_best_accs = []
        client_last_accs = []

        for client_idx in selected_clients:
            local = local_objects[client_idx]
            local.args = args

            net_global = copy.deepcopy(net_glob).to(args.device)
            net_personal = copy.deepcopy(local.net2).to(args.device)

            if epoch < args.net2_search_epochs:
                # ---- Stage 1: short net1 warmup + net2 best checkpoint search ----
                w, loss, w_personal, p_loss = local.train_phase1(net_global, net_personal)
                clean_ratio_local = 1.0
            elif epoch < args.fedrn_start_epoch:
                # ---- Stage 2: use best net2 clean set directly to bootstrap net1 ----
                w, loss, w_personal, p_loss, clean_ratio_local = local.train_phase_bootstrap(
                    net_global, net_personal)
            else:
                # ---- Stage 3: original FedRN selection ----
                neighbor_pool = all_client_ids
                sim_list, exp_list = [], []
                cosine_sim = nn.CosineSimilarity(dim=1)
                for u in neighbor_pool:
                    sim = cosine_sim(
                        local.arbitrary_output.view(1, -1).to(args.device),
                        local_objects[u].arbitrary_output.view(1, -1).to(args.device),
                    ).item()
                    sim_list.append(sim)
                    exp_list.append(local_objects[u].expertise)

                sim_norm = safe_minmax(sim_list)
                exp_norm = safe_minmax(exp_list)

                local_pos = neighbor_pool.index(client_idx)
                w_alpha = args.w_alpha
                prev_score = w_alpha * exp_norm[local_pos] + (1.0 - w_alpha) * 1.0

                scores = []
                for pos, (e_score, s_score) in enumerate(zip(exp_norm, sim_norm)):
                    gid = neighbor_pool[pos]
                    if gid == client_idx:
                        continue
                    score = w_alpha * e_score + (1.0 - w_alpha) * s_score
                    scores.append((score, gid))
                scores.sort(key=lambda x: x[0], reverse=True)

                neighbor_net1_list, neighbor_score_list = [], []
                for score, n_id in scores[:args.num_neighbors]:
                    neighbor_net1_list.append(
                        copy.deepcopy(local_objects[n_id].net1).to(args.device))
                    neighbor_score_list.append(score)

                w, loss, w_personal, p_loss, clean_ratio_local = local.train_phase2(
                    net_global, net_personal, prev_score,
                    neighbor_net1_list, neighbor_score_list)

            local_weights.update(int(client_idx), w)
            client_losses.append(loss)
            client_clean.append(clean_ratio_local)
            client_best_epochs.append(local.net2_best_epoch)
            client_best_accs.append(local.net2_best_holdout_acc)
            client_last_accs.append(local.net2_last_holdout_acc)

        # ---- 聚合 ----
        w_glob = local_weights.average()
        net_glob.load_state_dict(w_glob, strict=False)
        local_weights.init()
        # ---- 评估 ----
        global_test_acc, global_test_loss = test_img(net_glob, log_test_loader, args)
        train_loss = float(np.mean(client_losses)) if client_losses else 0.0
        clean_ratio_avg = float(np.mean(client_clean)) if client_clean else 0.0
        best_epoch_avg = float(np.mean([v for v in client_best_epochs if v >= 0])) if any(v >= 0 for v in client_best_epochs) else -1.0
        best_acc_avg = float(np.mean([v for v in client_best_accs if v >= 0])) if any(v >= 0 for v in client_best_accs) else -1.0
        last_acc_avg = float(np.mean([v for v in client_last_accs if v >= 0])) if any(v >= 0 for v in client_last_accs) else -1.0

        selected_stats = [
            local_objects[int(c_idx)].last_selection_stats
            for c_idx in selected_clients
            if local_objects[int(c_idx)].last_selection_stats
        ]
        scorea_stats_list = [
            local_objects[int(c_idx)].last_scorea_stats
            for c_idx in selected_clients
            if local_objects[int(c_idx)].last_scorea_stats
        ]
        fedrn_stats_list = [
            local_objects[int(c_idx)].last_fedrn_stats
            for c_idx in selected_clients
            if local_objects[int(c_idx)].last_fedrn_stats
        ]
        net2_stats_list = [
            local_objects[int(c_idx)].last_net2_stats
            for c_idx in selected_clients
            if local_objects[int(c_idx)].last_net2_stats
        ]

        def summarize_stats(stats_list):
            if not stats_list:
                return -1.0, -1.0, -1.0
            total = sum(s.get('total_count', 0) for s in stats_list)
            pred_clean = sum(s.get('pred_clean_count', 0) for s in stats_list)
            actual_clean = sum(s.get('actual_train_clean_count', 0) for s in stats_list)
            true_clean = sum(s.get('true_clean_count', 0) for s in stats_list)
            return (
                float(pred_clean) / max(total, 1),
                float(actual_clean) / max(pred_clean, 1),
                float(actual_clean) / max(true_clean, 1),
            )

        scorea_clean_ratio, scorea_precision, scorea_recall = summarize_stats(scorea_stats_list)
        fedrn_clean_ratio, fedrn_precision, fedrn_recall = summarize_stats(fedrn_stats_list)
        net2_clean_ratio, net2_precision, net2_recall = summarize_stats(net2_stats_list)

        pred_clean_total = sum(s.get('pred_clean_count', 0) for s in selected_stats)
        actual_clean_total = sum(s.get('actual_train_clean_count', 0) for s in selected_stats)
        true_clean_total = sum(s.get('true_clean_count', 0) for s in selected_stats)
        false_clean_total = sum(s.get('false_clean_count', 0) for s in selected_stats)
        missed_clean_total = sum(s.get('missed_clean_count', 0) for s in selected_stats)
        selection_precision = (
            actual_clean_total / max(pred_clean_total, 1) if selected_stats else -1.0
        )
        selection_recall = (
            actual_clean_total / max(true_clean_total, 1) if selected_stats else -1.0
        )

        # 日志输出
        stats_summary = ''
        if args.log_client_stats and len(selected_clients) > 0:
            stats_parts = []
            for c_idx in selected_clients:
                s = local_objects[c_idx].last_selection_stats
                if s:
                    stats_parts.append(
                        f'[Client {c_idx:03d}] '
                        f'clean={s.get("pred_clean_count","?"):}/{s.get("total_count","?"):} '
                        f'prec={s.get("clean_precision",0):.3f} '
                        f'recall={s.get("clean_recall",0):.3f} '
                        f'missed={s.get("missed_clean_count",0):} '
                        f'false={s.get("false_clean_count",0):}'
                    )
            if stats_parts:
                stats_summary = '\n  ' + '\n  '.join(stats_parts) if len(stats_parts) <= 10 else f'\n  (showing {len(stats_parts)} clients)'

        log_line = (
            f'\n==== Round {epoch:3d} ====\n'
            f'Global Test Acc: {global_test_acc:.2f}% | Loss: {global_test_loss:.4f} | '
            f'Train Loss: {train_loss:.4f} | Clean Ratio: {clean_ratio_avg:.3f} | '
            f'Phase: {phase_name} | Net2 Best Epoch Avg: {best_epoch_avg:.1f} | '
            f'Selected Net2 Acc Avg: {last_acc_avg:.2f}% | '
            f'Best Net2 Acc Avg: {best_acc_avg:.2f}%'
        )
        print(log_line)
        if stats_summary:
            print(stats_summary)

        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')
            if stats_summary:
                f.write(stats_summary + '\n')

        with open(metrics_filename, 'a', encoding='utf-8') as f:
            f.write(
                f'{epoch},{global_test_acc:.6f},{global_test_loss:.6f},{train_loss:.6f},'
                f'{clean_ratio_avg:.6f},{scorea_clean_ratio:.6f},{scorea_precision:.6f},{scorea_recall:.6f},'
                f'{fedrn_clean_ratio:.6f},{fedrn_precision:.6f},{fedrn_recall:.6f},'
                f'{net2_clean_ratio:.6f},{net2_precision:.6f},{net2_recall:.6f},'
                f'{selection_precision:.6f},{selection_recall:.6f},'
                f'{false_clean_total},{missed_clean_total},'
                f'{phase_name},{best_epoch_avg:.6f},{best_acc_avg:.6f},{last_acc_avg:.6f}\n'
            )

        with open(net2_holdout_filename, 'a', encoding='utf-8') as f:
            for c_idx in selected_clients:
                local_obj = local_objects[int(c_idx)]
                f.write(
                    f'{epoch},{int(c_idx)},'
                    f'{local_obj.net2_last_holdout_acc:.6f},'
                    f'{local_obj.net2_last_holdout_loss:.6f},'
                    f'{local_obj.net2_best_epoch},'
                    f'{local_obj.net2_best_holdout_acc:.6f},'
                    f'{int(local_obj.net2_best_updated)},'
                    f'{len(local_obj.data_indices)},{len(local_obj.holdout_indices)}\n'
                )
        with open(selected_net2_summary_filename, 'a', encoding='utf-8') as f:
            f.write(
                f'{epoch},{phase_name},{len(selected_clients)},'
                f'{last_acc_avg:.6f},{best_acc_avg:.6f},{best_epoch_avg:.6f}\n'
            )

        if epoch >= args.net2_search_epochs:
            with open(scorea_filename, 'a', encoding='utf-8') as f:
                for c_idx in selected_clients:
                    s = local_objects[int(c_idx)].last_scorea_stats
                    if s:
                        f.write(
                            f'{epoch},{int(c_idx)},{s["pred_clean_count"]},'
                            f'{s["clean_precision"]:.6f},{s["clean_recall"]:.6f},'
                            f'{s["false_clean_count"]},{s["missed_clean_count"]},'
                            f'{s["true_clean_count"]},{s["total_count"]}\n'
                        )
            with open(net2_bootstrap_filename, 'a', encoding='utf-8') as f:
                for c_idx in selected_clients:
                    local_obj = local_objects[int(c_idx)]
                    s = local_obj.last_net2_stats
                    if s:
                        f.write(
                            f'{epoch},{int(c_idx)},{s["pred_clean_count"]},'
                            f'{s["clean_precision"]:.6f},{s["clean_recall"]:.6f},'
                            f'{s["false_clean_count"]},{s["missed_clean_count"]},'
                            f'{s["true_clean_count"]},{s["total_count"]},'
                            f'{local_obj.net2_best_epoch},{local_obj.net2_best_holdout_acc:.6f}\n'
                        )
            with open(class_dist_filename, 'a', encoding='utf-8') as f:
                for c_idx in selected_clients:
                    counts_by_set = local_objects[int(c_idx)].last_class_counts
                    for set_name, counts in counts_by_set.items():
                        f.write(
                            f'{epoch},{int(c_idx)},{set_name},' +
                            ','.join(str(int(v)) for v in counts) + '\n'
                        )

    total_time = time.time() - start_time
    print(f'\nTotal time: {total_time:.1f}s')
    print(f'Log file: {log_filename}')
    print(f'Metrics file: {metrics_filename}')
    print(f'ScoreA stats file: {scorea_filename}')
    print(f'Net2 holdout file: {net2_holdout_filename}')
    print(f'Selected net2 summary file: {selected_net2_summary_filename}')
    print(f'Net2 bootstrap stats file: {net2_bootstrap_filename}')
    print(f'Class distribution file: {class_dist_filename}')


if __name__ == '__main__':
    main()
