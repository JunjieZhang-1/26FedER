#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PFedRN PlanA: Neighbor Net1/Net2 Fine-tuning
============================================

======== 核心思路 ========

场景：联邦学习 + 标签噪声（CIFAR-10, 100 clients, non-IID shards）

两个模型：
  1. net1 = 全局模型（参与 FedAvg 聚合 + 标准 FedRN 邻居+微调判断）
  2. net2 = 本地个性化模型（不上传、不聚合，只在该客户端本地保留和训练）

判断是否干净（PlanA）：
  warmup 阶段（默认 0-99 轮）：只训练 net1，net2 不做本地训练
  net2 训练观察阶段（默认 100-199 轮）：
    - 目标客户端 A 先用 netA1 做 GMM 得到 scoreA
    - A 的 net2 从这一轮开始，只用 scoreA 训练
    - net2 暂时不参与概率判断，判断仍然等价于原 FedRN
  net2 参与阶段（默认 200 轮以后）：
    - A 的 net2 继续只用 scoreA 训练
    - 对每个可靠邻居 B：
        复制 netB1，在 scoreA 上临时微调分类头，得到 p_B1_ft
        如果 netB2 已训练，则复制 netB2，在 scoreA 上临时微调分类头，得到 p_B2_ft
        p_B_ft = neighbor_net1_weight × p_B1_ft + neighbor_net2_weight × p_B2_ft
    - 最终概率 = 目标 A 的 p_A + 各邻居 p_B_ft 按 FedRN 可靠性分数加权融合
    - 根据 final_prob > p_threshold 判断是否干净

训练：
  - net2 前 warmup_epochs 轮不做本地训练
  - warmup 后，net2 只用 net1 初筛得到的 scoreA 训练
  - net1 上传到服务器做 FedAvg，net2 保留本地
  - 聚合后只同步 net2 的特征层，保留 net2 自己的 linear 分类头

用法示例：
  python main_pfedrn_planA_neighbor_net2_ft.py
    --dataset cifar10 --epochs 500 --num_users 100 --frac 0.1
    --local_ep 5 --local_bs 50 --num_shards 200
    --group_noise_rate 0 0.8 --noise_group_num 100
    --warmup_epochs 100 --num_neighbors 2 --w_alpha 0.6
    --neighbor_net1_weight 0.7 --neighbor_net2_weight 0.3
    --net2_start_epoch 200
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

from utils.options_pfedrn_planA_neighbor_net2_ft import args_parser
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
    """返回模型logits；当前PlanA中net1/net2均为原版FedRN单头模型。"""
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

    def __init__(self, args, dataset=None, user_idx=None, idxs=None,
                 gaussian_noise=None, clean_label_mask=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = list(idxs)
        self.data_indices = np.array(idxs)
        self.gaussian_noise = gaussian_noise
        self.clean_label_mask = clean_label_mask

        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.loss_func = nn.CrossEntropyLoss()

        # net1/net2都使用原版FedRN单头模型；net2只是本地保留，不参与聚合。
        self.net1 = get_fedrn_model(args).to(args.device)
        self.net2 = get_fedrn_model(args).to(args.device)

        # DataLoader 用于评估（不做 shuffle，返回真实下标）
        self.ldr_eval = DataLoader(
            DatasetSplit(dataset, idxs, real_idx_return=True),
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

    # ---------- 邻居微调 ----------
    def finetune_head(self, neighbor_list, pred_clean_idx, head='global', ft_ep=None):
        """在目标客户端 A 的 scoreA 上临时微调邻居模型的分类头。

        原版 FedRN 只微调邻居 netB1 的分类头；PlanA 会分别微调：
          - netB1: 原版单头分类头
          - netB2: 原版单头分类头
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
        """只训练 net2/personal model。PlanA 中它只在 warmup 后用 scoreA 训练。"""
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

    # ---------- Phase 1: warmup ----------
    def train_phase1(self, net_global, net_personal):
        """warmup：只用全部数据训练 net1；net2 不训练。"""
        w_g, loss_g = self.train_global_model(net_global, self.data_indices)
        w_p = copy.deepcopy(net_personal.state_dict())
        loss_p = 0.0
        self.set_expertise()
        self.set_arbitrary_output()
        return w_g, loss_g, w_p, loss_p

    # ---------- Phase 2: PlanA 邻居 netB1/netB2 微调 + 训练 ----------
    def train_phase2(self, net_global, net_personal, prev_score,
                     neighbor_net1_list, neighbor_net2_list, neighbor_score_list,
                     neighbor_net2_ready_list=None, use_neighbor_net2=True):
        """
        PlanA:
        1) 目标客户端 A 的 netA1 先做 GMM，得到 scoreA。
        2) 先按 FedRN 流程完成干净样本判断和 net1 更新。
        3) A 的 net2 从 warmup 后才开始，用 scoreA 训练；训练放在 net1 更新后，避免扰动本轮FedRN随机状态。
        4) 在 net2_start_epoch 之前，net2 只训练不参与概率判断。
        5) 每个可靠邻居 B：
             netB1 在 scoreA 上微调分类头后做 GMM。
             如果 netB2 已经用自身 score 训练过，则 netB2 在 scoreA 上微调分类头后做 GMM。
             如果 netB2 尚未训练，则该邻居只使用 netB1，避免随机/未训练net2污染判断。
             两者融合为该邻居的 p_B_ft。
        6) 按 FedRN 原可靠性分数融合 p_A 和各 p_B_ft。
        7) 根据 final_prob 筛选干净样本。
        8) 用最终 clean set 训练 A 的 net1。
        """
        # ---- 1. 目标客户端 A 的初始 clean probability，即 scoreA ----
        prob_global = self.fit_gmm(self.net1, head='global')
        pred_clean_idx_1, pred_noisy_idx_1 = self.get_clean_idx(prob_global)
        self.last_scorea_stats = self._build_selection_stats(
            pred_clean_idx_1, pred_noisy_idx_1)

        # ---- 2. 邻居 netB1/netB2 都在 scoreA 上临时微调，再分别做 GMM ----
        neighbor_net1_list = self.finetune_head(
            neighbor_net1_list, pred_clean_idx_1, head='global',
            ft_ep=getattr(self.args, 'neighbor_ft_ep', 1))
        if neighbor_net2_ready_list is None:
            neighbor_net2_ready_list = [True] * len(neighbor_net2_list)
        if not use_neighbor_net2:
            neighbor_net2_ready_list = [False] * len(neighbor_net2_ready_list)
        ready_neighbor_net2 = [
            net2 for net2, ready in zip(neighbor_net2_list, neighbor_net2_ready_list)
            if ready
        ]
        if len(ready_neighbor_net2) > 0:
            ready_neighbor_net2 = self.finetune_head(
                ready_neighbor_net2, pred_clean_idx_1,
                ft_ep=getattr(self.args, 'net2_ft_ep', 1))

        nb1_w = getattr(self.args, 'neighbor_net1_weight', 0.7)
        nb2_w = getattr(self.args, 'neighbor_net2_weight', 0.3)
        nb_total = nb1_w + nb2_w
        nb1_w = nb1_w / (nb_total + 1e-8)
        nb2_w = nb2_w / (nb_total + 1e-8)

        prob_list = [prob_global]
        fedrn_prob_list = [prob_global]
        net2_prob_list = [prob_global]
        ready_pos = 0
        for neighbor_net1, net2_ready in zip(neighbor_net1_list, neighbor_net2_ready_list):
            prob_b1 = self.fit_gmm(neighbor_net1, head='global')
            fedrn_prob_list.append(prob_b1)
            if net2_ready and ready_pos < len(ready_neighbor_net2):
                prob_b2 = self.fit_gmm(ready_neighbor_net2[ready_pos])
                prob_b = nb1_w * prob_b1 + nb2_w * prob_b2
                net2_prob_list.append(prob_b2)
                ready_pos += 1
            else:
                prob_b = prob_b1
                net2_prob_list.append(prob_b1)
            prob_list.append(prob_b)

        score_list = [prev_score] + neighbor_score_list
        score_sum = sum(score_list)
        score_list = [s / score_sum for s in score_list]

        final_prob = np.zeros(len(prob_global))
        fedrn_prob = np.zeros(len(prob_global))
        net2_prob = np.zeros(len(prob_global))
        for prob, sc in zip(prob_list, score_list):
            final_prob = np.add(final_prob, np.multiply(prob, sc))
        for prob, sc in zip(fedrn_prob_list, score_list):
            fedrn_prob = np.add(fedrn_prob, np.multiply(prob, sc))
        for prob, sc in zip(net2_prob_list, score_list):
            net2_prob = np.add(net2_prob, np.multiply(prob, sc))

        # ---- 3. 筛选 ----
        fedrn_clean_idx, fedrn_noisy_idx = self.get_clean_idx(fedrn_prob)
        net2_clean_idx, net2_noisy_idx = self.get_clean_idx(net2_prob)
        final_clean_idx, final_noisy_idx = self.get_clean_idx(final_prob)
        self.last_fedrn_stats = self._build_selection_stats(
            fedrn_clean_idx, fedrn_noisy_idx)
        self.last_selection_stats = self._build_selection_stats(
            final_clean_idx, final_noisy_idx)
        if use_neighbor_net2:
            self.last_net2_stats = self._build_selection_stats(
                net2_clean_idx, net2_noisy_idx)
            self.last_disagreement_stats = self._build_disagreement_stats(
                fedrn_clean_idx, net2_clean_idx, final_clean_idx)
        else:
            self.last_net2_stats = {}
            self.last_disagreement_stats = {}
        self.last_class_counts = {
            'scoreA': self._class_counts(pred_clean_idx_1),
            'fedrn': self._class_counts(fedrn_clean_idx),
            'final': self._class_counts(final_clean_idx),
        }
        if use_neighbor_net2:
            self.last_class_counts['net2'] = self._class_counts(net2_clean_idx)
        clean_ratio = self.last_selection_stats['pred_clean_count'] / max(len(self.data_indices), 1)

        # ---- 4. 用最终 clean set 训练 net1 ----
        w_g, loss_g = self.train_global_model(net_global, final_clean_idx)

        self.set_expertise()
        self.set_arbitrary_output()

        # ---- 5. 本轮FedRN/net1更新完成后，再用scoreA训练net2，避免影响本轮net1路径 ----
        rng_state = capture_rng_state()
        try:
            w_p, loss_p = self.train_personal_model(net_personal, pred_clean_idx_1)
        finally:
            restore_rng_state(rng_state)

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
        'Pfedrn_planA_neighbor_net2')
    os.makedirs(exp_dir, exist_ok=True)

    log_filename = os.path.join(exp_dir, f'detail_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.txt')
    metrics_filename = os.path.join(exp_dir, f'metrics_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.csv')
    scorea_filename = os.path.join(exp_dir, f'scoreA_stats_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.csv')
    disagreement_filename = os.path.join(exp_dir, f'disagreement_{args.exp_method}{args.epochs}_{args.dataset}_{args.num_users}_{noise_str}_{timestamp}.csv')
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
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == 'shard':
        dict_users = sample_noniid_shard(
            labels=labels, num_users=args.num_users,
            num_shards=args.num_shards,
            )
    elif args.partition == 'dirichlet':
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)
    else:
        raise ValueError(f'Unsupported partition: {args.partition}')

    # 注入噪声
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

    # 统计噪声率
    train_indices_all = [idx for user in range(args.num_users) for idx in dict_users[user]]
    clean_count = int(clean_label_mask[train_indices_all].sum())
    noisy_count = int(len(train_indices_all) - clean_count)
    total_count = clean_count + noisy_count
    print(f'Noisy train: clean={clean_count}/{total_count} ({clean_count/max(total_count,1):.4f}); '
          f'noisy={noisy_count} ({noisy_count/max(total_count,1):.4f})')

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
            idxs=dict_users[i],
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
            'net2_clean_ratio,net2_precision,net2_recall,'
            'selection_precision,selection_recall,false_clean,missed_clean,'
            'neighbor_net2_used_ratio,neighbor_ft_ep,net2_ft_ep\n'
        )
    with open(scorea_filename, 'w', encoding='utf-8') as f:
        f.write(
            'epoch,client_id,scoreA_count,scoreA_precision,scoreA_recall,'
            'scoreA_false_clean,scoreA_missed_clean,true_clean_count,total_count\n'
        )
    with open(disagreement_filename, 'w', encoding='utf-8') as f:
        f.write(
            'epoch,client_id,fedrn_clean_count,net2_clean_count,final_clean_count,'
            'fedrn_only_clean_count,net2_only_clean_count,both_clean_count,'
            'fedrn_only_precision,net2_only_precision,both_precision,'
            'fedrn_precision,net2_precision,final_precision,'
            'fedrn_recall,net2_recall,final_recall\n'
        )
    with open(class_dist_filename, 'w', encoding='utf-8') as f:
        f.write('epoch,client_id,set_name,' + ','.join([f'class_{i}' for i in range(args.num_classes)]) + '\n')

    all_client_ids = list(range(args.num_users))
    print(f'\n=== PFedRN PlanA: Neighbor Net1/Net2 Fine-tuning ===')
    print(
        f'neighbor_net1_weight={args.neighbor_net1_weight}, '
        f'neighbor_net2_weight={args.neighbor_net2_weight}\n'
    )

    start_time = time.time()

    for epoch in range(args.epochs):
        # 学习率衰减
        if (epoch + 1) in args.schedule:
            args.lr *= args.lr_decay
        args.g_epoch = epoch

        # 选客户端
        m = max(int(args.frac * args.num_users), 1)
        selected_clients = np.random.choice(range(args.num_users), m, replace=False)

        client_losses = []
        client_clean = []
        client_neighbor_net2_used = []

        for client_idx in selected_clients:
            local = local_objects[client_idx]
            local.args = args

            net_global = copy.deepcopy(net_glob).to(args.device)
            net_personal = copy.deepcopy(local.net2).to(args.device)

            if epoch < args.warmup_epochs:
                # ---- warmup ----
                w, loss, w_personal, p_loss = local.train_phase1(net_global, net_personal)
                clean_ratio_local = 1.0
            else:
                # ---- phase2: 双模型判断 ----
                # 邻居池
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

                neighbor_net1_list, neighbor_net2_list, neighbor_score_list = [], [], []
                neighbor_net2_ready_list = []
                for score, n_id in scores[:args.num_neighbors]:
                    neighbor_net1_list.append(
                        copy.deepcopy(local_objects[n_id].net1).to(args.device))
                    neighbor_net2_list.append(
                        copy.deepcopy(local_objects[n_id].net2).to(args.device))
                    neighbor_net2_ready_list.append(bool(local_objects[n_id].net2_ready))
                    neighbor_score_list.append(score)

                use_neighbor_net2 = epoch >= args.net2_start_epoch

                w, loss, w_personal, p_loss, clean_ratio_local = local.train_phase2(
                    net_global, net_personal, prev_score,
                    neighbor_net1_list, neighbor_net2_list, neighbor_score_list,
                    neighbor_net2_ready_list, use_neighbor_net2=use_neighbor_net2)
                client_neighbor_net2_used.append(
                    sum(1 for ready in neighbor_net2_ready_list if ready and use_neighbor_net2)
                    / max(len(neighbor_net2_ready_list), 1)
                )

            local_weights.update(int(client_idx), w)
            client_losses.append(loss)
            client_clean.append(clean_ratio_local)

        # ---- 聚合 ----
        w_glob = local_weights.average()
        net_glob.load_state_dict(w_glob, strict=False)
        local_weights.init()
        # 原FedRN保留每个客户端自己的net1用于GMM/邻居判断；这里只同步net2的backbone。
        for local in local_objects:
            sync_global_to_client(local.net2, w_glob, mode='personal')

        # ---- 评估 ----
        global_test_acc, global_test_loss = test_img(net_glob, log_test_loader, args)
        train_loss = float(np.mean(client_losses)) if client_losses else 0.0
        clean_ratio_avg = float(np.mean(client_clean)) if client_clean else 0.0
        neighbor_net2_used_avg = (
            float(np.mean(client_neighbor_net2_used)) if client_neighbor_net2_used else -1.0
        )

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
            f'Neighbor Net2 Used: {neighbor_net2_used_avg:.3f}'
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
                f'{false_clean_total},{missed_clean_total},{neighbor_net2_used_avg:.6f},'
                f'{int(getattr(args, "neighbor_ft_ep", 1))},{int(getattr(args, "net2_ft_ep", 1))}\n'
            )

        if epoch >= args.warmup_epochs:
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
            with open(disagreement_filename, 'a', encoding='utf-8') as f:
                for c_idx in selected_clients:
                    s = local_objects[int(c_idx)].last_disagreement_stats
                    if s:
                        f.write(
                            f'{epoch},{int(c_idx)},{s["fedrn_clean_count"]},{s["net2_clean_count"]},{s["final_clean_count"]},'
                            f'{s["fedrn_only_clean_count"]},{s["net2_only_clean_count"]},{s["both_clean_count"]},'
                            f'{s["fedrn_only_precision"]:.6f},{s["net2_only_precision"]:.6f},{s["both_precision"]:.6f},'
                            f'{s["fedrn_precision"]:.6f},{s["net2_precision"]:.6f},{s["final_precision"]:.6f},'
                            f'{s["fedrn_recall"]:.6f},{s["net2_recall"]:.6f},{s["final_recall"]:.6f}\n'
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
    print(f'Disagreement file: {disagreement_filename}')
    print(f'Class distribution file: {class_dist_filename}')


if __name__ == '__main__':
    main()
