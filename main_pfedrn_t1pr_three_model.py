"""
Three-model PFedRN-T1PR main.

This file implements the structure you described:
1) net1: global/main FedRN model, uploaded and aggregated as before;
2) net2: recovery peer model, trained with core clean + recovered samples;
3) net3: personalized local model, kept local and used as an independent PFL probability branch.

Compared with the previous two-model PFedRN-T1PR:
- net2 is no longer shared by recovery and personalization;
- net3 is introduced as an independent personalized model;
- net1/net2 support mutual-learning regularization;
- by default, recovered samples train net2 but do NOT directly train uploaded net1.

Main probability:
    p_rn    = FedRN/top-1 reliable-neighbor probability from net1
    p_rec   = recovery peer probability from net2
    p_local = personalized probability from net3 local head

    final_prob = w_rn * p_rn + w_rec * p_rec + w_pfl * p_local

The script keeps detailed statistics:
MissedClean / FalseClean / TotalWrong.
"""

import os
import sys
import copy
import time
import random
import datetime

# =============================================================================
# Windows DLL/PATH fix must run before importing numpy/torch.
# =============================================================================
if sys.platform.startswith("win"):
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
    dll_dirs = [
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "DLLs"),
        os.path.join(conda_prefix, "libs"),
        r"C:\Users\25839\.conda\envs\improve-FedRN-main\Library\bin",
    ]
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir) and dll_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

from utils import load_dataset
try:
    from utils.options_fedrn_t1pr_three_model import args_parser
except Exception:
    from utils.options_fedrn_t1pr import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label
from models.nets import get_model
from models.test import test_img
from models.update import LocalUpdatePFedRN, DatasetSplit


class DatasetSplitPlain(Dataset):
    """For evaluation loaders that only need image, label."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[int(item)]]
        return image, label


def get_logits(output, head="global"):
    """兼容单输出模型和双头 CNN4Conv。"""
    if isinstance(output, (tuple, list)):
        if head == "local" and len(output) > 1:
            return output[1]
        return output[0]
    return output


def is_local_head_key(key):
    return "fc_local" in key


def is_head_key(key):
    return (
        ("linear" in key)
        or ("fc_global" in key)
        or ("fc_local" in key)
        or key.startswith("fc2")
        or ("classifier" in key)
    )


def fedavg(weights, weight_list, skip_local_head=True):
    if len(weights) == 0:
        return None
    total = float(sum(weight_list))
    w_avg = copy.deepcopy(weights[0])

    for k in w_avg.keys():
        if skip_local_head and is_local_head_key(k):
            continue
        w_avg[k] = w_avg[k] * weight_list[0]
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k] * weight_list[i]
        w_avg[k] = torch.div(w_avg[k], total)
    return w_avg


def sync_global_to_client(local_model, global_state, mode="global"):
    """
    mode='global': 同步全局模型，保留 fc_local。
    mode='recovery': recovery model 吸收全局 backbone，保留自己的分类头。
    """
    state = local_model.state_dict()
    for k, v in global_state.items():
        if k not in state:
            continue
        if mode == "global":
            if is_local_head_key(k):
                continue
            state[k] = v.clone()
        elif mode == "recovery":
            if is_head_key(k):
                continue
            state[k] = v.clone()
    local_model.load_state_dict(state)


class LocalUpdateThreeModelT1PR(LocalUpdatePFedRN):
    """
    Three-model PFedRN-T1PR local update.

    net1: global/main model, uploaded and aggregated by FedAvg.
    net2: recovery peer model, not uploaded; trained on core + recovered samples.
    net3: personalized local model, not uploaded; local head provides p_local.

    The goal is to separate recovery and personalization:
      p_rn    = FedRN/top-1 reliable-neighbor clean probability from net1
      p_rec   = recovery peer clean probability from net2
      p_local = personalized clean probability from net3 local head

    final_prob = w_rn * p_rn + w_rec * p_rec + w_pfl * p_local
    """

    def _safe_indices(self, arr):
        arr = np.array(arr, dtype=np.int64)
        if len(arr) == 0:
            return arr
        return np.unique(arr)

    def _select_core_and_noisy_by_prob(self, prob, threshold):
        prob = np.asarray(prob)
        core_mask = prob > threshold
        noisy_mask = ~core_mask

        if core_mask.sum() == 0:
            keep_n = max(1, int(0.1 * len(prob)))
            top_pos = np.argsort(prob)[-keep_n:]
            core_mask[top_pos] = True
            noisy_mask = ~core_mask

        core_idx = self.data_indices[core_mask]
        noisy_idx = self.data_indices[noisy_mask]
        return self._safe_indices(core_idx), self._safe_indices(noisy_idx), core_mask, noisy_mask

    def _three_weights(self):
        """Return normalized weights for p_rn, p_rec, p_local."""
        w_rn = float(getattr(self.args, "t1pr_rn_weight", 0.80))
        w_rec = float(getattr(self.args, "t1pr_rec_weight", 0.10))
        w_pfl = float(getattr(self.args, "t1pr_pfl_weight", getattr(self.args, "pfl_local_weight", 0.10)))
        w_rn = max(w_rn, 0.0)
        w_rec = max(w_rec, 0.0)
        w_pfl = max(w_pfl, 0.0)
        s = w_rn + w_rec + w_pfl
        if s <= 1e-12:
            return 1.0, 0.0, 0.0
        return w_rn / s, w_rec / s, w_pfl / s

    def _select_recovery_from_noisy(self, p_rec, p_rn, p_local, noisy_mask):
        """
        Select recovered samples from noisy pool.

        To avoid FalseClean explosion in high noise, recovery is conservative by default:
        - net2/recovery must be confident: p_rec > recover_threshold
        - net1/FedRN should not strongly reject it: p_rn > recover_main_min_prob
        - optional net3 support can be enabled by recover_need_personal_support=1
        """
        threshold = float(getattr(self.args, "recover_threshold", 0.70))
        max_ratio = float(getattr(self.args, "recover_max_ratio", 0.10))
        max_ratio = min(max(max_ratio, 0.0), 1.0)

        candidate_mask = noisy_mask & (np.asarray(p_rec) > threshold)

        need_main_support = int(getattr(self.args, "recover_need_main_support", 1))
        if need_main_support:
            main_min = float(getattr(self.args, "recover_main_min_prob", 0.30))
            candidate_mask = candidate_mask & (np.asarray(p_rn) > main_min)

        need_personal_support = int(getattr(self.args, "recover_need_personal_support", 0))
        if need_personal_support:
            personal_min = float(getattr(self.args, "recover_personal_min_prob", 0.50))
            candidate_mask = candidate_mask & (np.asarray(p_local) > personal_min)

        candidate_pos = np.where(candidate_mask)[0]
        if len(candidate_pos) == 0:
            return np.array([], dtype=np.int64)

        noisy_count = int(noisy_mask.sum())
        max_recover = max(1, int(max_ratio * noisy_count))

        # Rank by combined recovery confidence, not only p_rec.
        score = 0.70 * np.asarray(p_rec) + 0.20 * np.asarray(p_rn) + 0.10 * np.asarray(p_local)
        candidate_pos = candidate_pos[np.argsort(score[candidate_pos])[::-1]]
        candidate_pos = candidate_pos[:max_recover]
        recover_idx = self.data_indices[candidate_pos]
        return self._safe_indices(recover_idx)

    def _kl_soft(self, student_logits, teacher_logits):
        temp = float(getattr(self.args, "t1pr_mutual_temp", 1.0))
        temp = max(temp, 1e-6)
        return F.kl_div(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits.detach() / temp, dim=1),
            reduction="batchmean",
        ) * (temp * temp)

    def _make_loader(self, indices, shuffle=True):
        return DataLoader(
            DatasetSplit(self.dataset, indices, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def train_three_models(self, net_main, net_recovery, net_personal, core_indices, recover_indices):
        """
        Train three separated models.

        net1/main:
            default uses only core clean samples, so recovered samples do not directly pollute upload model.
            Set --recover_train_global 1 if you want net1 to also use recovered samples.
        net2/recovery:
            uses core + recovered samples.
        net3/personal:
            uses core + recovered samples by default; set personal_train_mode='all' for PFL-only style.
        """
        recover_train_global = int(getattr(self.args, "recover_train_global", 0))
        personal_train_mode = str(getattr(self.args, "personal_train_mode", "core_recover"))

        core_indices = self._safe_indices(core_indices)
        recover_indices = self._safe_indices(recover_indices)

        if recover_train_global:
            main_train_indices = self._safe_indices(np.concatenate([core_indices, recover_indices]))
        else:
            main_train_indices = core_indices
        recovery_train_indices = self._safe_indices(np.concatenate([core_indices, recover_indices]))

        if personal_train_mode == "all":
            personal_train_indices = self.data_indices
        else:
            personal_train_indices = self._safe_indices(np.concatenate([core_indices, recover_indices]))

        if len(main_train_indices) == 0:
            main_train_indices = self.data_indices
        if len(recovery_train_indices) == 0:
            recovery_train_indices = core_indices if len(core_indices) > 0 else self.data_indices
        if len(personal_train_indices) == 0:
            personal_train_indices = core_indices if len(core_indices) > 0 else self.data_indices

        self.last_core_indices = self._safe_indices(core_indices)
        self.last_recover_indices = self._safe_indices(recover_indices)
        self.last_main_train_indices = self._safe_indices(main_train_indices)
        self.last_recovery_train_indices = self._safe_indices(recovery_train_indices)
        self.last_personal_train_indices = self._safe_indices(personal_train_indices)

        loader_main = self._make_loader(main_train_indices, shuffle=True)
        loader_recovery = self._make_loader(recovery_train_indices, shuffle=True)
        loader_personal = self._make_loader(personal_train_indices, shuffle=True)

        net_main.train()
        net_recovery.train()
        net_personal.train()

        optimizer_args = dict(
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        optimizer_main = torch.optim.SGD(net_main.parameters(), **optimizer_args)
        optimizer_recovery = torch.optim.SGD(net_recovery.parameters(), **optimizer_args)
        optimizer_personal = torch.optim.SGD(net_personal.parameters(), **optimizer_args)

        local_ep_main = int(getattr(self.args, "local_ep", 5))
        local_ep_recovery = int(getattr(self.args, "recover_ep", 1))
        local_ep_personal = int(getattr(self.args, "pfl_personal_ep", 1))
        mutual_weight = float(getattr(self.args, "t1pr_mutual_weight", 0.10))
        personal_distill_weight = float(getattr(self.args, "personal_distill_weight", 0.0))

        main_losses, recovery_losses, personal_losses = [], [], []

        # net1 learns from stable core samples and is softly regularized by net2.
        for _ in range(local_ep_main):
            batch_losses = []
            for inputs, targets, items, idxs in loader_main:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_main.zero_grad()
                logits_main = self._forward_logits(net_main, inputs, head="global")
                loss = self.loss_func(logits_main, targets)
                if mutual_weight > 0:
                    with torch.no_grad():
                        logits_recovery_teacher = self._forward_logits(net_recovery, inputs, head="global")
                    loss = loss + mutual_weight * self._kl_soft(logits_main, logits_recovery_teacher)
                loss.backward()
                optimizer_main.step()
                batch_losses.append(loss.item())
            if batch_losses:
                main_losses.append(float(np.mean(batch_losses)))

        # net2 learns from core + recovered samples and is softly regularized by net1.
        for _ in range(local_ep_recovery):
            batch_losses = []
            for inputs, targets, items, idxs in loader_recovery:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_recovery.zero_grad()
                logits_recovery = self._forward_logits(net_recovery, inputs, head="global")
                loss = self.loss_func(logits_recovery, targets)
                if mutual_weight > 0:
                    with torch.no_grad():
                        logits_main_teacher = self._forward_logits(net_main, inputs, head="global")
                    loss = loss + mutual_weight * self._kl_soft(logits_recovery, logits_main_teacher)
                loss.backward()
                optimizer_recovery.step()
                batch_losses.append(loss.item())
            if batch_losses:
                recovery_losses.append(float(np.mean(batch_losses)))

        # net3 is a separated personalized local model. It uses the local head.
        for _ in range(local_ep_personal):
            batch_losses = []
            for inputs, targets, items, idxs in loader_personal:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_personal.zero_grad()
                logits_personal = self._forward_logits(net_personal, inputs, head="local")
                loss = self.loss_func(logits_personal, targets)
                if personal_distill_weight > 0:
                    with torch.no_grad():
                        logits_main_teacher = self._forward_logits(net_main, inputs, head="global")
                    loss = loss + personal_distill_weight * self._kl_soft(logits_personal, logits_main_teacher)
                loss.backward()
                optimizer_personal.step()
                batch_losses.append(loss.item())
            if batch_losses:
                personal_losses.append(float(np.mean(batch_losses)))

        self.net1.load_state_dict(net_main.state_dict())
        self.net2.load_state_dict(net_recovery.state_dict())
        self.net3.load_state_dict(net_personal.state_dict())
        self.last_updated = self.args.g_epoch

        loss_main = float(np.mean(main_losses)) if main_losses else 0.0
        loss_recovery = float(np.mean(recovery_losses)) if recovery_losses else 0.0
        loss_personal = float(np.mean(personal_losses)) if personal_losses else 0.0
        return (
            net_main.state_dict(), loss_main,
            net_recovery.state_dict(), loss_recovery,
            net_personal.state_dict(), loss_personal,
        )

    def train_phase1_three(self, net_main, net_recovery, net_personal):
        """Warmup: all three models use all local noisy data."""
        w_main, loss_main, w_recovery, loss_recovery, w_personal, loss_personal = self.train_three_models(
            net_main,
            net_recovery,
            net_personal,
            core_indices=self.data_indices,
            recover_indices=np.array([], dtype=np.int64),
        )
        self.set_expertise()
        self.set_arbitrary_output()
        return w_main, loss_main, w_recovery, loss_recovery, w_personal, loss_personal

    def train_phase2_t1pr(self, net_main, net_recovery, net_personal, self_score, neighbor_net, neighbor_score):
        """
        Phase2:
        1) net1 + top-1 reliable neighbor produce p_rn;
        2) net2/recovery peer produces p_rec;
        3) net3/personal local head produces p_local;
        4) weighted fusion gives final_prob for core/noisy split;
        5) net2 recovers samples from noisy pool conservatively;
        6) net1/net2 mutual learning, net3 personalized training.
        """
        # 1. Main/global self probability.
        p_self = self.fit_gmm(self.net1, head="global")
        init_clean_idx, _, _, _ = self._select_core_and_noisy_by_prob(
            p_self, threshold=float(getattr(self.args, "p_threshold", 0.5))
        )

        # 2. Top-1 neighbor fine-tuning + probability.
        prob_list = [p_self]
        score_list = [float(self_score)]

        if neighbor_net is not None:
            neighbor_list = self.finetune_head([neighbor_net], init_clean_idx)
            p_neighbor = self.fit_gmm(neighbor_list[0], head="global")
            prob_list.append(p_neighbor)
            score_list.append(float(neighbor_score))

        score_sum = float(sum(score_list)) + 1e-8
        score_list = [s / score_sum for s in score_list]

        p_rn = np.zeros(len(p_self), dtype=np.float64)
        for p, s in zip(prob_list, score_list):
            p_rn += np.asarray(p) * s

        # 3. Separated recovery and personalized clean probabilities.
        p_rec = self.fit_gmm(self.net2, head="global")
        p_local = self.fit_gmm(self.net3, head="local")

        w_rn, w_rec, w_pfl = self._three_weights()
        final_prob = w_rn * p_rn + w_rec * p_rec + w_pfl * p_local

        # Optional agreement correction between global and personalized models.
        agreement = self.get_global_local_agreement(self.net1, self.net3)
        agree_weight = float(getattr(self.args, "pfl_agree_weight", 0.0))
        if abs(agree_weight) > 1e-12:
            final_prob = final_prob + agree_weight * (2.0 * agreement - 1.0)
            final_prob = np.clip(final_prob, 0.0, 1.0)
        agreement_ratio = float(np.mean(agreement)) if len(agreement) > 0 else 0.0

        threshold = float(getattr(self.args, "p_threshold", 0.5))
        core_idx, noisy_idx, core_mask, noisy_mask = self._select_core_and_noisy_by_prob(final_prob, threshold)

        recover_interval = int(getattr(self.args, "recover_interval", 5))
        recover_interval = max(recover_interval, 1)
        do_recover = (
            self.args.g_epoch >= self.args.warmup_epochs
            and ((self.args.g_epoch - self.args.warmup_epochs) % recover_interval == 0)
        )

        if do_recover:
            recover_idx = self._select_recovery_from_noisy(p_rec, p_rn, p_local, noisy_mask)
        else:
            recover_idx = np.array([], dtype=np.int64)

        w_main, loss_main, w_recovery, loss_recovery, w_personal, loss_personal = self.train_three_models(
            net_main,
            net_recovery,
            net_personal,
            core_indices=core_idx,
            recover_indices=recover_idx,
        )

        self.set_expertise()
        self.set_arbitrary_output()

        core_ratio = float(len(core_idx)) / float(len(self.data_indices) + 1e-8)
        recover_ratio = float(len(recover_idx)) / float(len(self.data_indices) + 1e-8)
        train_ratio = float(len(getattr(self, "last_main_train_indices", []))) / float(len(self.data_indices) + 1e-8)

        return (
            w_main,
            loss_main,
            w_recovery,
            loss_recovery,
            w_personal,
            loss_personal,
            core_ratio,
            recover_ratio,
            train_ratio,
            do_recover,
            agreement_ratio,
            w_rn,
            w_rec,
            w_pfl,
        )



# =============================================================================
# Evaluation helpers: optional holdout evaluation for net2 and net3.
# 默认 dict_users_test 仍然是空，所以这些指标会是 0；后续打开 APA 时可用。
# =============================================================================
def evaluate_local_model(local_objects, dataset, dict_users_test, args, model_name="net3", head="local"):
    acc_sum, loss_sum, valid = 0.0, 0.0, 0
    for client_idx, local in enumerate(local_objects):
        test_idxs = dict_users_test.get(client_idx, [])
        if len(test_idxs) == 0:
            continue
        loader = DataLoader(
            DatasetSplitPlain(dataset, test_idxs),
            batch_size=args.local_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        model = getattr(local, model_name)
        model.eval()
        correct, total, loss_total = 0.0, 0, 0.0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(args.device), labels.to(args.device)
                logits = get_logits(model(images), head=head)
                loss_total += F.cross_entropy(logits, labels, reduction="sum").item()
                pred = logits.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).float().sum().item()
                total += labels.size(0)
        if total > 0:
            acc_sum += 100.0 * correct / total
            loss_sum += loss_total / total
            valid += 1
    if valid == 0:
        return 0.0, 0.0
    return acc_sum / valid, loss_sum / valid

def main():
    start_time = time.time()
    args = args_parser()

    # 保持 get_model / 数据划分等逻辑兼容原代码；实验名改为 fedrn_t1pr，避免与 pfedrn 结果混淆。
    args.method = "pfedrn_t1pr_three"   # 新实验名：三模型 PFedRN + Top-1 + Periodic Recovery
    args.exp_method = "PFedRN-T1PR-3M"
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    args.schedule = [int(x) for x in args.schedule]

    # Step-1: 强制 top-1 neighbor。
    args.num_neighbors = 1

    # 先做云端 / 单边缘版本，减少边缘聚合变量。
    args.send_2_models = False
    args.num_edges = 1
    args.neighbor_scope = "global"

    # Step-2: recovery defaults. 如需命令行传参，需要在 options.py 里新增这些参数。
    if not hasattr(args, "recover_threshold"):
        args.recover_threshold = 0.60       # noisy pool 中 p_recovery 大于该值才允许放回
    if not hasattr(args, "recover_max_ratio"):
        args.recover_max_ratio = 0.30       # 最多放回 noisy pool 的 30%
    if not hasattr(args, "recover_ep"):
        args.recover_ep = getattr(args, "pfl_personal_ep", args.local_ep)
    if not hasattr(args, "recover_train_global"):
        args.recover_train_global = 1       # 1: net1 也使用 recovered samples；0: net1 只用 core clean
    if not hasattr(args, "recover_interval"):
        args.recover_interval = 5           # 每 5 个 global round 执行一次 noisy sample recovery

    # PFL guidance defaults. 建议先保持 agree=0.0，与之前 PFedRN no-agree 对齐。
    if not hasattr(args, "pfl_local_weight"):
        args.pfl_local_weight = 0.30
    if not hasattr(args, "pfl_agree_weight"):
        args.pfl_agree_weight = 0.0
    if not hasattr(args, "pfl_personal_ep"):
        args.pfl_personal_ep = 1

    # Three-model defaults. If command line does not explicitly set recover_train_global,
    # keep net1 conservative: recovered samples train net2/net3, but not uploaded net1.
    if "--recover_train_global" not in sys.argv:
        args.recover_train_global = 0
    if not hasattr(args, "t1pr_rn_weight"):
        args.t1pr_rn_weight = 0.80
    if not hasattr(args, "t1pr_rec_weight"):
        args.t1pr_rec_weight = 0.10
    if not hasattr(args, "t1pr_pfl_weight"):
        args.t1pr_pfl_weight = 0.10
    if not hasattr(args, "t1pr_mutual_weight"):
        args.t1pr_mutual_weight = 0.10
    if not hasattr(args, "t1pr_mutual_temp"):
        args.t1pr_mutual_temp = 1.0
    if not hasattr(args, "recover_need_main_support"):
        args.recover_need_main_support = 1
    if not hasattr(args, "recover_main_min_prob"):
        args.recover_main_min_prob = 0.30
    if not hasattr(args, "recover_need_personal_support"):
        args.recover_need_personal_support = 0
    if not hasattr(args, "recover_personal_min_prob"):
        args.recover_personal_min_prob = 0.50
    if not hasattr(args, "personal_train_mode"):
        args.personal_train_mode = "core_recover"
    if not hasattr(args, "personal_distill_weight"):
        args.personal_distill_weight = 0.0

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    classes_per_user = args.num_shards // args.num_users if args.partition == "shard" else "dirichlet"
    save_dir = os.path.join("resultdate", "PFedRN_T1PR_ThreeModel")
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(
        save_dir,
        f"详细模型_PFedRN-T1PR-3M{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
    )
    metrics_log_filename = os.path.join(
        save_dir,
        f"总体模型_PFedRN-T1PR-3M{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
    )

    print("Results will be saved to:", log_filename)
    print("Metrics ONLY will be saved to:", metrics_log_filename)
    for x in vars(args).items():
        print(x)

    if not torch.cuda.is_available() and args.gpu != -1:
        raise RuntimeError("Cuda is not available. Use --gpu -1 for CPU debugging.")

    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---------------------------------------------------------------------
    # Dataset and partition
    # ---------------------------------------------------------------------
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    labels = np.array(dataset_train.targets if hasattr(dataset_train, "targets") else dataset_train.train_labels)
    # Save labels before noise injection for analysis only. It never affects training.
    original_train_labels = labels.copy()
    args.img_size = int(dataset_train[0][0].shape[1])

    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == "shard":
        dict_users = sample_noniid_shard(labels=labels, num_users=args.num_users, num_shards=args.num_shards)
    else:
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)

    # 当前仍不切 holdout，保持和 FedRN / PFedRN 主结果公平。后续要看 APA 再打开。
    dict_users_train, dict_users_test = {}, {}
    for i in range(args.num_users):
        idxs = list(copy.deepcopy(dict_users[i]))
        rng = np.random.RandomState(args.seed + i)
        rng.shuffle(idxs)
        dict_users_train[i] = idxs
        dict_users_test[i] = []
        # 如需打开本地 APA，可改为：
        # split = int(len(idxs) * 0.8)
        # dict_users_train[i] = idxs[:split]
        # dict_users_test[i] = idxs[split:]

    # ---------------------------------------------------------------------
    # Noise injection: only training indices are noised.
    # ---------------------------------------------------------------------
    if sum(args.noise_group_num) != args.num_users:
        raise ValueError("sum(args.noise_group_num) must equal args.num_users")

    if len(args.group_noise_rate) == 1:
        args.group_noise_rate = args.group_noise_rate * 2
    args.group_noise_rate = [
        (args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
        for i in range(len(args.group_noise_rate) // 2)
    ]

    user_noise_type_rates = []
    for num_users_in_group, noise_type, (min_r, max_r) in zip(
        args.noise_group_num, args.noise_type_lst, args.group_noise_rate
    ):
        step = (max_r - min_r) / max(num_users_in_group, 1)
        rates = np.array(range(num_users_in_group)) * step + min_r
        user_noise_type_rates += list(zip([noise_type] * num_users_in_group, rates))

    for user, (noise_type, noise_rate) in enumerate(user_noise_type_rates):
        if noise_type == "clean":
            continue
        data_indices = list(copy.deepcopy(dict_users_train[user]))
        random.seed(args.seed + user)
        random.shuffle(data_indices)
        noise_num = int(len(data_indices) * noise_rate)
        for d_idx in data_indices[:noise_num]:
            if hasattr(dataset_train, "targets"):
                y = dataset_train.targets[d_idx]
                dataset_train.targets[d_idx] = noisify_label(y, num_classes=args.num_classes, noise_type=noise_type)
            else:
                y = dataset_train.train_labels[d_idx]
                dataset_train.train_labels[d_idx] = noisify_label(y, num_classes=args.num_classes, noise_type=noise_type)

    # torchvision 版本兼容：有些旧代码读 train_labels。
    if hasattr(dataset_train, "targets") and not hasattr(dataset_train, "train_labels"):
        dataset_train.train_labels = dataset_train.targets

    noisy_train_labels = np.array(dataset_train.targets if hasattr(dataset_train, "targets") else dataset_train.train_labels)
    real_clean_mask = (original_train_labels == noisy_train_labels)
    actual_clean_total = int(real_clean_mask.sum())
    actual_total = int(len(real_clean_mask))
    msg_actual_clean = "Actual clean labels in noisy train set: {}/{} ({:.3f})".format(
        actual_clean_total,
        actual_total,
        actual_clean_total / float(actual_total + 1e-8),
    )
    print(msg_actual_clean)

    log_train_loader = DataLoader(dataset_train, batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)
    log_test_loader = DataLoader(dataset_test, batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)

    net_glob = get_model(args).to(args.device)
    initial_state = copy.deepcopy(net_glob.state_dict())
    gaussian_noise = torch.randn(1, args.num_channels, args.img_size, args.img_size).to(args.device)

    local_objects = []
    for i in range(args.num_users):
        local = LocalUpdateThreeModelT1PR(
            args=args,
            user_idx=i,
            dataset=dataset_train,
            idxs=dict_users_train[i],
            gaussian_noise=gaussian_noise,
        )
        local.net1.load_state_dict(initial_state)
        local.net2.load_state_dict(initial_state)
        local.net3 = get_model(args).to(args.device)
        local.net3.load_state_dict(initial_state)
        local_objects.append(local)

    num_edges = max(args.num_edges, 1)
    clients_per_edge = args.num_users // num_edges
    all_client_ids = list(range(args.num_users))
    edge_clients_map = {}
    for e in range(num_edges):
        s = e * clients_per_edge
        edge_clients_map[e] = all_client_ids[s:] if e == num_edges - 1 else all_client_ids[s:s + clients_per_edge]

    print("\nStructure: {} edge server(s), neighbor_scope={}".format(num_edges, args.neighbor_scope))
    print("PFedRN-T1PR-3M: net1 global + net2 recovery peer + net3 personalized local. Only net1 is aggregated.")
    print("Three weights: rn={}, rec={}, pfl={}".format(args.t1pr_rn_weight, args.t1pr_rec_weight, args.t1pr_pfl_weight))
    print("Mutual weight: {}, recover_train_global: {}".format(args.t1pr_mutual_weight, args.recover_train_global))
    print("Recovery interval: every {} round(s) after warmup.\n".format(args.recover_interval))

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Experiment Start: {timestamp}\n")
        f.write(f"Args: {args}\n")
        f.write(msg_actual_clean + "\n")
        f.write("=" * 80 + "\n")
    with open(metrics_log_filename, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,global_test_acc,global_test_loss,rec_apa,rec_loss,personal_apa,personal_loss,"
            "core_clean_ratio,recover_ratio,train_ratio,agreement_ratio,recovery_trigger,"
            "w_rn,w_rec,w_pfl,core_count,recover_count,train_count,total_count,actual_train_clean_count,"
            "actual_round_clean_count,missed_clean_count,false_clean_count,total_wrong_count,actual_train_clean_ratio,clean_recall\n"
        )

    # ---------------------------------------------------------------------
    # Federated training loop
    # ---------------------------------------------------------------------
    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}: {} => {}".format(epoch + 1, args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay
        args.g_epoch = epoch

        edge_weights, edge_samples = [], []
        edge_logs = []
        round_losses, round_core, round_recover, round_train_ratio, round_recovery_trigger, round_agree = [], [], [], [], [], []
        round_w_rn, round_w_rec, round_w_pfl = [], [], []
        round_core_count, round_recover_count, round_train_count = 0, 0, 0
        round_total_count, round_actual_train_clean_count = 0, 0
        round_actual_clean_count, round_missed_clean_count = 0, 0
        round_false_clean_count, round_total_wrong_count = 0, 0

        for edge_id, current_edge_clients in edge_clients_map.items():
            m = max(int(args.frac * len(current_edge_clients)), 1)
            selected_clients = np.random.choice(current_edge_clients, m, replace=False)

            client_weights, client_samples = [], []
            client_losses, client_core, client_recover, client_train_ratio, client_recovery_trigger, client_agree = [], [], [], [], [], []
            client_w_rn, client_w_rec, client_w_pfl = [], [], []
            edge_core_count, edge_recover_count, edge_train_count = 0, 0, 0
            edge_total_count, edge_actual_train_clean_count = 0, 0
            edge_actual_clean_count, edge_missed_clean_count = 0, 0
            edge_false_clean_count, edge_total_wrong_count = 0, 0

            for client_idx in selected_clients:
                local = local_objects[client_idx]
                local.args = args

                net_main = copy.deepcopy(local.net1).to(args.device)
                net_recovery = copy.deepcopy(local.net2).to(args.device)
                net_personal = copy.deepcopy(local.net3).to(args.device)
                sample_size = len(dict_users_train[client_idx])

                if epoch < args.warmup_epochs:
                    w_main, loss_main, w_recovery, loss_recovery, w_personal, loss_personal = local.train_phase1_three(
                        net_main,
                        net_recovery,
                        net_personal,
                    )
                    core_ratio = 1.0
                    recover_ratio = 0.0
                    train_ratio = 1.0 if int(getattr(args, "recover_train_global", 0)) else 1.0
                    do_recover = False
                    agreement_ratio = 0.0
                    w_rn_used, w_rec_used, w_pfl_used = 1.0, 0.0, 0.0
                else:
                    # Top-1 reliable neighbor pool.
                    neighbor_pool = all_client_ids if args.neighbor_scope == "global" else current_edge_clients

                    sim_list, exp_list = [], []
                    cosine_sim = torch.nn.CosineSimilarity(dim=1)
                    for u in neighbor_pool:
                        sim = cosine_sim(
                            local.arbitrary_output.view(1, -1).to(args.device),
                            local_objects[u].arbitrary_output.view(1, -1).to(args.device),
                        ).item()
                        sim_list.append(sim)
                        exp_list.append(local_objects[u].expertise)

                    sim_min, sim_max = min(sim_list), max(sim_list)
                    exp_min, exp_max = min(exp_list), max(exp_list)
                    sim_norm = [(s - sim_min) / (sim_max - sim_min + 1e-8) for s in sim_list]
                    exp_norm = [(e - exp_min) / (exp_max - exp_min + 1e-8) for e in exp_list]

                    local_pos = neighbor_pool.index(client_idx)
                    w_alpha = getattr(args, "w_alpha", 0.6)
                    self_score = w_alpha * exp_norm[local_pos] + (1.0 - w_alpha) * 1.0

                    scores = []
                    for pos, (e_score, s_score) in enumerate(zip(exp_norm, sim_norm)):
                        global_client_id = neighbor_pool[pos]
                        if global_client_id == client_idx:
                            continue
                        score = w_alpha * e_score + (1.0 - w_alpha) * s_score
                        scores.append((score, global_client_id))
                    scores.sort(key=lambda x: x[0], reverse=True)

                    if len(scores) > 0:
                        top1_score, top1_id = scores[0]
                        top1_neighbor = copy.deepcopy(local_objects[top1_id].net1).to(args.device)
                    else:
                        top1_score, top1_neighbor = 0.0, None

                    w_main, loss_main, w_recovery, loss_recovery, w_personal, loss_personal, core_ratio, recover_ratio, train_ratio, do_recover, agreement_ratio, w_rn_used, w_rec_used, w_pfl_used = (
                        local.train_phase2_t1pr(
                            net_main,
                            net_recovery,
                            net_personal,
                            self_score,
                            top1_neighbor,
                            top1_score,
                        )
                    )

                # Only main/global model is uploaded.
                w_cpu = {k: v.detach().cpu() for k, v in w_main.items()}
                client_weights.append(w_cpu)
                client_samples.append(sample_size)
                client_losses.append(loss_main)
                client_core.append(core_ratio)
                client_recover.append(recover_ratio)
                client_train_ratio.append(train_ratio)
                client_recovery_trigger.append(1.0 if do_recover else 0.0)
                client_agree.append(agreement_ratio)
                client_w_rn.append(w_rn_used)
                client_w_rec.append(w_rec_used)
                client_w_pfl.append(w_pfl_used)

                # Analysis-only sample-selection statistics.
                local_data_indices = np.asarray(local.data_indices, dtype=np.int64)
                main_train_indices = np.asarray(
                    getattr(local, "last_main_train_indices", local_data_indices), dtype=np.int64
                )
                core_indices_logged = np.asarray(
                    getattr(local, "last_core_indices", local_data_indices if epoch < args.warmup_epochs else []), dtype=np.int64
                )
                recover_indices_logged = np.asarray(
                    getattr(local, "last_recover_indices", []), dtype=np.int64
                )

                local_actual_clean_count = int(real_clean_mask[local_data_indices].sum()) if len(local_data_indices) else 0
                train_count_i = int(len(main_train_indices))
                actual_train_clean_i = int(real_clean_mask[main_train_indices].sum()) if train_count_i > 0 else 0
                # FalseClean: samples selected for main/global training but actually noisy.
                false_clean_i = max(train_count_i - actual_train_clean_i, 0)
                # MissedClean: actually clean samples excluded from main/global training.
                missed_clean_i = max(local_actual_clean_count - actual_train_clean_i, 0)
                # TotalWrong: all sample-selection mistakes under the analysis-only true clean mask.
                total_wrong_i = false_clean_i + missed_clean_i

                edge_core_count += int(len(core_indices_logged))
                edge_recover_count += int(len(recover_indices_logged))
                edge_train_count += train_count_i
                edge_total_count += sample_size
                edge_actual_train_clean_count += actual_train_clean_i
                edge_actual_clean_count += local_actual_clean_count
                edge_missed_clean_count += missed_clean_i
                edge_false_clean_count += false_clean_i
                edge_total_wrong_count += total_wrong_i

            if len(client_weights) > 0:
                w_edge = fedavg(client_weights, client_samples, skip_local_head=True)
                edge_weights.append(w_edge)
                edge_samples.append(sum(client_samples))
                edge_actual_train_clean_ratio = edge_actual_train_clean_count / float(edge_train_count + 1e-8)
                edge_clean_recall = edge_actual_train_clean_count / float(edge_actual_clean_count + 1e-8)
                edge_logs.append(
                    "  --> [Edge {:02d}] clients={} loss={:.4f} core={:.3f} recover={:.3f} train={:.3f} agree={:.3f} "
                    "CleanCount={}/{} ActualTrainClean={:.3f} CleanRecall={:.3f} MissedClean={} FalseClean={} TotalWrong={} trigger={:.0f} weights=({:.2f},{:.2f},{:.2f})".format(
                        edge_id + 1,
                        len(client_weights),
                        float(np.mean(client_losses)),
                        float(np.mean(client_core)),
                        float(np.mean(client_recover)),
                        float(np.mean(client_train_ratio)),
                        float(np.mean(client_agree)) if client_agree else 0.0,
                        edge_train_count,
                        edge_total_count,
                        edge_actual_train_clean_ratio,
                        edge_clean_recall,
                        edge_missed_clean_count,
                        edge_false_clean_count,
                        edge_total_wrong_count,
                        float(np.max(client_recovery_trigger)) if client_recovery_trigger else 0.0,
                        float(np.mean(client_w_rn)) if client_w_rn else 0.0,
                        float(np.mean(client_w_rec)) if client_w_rec else 0.0,
                        float(np.mean(client_w_pfl)) if client_w_pfl else 0.0,
                    )
                )
                round_losses.extend(client_losses)
                round_core.extend(client_core)
                round_recover.extend(client_recover)
                round_train_ratio.extend(client_train_ratio)
                round_recovery_trigger.extend(client_recovery_trigger)
                round_agree.extend(client_agree)
                round_w_rn.extend(client_w_rn)
                round_w_rec.extend(client_w_rec)
                round_w_pfl.extend(client_w_pfl)
                round_core_count += edge_core_count
                round_recover_count += edge_recover_count
                round_train_count += edge_train_count
                round_total_count += edge_total_count
                round_actual_train_clean_count += edge_actual_train_clean_count
                round_actual_clean_count += edge_actual_clean_count
                round_missed_clean_count += edge_missed_clean_count
                round_false_clean_count += edge_false_clean_count
                round_total_wrong_count += edge_total_wrong_count

        if len(edge_weights) > 0:
            w_glob = fedavg(edge_weights, edge_samples, skip_local_head=True)
            net_glob.load_state_dict(w_glob, strict=False)

            for local in local_objects:
                sync_global_to_client(local.net1, w_glob, mode="global")
                sync_global_to_client(local.net2, w_glob, mode="recovery")
                sync_global_to_client(local.net3, w_glob, mode="recovery")

        global_train_acc, global_train_loss = test_img(net_glob, log_train_loader, args)
        global_test_acc, global_test_loss = test_img(net_glob, log_test_loader, args)
        rec_apa, rec_loss = evaluate_local_model(local_objects, dataset_train, dict_users_test, args, model_name="net2", head="global")
        personal_apa, personal_loss = evaluate_local_model(local_objects, dataset_train, dict_users_test, args, model_name="net3", head="local")

        train_loss = float(np.mean(round_losses)) if round_losses else 0.0
        core_ratio = float(np.mean(round_core)) if round_core else 0.0
        recover_ratio = float(np.mean(round_recover)) if round_recover else 0.0
        train_ratio = float(np.mean(round_train_ratio)) if round_train_ratio else 0.0
        recovery_trigger = float(np.max(round_recovery_trigger)) if round_recovery_trigger else 0.0
        agree_ratio = float(np.mean(round_agree)) if round_agree else 0.0
        w_rn_mean = float(np.mean(round_w_rn)) if round_w_rn else 1.0
        w_rec_mean = float(np.mean(round_w_rec)) if round_w_rec else 0.0
        w_pfl_mean = float(np.mean(round_w_pfl)) if round_w_pfl else 0.0
        actual_train_clean_ratio = round_actual_train_clean_count / float(round_train_count + 1e-8)
        clean_recall = round_actual_train_clean_count / float(round_actual_clean_count + 1e-8)

        log_round = "\n==================== Round {:3d} ====================".format(epoch)
        log_metric = (
            "Global Acc: {:.2f}% | Global Loss: {:.4f} | Rec APA: {:.2f}% | Personal APA: {:.2f}% | "
            "Train Loss: {:.4f} | CoreClean: {:.3f} | Recover: {:.3f} | TrainRatio: {:.3f} | Agree: {:.3f} | Weights: ({:.2f},{:.2f},{:.2f}) | "
            "CleanCount: {}/{} | ActualTrainClean: {:.3f} ({}/{}) | "
            "ActualCleanInRound: {} | CleanRecall: {:.3f} | MissedClean: {} | FalseClean: {} | TotalWrong: {} | RecoveryTrigger: {:.0f}"
        ).format(
            global_test_acc,
            global_test_loss,
            rec_apa,
            personal_apa,
            train_loss,
            core_ratio,
            recover_ratio,
            train_ratio,
            agree_ratio,
            w_rn_mean,
            w_rec_mean,
            w_pfl_mean,
            round_train_count,
            round_total_count,
            actual_train_clean_ratio,
            round_actual_train_clean_count,
            round_train_count,
            round_actual_clean_count,
            clean_recall,
            round_missed_clean_count,
            round_false_clean_count,
            round_total_wrong_count,
            recovery_trigger,
        )

        print(log_round)
        for s in edge_logs:
            print(s)
        print(log_metric)

        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(log_round + "\n")
            for s in edge_logs:
                f.write(s + "\n")
            f.write(log_metric + "\n")
        with open(metrics_log_filename, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{global_test_acc:.6f},{global_test_loss:.6f},"
                f"{rec_apa:.6f},{rec_loss:.6f},{personal_apa:.6f},{personal_loss:.6f},"
                f"{core_ratio:.6f},{recover_ratio:.6f},{train_ratio:.6f},{agree_ratio:.6f},{recovery_trigger:.0f},"
                f"{w_rn_mean:.6f},{w_rec_mean:.6f},{w_pfl_mean:.6f},{round_core_count},{round_recover_count},"
                f"{round_train_count},{round_total_count},{round_actual_train_clean_count},"
                f"{round_actual_clean_count},{round_missed_clean_count},{round_false_clean_count},{round_total_wrong_count},"
                f"{actual_train_clean_ratio:.6f},{clean_recall:.6f}\n"
            )

    print("Total time: {:.1f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
