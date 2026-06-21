"""
PFedRN Dual-Correction FedClean-style version with persistent label correction.

Design:
1) net1: FedRN/global upload branch. It uses FedRN reliable-neighbor selection to build S_core.
2) net2: local CNLL-style correction branch. It is kept local and generates inferred labels for samples rejected by net1/FedRN.
3) Noisy-pool samples are handled similarly to FedClean-style local/global agreement:
   - y1 == y2 == noisy_label: S_recover, original label is confirmed and returned.
   - y1 == y2 != noisy_label: S_corr, label is corrected to y1/y2.
4) In this persistent version, S_corr labels are written back into dataset_train.targets/train_labels once they are corrected.
   net1 trains on S_core + S_recover by default. S_corr is corrected persistently and used by net2; set --net1_use_corrected 1 if net1 should also use corrected samples in the same round.
5) Aggregation is NOT changed: uploaded net1 models are aggregated with the current sample-size FedAvg weight.

This file removes net3 and changes recovery into correction while keeping the aggregation style unchanged.
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
    from utils.options_pfedrn_dual_correction_fedclean import args_parser
except Exception:
    # Fallback for old project layout.
    from utils.options_fedrn_t1pr_three_model import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label
from models.nets import get_model
from models.test import test_img
from models.update import LocalUpdatePFedRN, DatasetSplit


# =============================================================================
# Optional custom CLI arguments.
# The original options parser may not know these flags. We remove them from
# sys.argv before args_parser() and apply them manually after parsing.
# =============================================================================
_CUSTOM_ARG_TYPES = {
    "--correction_interval": int,
    "--correction_conf_global": float,
    "--correction_conf_local": float,
    "--correction_min_rn_prob": float,
    "--correction_max_ratio": float,
    "--net1_use_corrected": int,
    "--correction_use_mutual": int,
    "--persistent_correction": int,
}


def _pop_custom_cli_args():
    overrides = {}
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in _CUSTOM_ARG_TYPES:
            if i + 1 >= len(sys.argv):
                raise ValueError(f"Missing value for custom argument {arg}")
            caster = _CUSTOM_ARG_TYPES[arg]
            key = arg[2:]
            overrides[key] = caster(sys.argv[i + 1])
            i += 2
        else:
            new_argv.append(arg)
            i += 1
    sys.argv = new_argv
    return overrides


CUSTOM_OVERRIDES = _pop_custom_cli_args()


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


class DatasetSplitCorrected(Dataset):
    """
    Dataset split that can override labels for corrected samples.

    Returns the same 4-tuple style used by LocalUpdate code:
        image, label, item_position, real_dataset_index
    """
    def __init__(self, dataset, idxs, corrected_label_map=None):
        self.dataset = dataset
        self.idxs = [int(x) for x in list(idxs)]
        self.corrected_label_map = corrected_label_map or {}

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        real_idx = int(self.idxs[int(item)])
        image, label = self.dataset[real_idx]
        if real_idx in self.corrected_label_map:
            label = int(self.corrected_label_map[real_idx])
        return image, int(label), int(item), real_idx


def get_logits(output, head="global"):
    """Compatible with single-output and dual-head models."""
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
    total = max(total, 1e-12)
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
    mode='global': sync global/upload model, preserve local head if present.
    mode='correction': sync global backbone into local correction model, preserve its classifier/head.
    """
    state = local_model.state_dict()
    for k, v in global_state.items():
        if k not in state:
            continue
        if mode == "global":
            if is_local_head_key(k):
                continue
            state[k] = v.clone()
        elif mode == "correction":
            if is_head_key(k):
                continue
            state[k] = v.clone()
    local_model.load_state_dict(state)


def _get_train_labels_array(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "train_labels"):
        return np.asarray(dataset.train_labels, dtype=np.int64)
    raise AttributeError("Dataset has neither targets nor train_labels")


def _set_train_label(dataset, idx, value):
    """Persistently set one training label while keeping targets/train_labels aliases consistent."""
    idx = int(idx)
    value = int(value)
    if hasattr(dataset, "targets"):
        dataset.targets[idx] = value
    if hasattr(dataset, "train_labels"):
        dataset.train_labels[idx] = value


def _apply_persistent_corrections(dataset, corrected_label_map):
    """Write corrected labels back to the shared dataset. Returns number of updates."""
    updated = 0
    for idx, label in (corrected_label_map or {}).items():
        _set_train_label(dataset, int(idx), int(label))
        updated += 1
    return int(updated)


class LocalUpdateDualCorrectionFedClean(LocalUpdatePFedRN):
    """
    Dual-model FedRN + CNLL-style correction.

    net1: global/FedRN upload branch.
    net2: local correction branch; not uploaded.

    Core clean is produced only by net1/FedRN p_rn. The local net2 branch is used only
    to confirm/correct samples in the noisy pool.
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

    def _kl_soft(self, student_logits, teacher_logits):
        temp = float(getattr(self.args, "t1pr_mutual_temp", 1.0))
        temp = max(temp, 1e-6)
        return F.kl_div(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits.detach() / temp, dim=1),
            reduction="batchmean",
        ) * (temp * temp)

    def _make_loader(self, indices, shuffle=True, corrected_label_map=None):
        return DataLoader(
            DatasetSplitCorrected(self.dataset, indices, corrected_label_map=corrected_label_map),
            batch_size=self.args.local_bs,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def _predict_label_conf(self, model, head="global"):
        model.eval()
        loader = self._make_loader(self.data_indices, shuffle=False, corrected_label_map=None)
        pred = np.zeros(len(self.data_indices), dtype=np.int64)
        conf = np.zeros(len(self.data_indices), dtype=np.float64)
        with torch.no_grad():
            for inputs, targets, items, idxs in loader:
                inputs = inputs.to(self.args.device)
                logits = self._forward_logits(model, inputs, head=head)
                prob = F.softmax(logits, dim=1)
                c, y = torch.max(prob, dim=1)
                pos = items.cpu().numpy().astype(np.int64)
                pred[pos] = y.detach().cpu().numpy().astype(np.int64)
                conf[pos] = c.detach().cpu().numpy().astype(np.float64)
        return pred, conf

    def _select_correction_from_noisy(self, p_rn, noisy_mask, net_main, net_correction):
        """
        CNLL-style correction on the noisy pool.

        Return:
            recover_clean_idx: y1 == y2 == y_anno, use original label.
            corrected_idx: y1 == y2 != y_anno, use corrected label y1/y2.
            corrected_label_map: {dataset_index: corrected_label}
            net1_return_idx: samples returned to net1; by default this is S_recover only.
        """
        p_rn = np.asarray(p_rn)
        noisy_mask = np.asarray(noisy_mask).astype(bool)
        data_indices = np.asarray(self.data_indices, dtype=np.int64)
        if noisy_mask.sum() == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                {},
                np.array([], dtype=np.int64),
                dict(correction_acc_proxy=0.0, agree_count=0, candidate_count=0),
            )

        y_anno_all = _get_train_labels_array(self.dataset)[data_indices]
        y1, conf1 = self._predict_label_conf(net_main, head="global")
        y2, conf2 = self._predict_label_conf(net_correction, head="global")

        tau_g = float(getattr(self.args, "correction_conf_global", 0.70))
        tau_l = float(getattr(self.args, "correction_conf_local", 0.70))
        min_rn = float(getattr(self.args, "correction_min_rn_prob", 0.20))

        agree = (y1 == y2)
        confident = (conf1 >= tau_g) & (conf2 >= tau_l)
        not_strongly_rejected = p_rn >= min_rn
        candidate_mask = noisy_mask & agree & confident & not_strongly_rejected

        candidate_pos = np.where(candidate_mask)[0]
        if len(candidate_pos) == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                {},
                np.array([], dtype=np.int64),
                dict(correction_acc_proxy=0.0, agree_count=int(agree[noisy_mask].sum()), candidate_count=0),
            )

        max_ratio = float(getattr(self.args, "correction_max_ratio", 0.10))
        max_ratio = min(max(max_ratio, 0.0), 1.0)
        noisy_count = int(noisy_mask.sum())
        max_return = max(1, int(max_ratio * noisy_count))

        score = 0.45 * conf1 + 0.45 * conf2 + 0.10 * p_rn
        candidate_pos = candidate_pos[np.argsort(score[candidate_pos])[::-1]]
        candidate_pos = candidate_pos[:max_return]

        recover_pos = candidate_pos[y1[candidate_pos] == y_anno_all[candidate_pos]]
        corrected_pos = candidate_pos[y1[candidate_pos] != y_anno_all[candidate_pos]]

        recover_clean_idx = self._safe_indices(data_indices[recover_pos])
        corrected_idx = self._safe_indices(data_indices[corrected_pos])
        corrected_label_map = {int(data_indices[pos]): int(y1[pos]) for pos in corrected_pos}

        # Persistent correction option: directly write corrected labels back to dataset_train.
        # This makes future rounds use corrected labels even if the client is selected again later.
        persistent_correction = int(getattr(self.args, "persistent_correction", 1))
        persistent_update_count = 0
        if persistent_correction and len(corrected_label_map) > 0:
            persistent_update_count = _apply_persistent_corrections(self.dataset, corrected_label_map)

        # FedClean-style: S_recover returns to net1; S_corr is corrected and used by net2.
        # By default, corrected samples do not train net1.
        net1_use_corrected = int(getattr(self.args, "net1_use_corrected", 0))
        if net1_use_corrected:
            net1_return_idx = self._safe_indices(np.concatenate([recover_clean_idx, corrected_idx]))
        else:
            net1_return_idx = self._safe_indices(recover_clean_idx)

        info = dict(
            agree_count=int(agree[noisy_mask].sum()),
            candidate_count=int(len(candidate_pos)),
            recover_clean_count=int(len(recover_clean_idx)),
            corrected_count=int(len(corrected_idx)),
            persistent_update_count=int(persistent_update_count),
        )
        return recover_clean_idx, corrected_idx, corrected_label_map, net1_return_idx, info

    def train_dual_models(self, net_main, net_correction, core_indices, recover_clean_indices, corrected_label_map, net1_return_indices):
        core_indices = self._safe_indices(core_indices)
        recover_clean_indices = self._safe_indices(recover_clean_indices)
        corrected_indices = self._safe_indices(list((corrected_label_map or {}).keys()))
        net1_return_indices = self._safe_indices(net1_return_indices)

        # net1/global upload branch: use S_core + S_recover by default.
        # If --net1_use_corrected 1, corrected samples are also included through net1_return_indices.
        main_train_indices = self._safe_indices(np.concatenate([core_indices, net1_return_indices]))
        # net2/local correction branch: uses S_core + S_recover + S_corr with corrected labels.
        correction_train_indices = self._safe_indices(np.concatenate([core_indices, recover_clean_indices, corrected_indices]))

        if len(main_train_indices) == 0:
            main_train_indices = self.data_indices
        if len(correction_train_indices) == 0:
            correction_train_indices = core_indices if len(core_indices) > 0 else self.data_indices

        # Corrected labels are applied only for corrected samples; core/recovered-confirmed samples use dataset labels.
        corrected_label_map = corrected_label_map or {}
        main_corrected_map = {int(k): int(v) for k, v in corrected_label_map.items() if int(k) in set(main_train_indices.tolist())}
        correction_corrected_map = {int(k): int(v) for k, v in corrected_label_map.items() if int(k) in set(correction_train_indices.tolist())}

        self.last_core_indices = self._safe_indices(core_indices)
        self.last_recover_clean_indices = self._safe_indices(recover_clean_indices)
        self.last_corrected_indices = self._safe_indices(corrected_indices)
        self.last_return_indices = self._safe_indices(np.concatenate([recover_clean_indices, corrected_indices]))
        self.last_net1_return_indices = self._safe_indices(net1_return_indices)
        self.last_main_train_indices = self._safe_indices(main_train_indices)
        self.last_correction_train_indices = self._safe_indices(correction_train_indices)
        self.last_corrected_label_map = dict(corrected_label_map)

        # Aggregation is intentionally unchanged in this version.
        # The main loop still uses sample_size as the FedAvg weight.
        self.last_agg_weight = float(len(self.data_indices))

        loader_main = self._make_loader(main_train_indices, shuffle=True, corrected_label_map=main_corrected_map)
        loader_correction = self._make_loader(correction_train_indices, shuffle=True, corrected_label_map=correction_corrected_map)

        net_main.train()
        net_correction.train()

        optimizer_args = dict(lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        optimizer_main = torch.optim.SGD(net_main.parameters(), **optimizer_args)
        optimizer_correction = torch.optim.SGD(net_correction.parameters(), **optimizer_args)

        local_ep_main = int(getattr(self.args, "local_ep", 5))
        local_ep_correction = int(getattr(self.args, "recover_ep", 1))
        mutual_weight = float(getattr(self.args, "t1pr_mutual_weight", 0.10))
        use_mutual = int(getattr(self.args, "correction_use_mutual", 1))
        if not use_mutual:
            mutual_weight = 0.0

        main_losses, correction_losses = [], []

        # net1 trains on FedRN core plus S_recover confirmed samples.
        for _ in range(local_ep_main):
            batch_losses = []
            for inputs, targets, items, idxs in loader_main:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_main.zero_grad()
                logits_main = self._forward_logits(net_main, inputs, head="global")
                loss = self.loss_func(logits_main, targets)
                if mutual_weight > 0:
                    with torch.no_grad():
                        logits_teacher = self._forward_logits(net_correction, inputs, head="global")
                    loss = loss + mutual_weight * self._kl_soft(logits_main, logits_teacher)
                loss.backward()
                optimizer_main.step()
                batch_losses.append(loss.item())
            if batch_losses:
                main_losses.append(float(np.mean(batch_losses)))

        # net2 trains on core + recover-confirmed + corrected samples.
        for _ in range(local_ep_correction):
            batch_losses = []
            for inputs, targets, items, idxs in loader_correction:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_correction.zero_grad()
                logits_correction = self._forward_logits(net_correction, inputs, head="global")
                loss = self.loss_func(logits_correction, targets)
                if mutual_weight > 0:
                    with torch.no_grad():
                        logits_teacher = self._forward_logits(net_main, inputs, head="global")
                    loss = loss + mutual_weight * self._kl_soft(logits_correction, logits_teacher)
                loss.backward()
                optimizer_correction.step()
                batch_losses.append(loss.item())
            if batch_losses:
                correction_losses.append(float(np.mean(batch_losses)))

        self.net1.load_state_dict(net_main.state_dict())
        self.net2.load_state_dict(net_correction.state_dict())
        self.last_updated = self.args.g_epoch

        loss_main = float(np.mean(main_losses)) if main_losses else 0.0
        loss_correction = float(np.mean(correction_losses)) if correction_losses else 0.0
        return net_main.state_dict(), loss_main, net_correction.state_dict(), loss_correction

    def train_phase1_dual(self, net_main, net_correction):
        """Warmup: both models use all local noisy data."""
        w_main, loss_main, w_corr, loss_corr = self.train_dual_models(
            net_main,
            net_correction,
            core_indices=self.data_indices,
            recover_clean_indices=np.array([], dtype=np.int64),
            corrected_label_map={},
            net1_return_indices=np.array([], dtype=np.int64),
        )
        self.set_expertise()
        self.set_arbitrary_output()
        return w_main, loss_main, w_corr, loss_corr

    def train_phase2_dual(self, net_main, net_correction, self_score, neighbor_nets=None, neighbor_scores=None):
        """
        Phase2:
        1) net1 + top-k reliable neighbors produce p_rn.
        2) p_rn alone decides core/noisy split.
        3) noisy pool is corrected/confirmed only when net1 and net2 agree with high confidence.
        4) S_recover trains net1; S_recover + S_corr train net2.
        """
        p_self = self.fit_gmm(self.net1, head="global")
        init_clean_idx, _, _, _ = self._select_core_and_noisy_by_prob(
            p_self, threshold=float(getattr(self.args, "p_threshold", 0.5))
        )

        prob_list = [p_self]
        score_list = [float(self_score)]

        if neighbor_nets is None:
            neighbor_nets = []
        elif not isinstance(neighbor_nets, (list, tuple)):
            neighbor_nets = [neighbor_nets]
        if neighbor_scores is None:
            neighbor_scores = []
        elif not isinstance(neighbor_scores, (list, tuple)):
            neighbor_scores = [neighbor_scores]

        valid_neighbors = [(net, score) for net, score in zip(neighbor_nets, neighbor_scores) if net is not None]
        if len(valid_neighbors) > 0:
            neighbor_model_list = [net for net, _ in valid_neighbors]
            neighbor_score_list = [float(score) for _, score in valid_neighbors]
            neighbor_model_list = self.finetune_head(neighbor_model_list, init_clean_idx)
            for neighbor_model, neighbor_score in zip(neighbor_model_list, neighbor_score_list):
                p_neighbor = self.fit_gmm(neighbor_model, head="global")
                prob_list.append(p_neighbor)
                score_list.append(float(neighbor_score))

        score_sum = float(sum(score_list)) + 1e-8
        score_list = [s / score_sum for s in score_list]
        p_rn = np.zeros(len(p_self), dtype=np.float64)
        for p, s in zip(prob_list, score_list):
            p_rn += np.asarray(p) * s

        threshold = float(getattr(self.args, "p_threshold", 0.5))
        core_idx, noisy_idx, core_mask, noisy_mask = self._select_core_and_noisy_by_prob(p_rn, threshold)

        correction_interval = int(getattr(self.args, "correction_interval", getattr(self.args, "recover_interval", 5)))
        correction_interval = max(correction_interval, 1)
        do_correction = (
            self.args.g_epoch >= self.args.warmup_epochs
            and ((self.args.g_epoch - self.args.warmup_epochs) % correction_interval == 0)
        )

        if do_correction:
            recover_clean_idx, corrected_idx, corrected_label_map, net1_return_idx, corr_info = self._select_correction_from_noisy(
                p_rn=p_rn,
                noisy_mask=noisy_mask,
                net_main=net_main,
                net_correction=net_correction,
            )
        else:
            recover_clean_idx = np.array([], dtype=np.int64)
            corrected_idx = np.array([], dtype=np.int64)
            corrected_label_map = {}
            net1_return_idx = np.array([], dtype=np.int64)
            corr_info = dict(agree_count=0, candidate_count=0, recover_clean_count=0, corrected_count=0, persistent_update_count=0)

        w_main, loss_main, w_corr, loss_corr = self.train_dual_models(
            net_main,
            net_correction,
            core_indices=core_idx,
            recover_clean_indices=recover_clean_idx,
            corrected_label_map=corrected_label_map,
            net1_return_indices=net1_return_idx,
        )

        self.set_expertise()
        self.set_arbitrary_output()

        total = float(len(self.data_indices) + 1e-8)
        core_ratio = float(len(core_idx)) / total
        recover_clean_ratio = float(len(recover_clean_idx)) / total
        corrected_ratio = float(len(corrected_idx)) / total
        train_ratio = float(len(getattr(self, "last_main_train_indices", []))) / total
        corr_info.update(dict(
            core_count=int(len(core_idx)),
            recover_clean_count=int(len(recover_clean_idx)),
            corrected_count=int(len(corrected_idx)),
            net1_return_count=int(len(net1_return_idx)),
            main_train_count=int(len(getattr(self, "last_main_train_indices", []))),
            correction_train_count=int(len(getattr(self, "last_correction_train_indices", []))),
        ))

        return (
            w_main,
            loss_main,
            w_corr,
            loss_corr,
            core_ratio,
            recover_clean_ratio,
            corrected_ratio,
            train_ratio,
            do_correction,
            corr_info,
        )


# =============================================================================
# Evaluation helpers: optional holdout evaluation for net2.
# 默认 dict_users_test 仍然是空，所以这些指标一般为 0。
# =============================================================================
def evaluate_local_model(local_objects, dataset, dict_users_test, args, model_name="net2", head="global"):
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


def _label_correct_count(indices, corrected_label_map, noisy_labels, original_labels):
    """Count how many used labels are equal to the original clean labels."""
    count = 0
    for idx in [int(x) for x in list(indices)]:
        if idx in corrected_label_map:
            used_y = int(corrected_label_map[idx])
        else:
            used_y = int(noisy_labels[idx])
        if used_y == int(original_labels[idx]):
            count += 1
    return int(count)


def _corrected_accuracy(corrected_indices, corrected_label_map, original_labels):
    corrected_indices = [int(x) for x in list(corrected_indices)]
    if len(corrected_indices) == 0:
        return 0, 0, 0.0
    correct = 0
    for idx in corrected_indices:
        if int(corrected_label_map.get(idx, -1)) == int(original_labels[idx]):
            correct += 1
    false = len(corrected_indices) - correct
    return int(correct), int(false), float(correct) / float(len(corrected_indices))


def main():
    start_time = time.time()
    args = args_parser()
    for k, v in CUSTOM_OVERRIDES.items():
        setattr(args, k, v)

    args.method = "pfedrn_dual_correction_fedclean"
    args.exp_method = "PFedRN-DualCorrection-FedCleanStyle"
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    args.schedule = [int(x) for x in args.schedule]

    # Do not force top-1. Keep --num_neighbors effective, so this can align with original FedRN.
    if not hasattr(args, "num_neighbors"):
        args.num_neighbors = 2

    args.send_2_models = False
    args.num_edges = 1
    args.neighbor_scope = "global"

    # FedClean-style defaults. These can be passed by the custom CLI flags above.
    if not hasattr(args, "correction_interval"):
        args.correction_interval = int(getattr(args, "recover_interval", 5))
    if not hasattr(args, "correction_conf_global"):
        args.correction_conf_global = 0.70
    if not hasattr(args, "correction_conf_local"):
        args.correction_conf_local = 0.70
    if not hasattr(args, "correction_min_rn_prob"):
        args.correction_min_rn_prob = 0.20
    if not hasattr(args, "correction_max_ratio"):
        args.correction_max_ratio = 0.10
    if not hasattr(args, "net1_use_corrected"):
        args.net1_use_corrected = 0
    if not hasattr(args, "correction_use_mutual"):
        args.correction_use_mutual = 1
    if not hasattr(args, "persistent_correction"):
        args.persistent_correction = 1
    if not hasattr(args, "recover_ep"):
        args.recover_ep = getattr(args, "pfl_personal_ep", 1)
    if not hasattr(args, "t1pr_mutual_weight"):
        args.t1pr_mutual_weight = 0.05
    if not hasattr(args, "t1pr_mutual_temp"):
        args.t1pr_mutual_temp = 1.0

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    classes_per_user = args.num_shards // args.num_users if args.partition == "shard" else "dirichlet"
    save_dir = os.path.join("resultdate", "PFedRN_DualCorrection_FedCleanStyle")
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(
        save_dir,
        f"详细模型_PFedRN-DualCorrection-FedClean{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
    )
    metrics_log_filename = os.path.join(
        save_dir,
        f"总体模型_PFedRN-DualCorrection-FedClean{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
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
    # Dataset and partition.
    # ---------------------------------------------------------------------
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    labels = np.array(dataset_train.targets if hasattr(dataset_train, "targets") else dataset_train.train_labels)
    original_train_labels = labels.copy()
    args.img_size = int(dataset_train[0][0].shape[1])

    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == "shard":
        dict_users = sample_noniid_shard(labels=labels, num_users=args.num_users, num_shards=args.num_shards)
    else:
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)

    # Keep data split aligned with FedRN original: no extra per-client shuffle.
    dict_users_train, dict_users_test = {}, {}
    for i in range(args.num_users):
        dict_users_train[i] = list(copy.deepcopy(dict_users[i]))
        dict_users_test[i] = []

    # ---------------------------------------------------------------------
    # Noise injection: aligned with original FedRN detailed logger.
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
        random.seed(args.seed)
        random.shuffle(data_indices)
        noise_num = int(len(data_indices) * noise_rate)
        for d_idx in data_indices[:noise_num]:
            if hasattr(dataset_train, "targets"):
                y = dataset_train.targets[d_idx]
                dataset_train.targets[d_idx] = noisify_label(y, num_classes=args.num_classes, noise_type=noise_type)
            else:
                y = dataset_train.train_labels[d_idx]
                dataset_train.train_labels[d_idx] = noisify_label(y, num_classes=args.num_classes, noise_type=noise_type)

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
        local = LocalUpdateDualCorrectionFedClean(
            args=args,
            user_idx=i,
            dataset=dataset_train,
            idxs=dict_users_train[i],
            gaussian_noise=gaussian_noise,
        )
        local.net1.load_state_dict(initial_state)
        local.net2.load_state_dict(initial_state)
        local_objects.append(local)

    num_edges = max(args.num_edges, 1)
    clients_per_edge = args.num_users // num_edges
    all_client_ids = list(range(args.num_users))
    edge_clients_map = {}
    for e in range(num_edges):
        s = e * clients_per_edge
        edge_clients_map[e] = all_client_ids[s:] if e == num_edges - 1 else all_client_ids[s:s + clients_per_edge]

    print("\nStructure: {} edge server(s), neighbor_scope={}".format(num_edges, args.neighbor_scope))
    print("PFedRN-DualCorrection-FedCleanStyle: net1 FedRN/global upload + net2 local CNLL-style correction. Only net1 is aggregated.")
    print("Reliable neighbors used by p_rn: num_neighbors={}".format(args.num_neighbors))
    print("Correction interval: every {} round(s) after warmup.".format(args.correction_interval))
    print("Correction thresholds: global_conf={}, local_conf={}, min_rn_prob={}".format(
        args.correction_conf_global, args.correction_conf_local, args.correction_min_rn_prob
    ))
    print("Net1 train: S_core + S_recover; net1_use_corrected={}, correction_max_ratio={}".format(
        args.net1_use_corrected, args.correction_max_ratio
    ))
    print("Persistent correction: {} (S_corr labels are written back to dataset_train)\n".format(args.persistent_correction))

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Experiment Start: {timestamp}\n")
        f.write(f"Args: {args}\n")
        f.write(msg_actual_clean + "\n")
        f.write("=" * 100 + "\n")
    with open(metrics_log_filename, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,global_test_acc,global_test_loss,correction_apa,correction_loss,"
            "core_ratio,recover_clean_ratio,corrected_ratio,net1_train_ratio,correction_trigger,"
            "core_count,recover_clean_count,corrected_count,net1_return_count,net1_train_count,net2_train_count,total_count,"
            "actual_net1_label_correct_count,actual_round_clean_count,selected_original_clean_count,missed_clean_count,"
            "false_train_label_count,correction_correct_count,correction_false_count,persistent_update_count,correction_acc,agg_weight_sum,"
            "actual_net1_label_correct_ratio,orig_clean_recall\n"
        )

    # ---------------------------------------------------------------------
    # Federated training loop.
    # ---------------------------------------------------------------------
    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}: {} => {}".format(epoch + 1, args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay
        args.g_epoch = epoch

        edge_weights, edge_samples = [], []
        edge_logs = []
        round_losses = []
        round_core, round_recover_clean, round_corrected, round_train_ratio, round_trigger = [], [], [], [], []
        round_core_count, round_recover_clean_count, round_corrected_count, round_net1_return_count = 0, 0, 0, 0
        round_net1_train_count, round_net2_train_count, round_total_count = 0, 0, 0
        round_actual_label_correct_count, round_actual_clean_count, round_selected_original_clean_count = 0, 0, 0
        round_missed_clean_count, round_false_train_label_count = 0, 0
        round_corr_correct_count, round_corr_false_count = 0, 0
        round_persistent_update_count = 0
        round_agg_weight = 0.0

        for edge_id, current_edge_clients in edge_clients_map.items():
            m = max(int(args.frac * len(current_edge_clients)), 1)
            selected_clients = np.random.choice(current_edge_clients, m, replace=False)

            client_weights, client_samples = [], []
            client_losses = []
            client_core, client_recover_clean, client_corrected, client_train_ratio, client_trigger = [], [], [], [], []
            edge_core_count, edge_recover_clean_count, edge_corrected_count, edge_net1_return_count = 0, 0, 0, 0
            edge_net1_train_count, edge_net2_train_count, edge_total_count = 0, 0, 0
            edge_actual_label_correct_count, edge_actual_clean_count, edge_selected_original_clean_count = 0, 0, 0
            edge_missed_clean_count, edge_false_train_label_count = 0, 0
            edge_corr_correct_count, edge_corr_false_count = 0, 0
            edge_persistent_update_count = 0
            edge_agg_weight = 0.0

            for client_idx in selected_clients:
                local = local_objects[client_idx]
                local.args = args

                net_main = copy.deepcopy(local.net1).to(args.device)
                net_correction = copy.deepcopy(local.net2).to(args.device)
                sample_size = len(dict_users_train[client_idx])

                if epoch < args.warmup_epochs:
                    w_main, loss_main, w_corr, loss_corr = local.train_phase1_dual(net_main, net_correction)
                    core_ratio = 1.0
                    recover_clean_ratio = 0.0
                    corrected_ratio = 0.0
                    train_ratio = 1.0
                    do_correction = False
                    corr_info = dict(core_count=sample_size, recover_clean_count=0, corrected_count=0, persistent_update_count=0,
                                     net1_return_count=0, main_train_count=sample_size, correction_train_count=sample_size)
                else:
                    # Top-k reliable neighbor pool for FedRN p_rn.
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

                    k_neighbors = max(int(getattr(args, "num_neighbors", 1)), 0)
                    neighbor_nets, neighbor_scores = [], []
                    for neighbor_score, neighbor_id in scores[:k_neighbors]:
                        neighbor_nets.append(copy.deepcopy(local_objects[neighbor_id].net1).to(args.device))
                        neighbor_scores.append(float(neighbor_score))

                    w_main, loss_main, w_corr, loss_corr, core_ratio, recover_clean_ratio, corrected_ratio, train_ratio, do_correction, corr_info = (
                        local.train_phase2_dual(
                            net_main,
                            net_correction,
                            self_score,
                            neighbor_nets,
                            neighbor_scores,
                        )
                    )

                # Only net1/global branch is uploaded.
                w_cpu = {k: v.detach().cpu() for k, v in w_main.items()}
                client_weights.append(w_cpu)

                # Keep current aggregation unchanged: sample-size weighted FedAvg.
                agg_weight_i = float(sample_size)
                client_samples.append(sample_size)

                client_losses.append(loss_main)
                client_core.append(core_ratio)
                client_recover_clean.append(recover_clean_ratio)
                client_corrected.append(corrected_ratio)
                client_train_ratio.append(train_ratio)
                client_trigger.append(1.0 if do_correction else 0.0)

                # Analysis-only statistics.
                local_data_indices = np.asarray(local.data_indices, dtype=np.int64)
                main_train_indices = np.asarray(getattr(local, "last_main_train_indices", local_data_indices), dtype=np.int64)
                core_indices_logged = np.asarray(getattr(local, "last_core_indices", local_data_indices if epoch < args.warmup_epochs else []), dtype=np.int64)
                recover_clean_indices_logged = np.asarray(getattr(local, "last_recover_clean_indices", []), dtype=np.int64)
                corrected_indices_logged = np.asarray(getattr(local, "last_corrected_indices", []), dtype=np.int64)
                net1_return_indices_logged = np.asarray(getattr(local, "last_net1_return_indices", []), dtype=np.int64)
                correction_train_indices_logged = np.asarray(getattr(local, "last_correction_train_indices", []), dtype=np.int64)
                corrected_label_map = getattr(local, "last_corrected_label_map", {})

                # Labels may have been persistently corrected in dataset_train.
                current_train_labels = _get_train_labels_array(dataset_train)
                current_clean_mask = (original_train_labels == current_train_labels)
                local_actual_clean_count = int(current_clean_mask[local_data_indices].sum()) if len(local_data_indices) else 0
                selected_original_clean_i = int(current_clean_mask[main_train_indices].sum()) if len(main_train_indices) else 0
                train_count_i = int(len(main_train_indices))
                label_correct_i = _label_correct_count(main_train_indices, corrected_label_map, current_train_labels, original_train_labels)
                false_train_label_i = max(train_count_i - label_correct_i, 0)
                missed_clean_i = max(local_actual_clean_count - selected_original_clean_i, 0)
                corr_correct_i, corr_false_i, _ = _corrected_accuracy(corrected_indices_logged, corrected_label_map, original_train_labels)
                persistent_update_i = int(corr_info.get("persistent_update_count", 0))

                edge_core_count += int(len(core_indices_logged))
                edge_recover_clean_count += int(len(recover_clean_indices_logged))
                edge_corrected_count += int(len(corrected_indices_logged))
                edge_net1_return_count += int(len(net1_return_indices_logged))
                edge_net1_train_count += train_count_i
                edge_net2_train_count += int(len(correction_train_indices_logged))
                edge_total_count += sample_size
                edge_actual_label_correct_count += label_correct_i
                edge_actual_clean_count += local_actual_clean_count
                edge_selected_original_clean_count += selected_original_clean_i
                edge_missed_clean_count += missed_clean_i
                edge_false_train_label_count += false_train_label_i
                edge_corr_correct_count += corr_correct_i
                edge_corr_false_count += corr_false_i
                edge_persistent_update_count += persistent_update_i
                edge_agg_weight += max(1.0, agg_weight_i)

            if len(client_weights) > 0:
                w_edge = fedavg(client_weights, client_samples, skip_local_head=True)
                edge_weights.append(w_edge)
                edge_samples.append(sum(client_samples))

                edge_actual_label_correct_ratio = edge_actual_label_correct_count / float(edge_net1_train_count + 1e-8)
                edge_orig_clean_recall = edge_selected_original_clean_count / float(edge_actual_clean_count + 1e-8)
                edge_corr_total = edge_corr_correct_count + edge_corr_false_count
                edge_corr_acc = edge_corr_correct_count / float(edge_corr_total + 1e-8)
                edge_logs.append(
                    "  --> [Edge {:02d}] clients={} loss={:.4f} core={:.3f} recoverC={:.3f} corr={:.3f} train={:.3f} "
                    "Net1Train={}/{} LabelCorrect={:.3f} OrigCleanRecall={:.3f} MissedClean={} FalseTrainLabel={} "
                    "RecoverClean={} Corrected={} PersistUpdate={} CorrAcc={:.3f} Net1Return={} AggW={:.1f} trigger={:.0f}".format(
                        edge_id + 1,
                        len(client_weights),
                        float(np.mean(client_losses)),
                        float(np.mean(client_core)),
                        float(np.mean(client_recover_clean)),
                        float(np.mean(client_corrected)),
                        float(np.mean(client_train_ratio)),
                        edge_net1_train_count,
                        edge_total_count,
                        edge_actual_label_correct_ratio,
                        edge_orig_clean_recall,
                        edge_missed_clean_count,
                        edge_false_train_label_count,
                        edge_recover_clean_count,
                        edge_corrected_count,
                        edge_persistent_update_count,
                        edge_corr_acc,
                        edge_net1_return_count,
                        edge_agg_weight,
                        float(np.max(client_trigger)) if client_trigger else 0.0,
                    )
                )

                round_losses.extend(client_losses)
                round_core.extend(client_core)
                round_recover_clean.extend(client_recover_clean)
                round_corrected.extend(client_corrected)
                round_train_ratio.extend(client_train_ratio)
                round_trigger.extend(client_trigger)
                round_core_count += edge_core_count
                round_recover_clean_count += edge_recover_clean_count
                round_corrected_count += edge_corrected_count
                round_net1_return_count += edge_net1_return_count
                round_net1_train_count += edge_net1_train_count
                round_net2_train_count += edge_net2_train_count
                round_total_count += edge_total_count
                round_actual_label_correct_count += edge_actual_label_correct_count
                round_actual_clean_count += edge_actual_clean_count
                round_selected_original_clean_count += edge_selected_original_clean_count
                round_missed_clean_count += edge_missed_clean_count
                round_false_train_label_count += edge_false_train_label_count
                round_corr_correct_count += edge_corr_correct_count
                round_corr_false_count += edge_corr_false_count
                round_persistent_update_count += edge_persistent_update_count
                round_agg_weight += edge_agg_weight

        if len(edge_weights) > 0:
            w_glob = fedavg(edge_weights, edge_samples, skip_local_head=True)
            net_glob.load_state_dict(w_glob, strict=False)
            for local in local_objects:
                sync_global_to_client(local.net1, w_glob, mode="global")
                sync_global_to_client(local.net2, w_glob, mode="correction")

        global_train_acc, global_train_loss = test_img(net_glob, log_train_loader, args)
        global_test_acc, global_test_loss = test_img(net_glob, log_test_loader, args)
        corr_apa, corr_loss = evaluate_local_model(local_objects, dataset_train, dict_users_test, args, model_name="net2", head="global")

        train_loss = float(np.mean(round_losses)) if round_losses else 0.0
        core_ratio = float(np.mean(round_core)) if round_core else 0.0
        recover_clean_ratio = float(np.mean(round_recover_clean)) if round_recover_clean else 0.0
        corrected_ratio = float(np.mean(round_corrected)) if round_corrected else 0.0
        net1_train_ratio = float(np.mean(round_train_ratio)) if round_train_ratio else 0.0
        correction_trigger = float(np.max(round_trigger)) if round_trigger else 0.0
        actual_label_correct_ratio = round_actual_label_correct_count / float(round_net1_train_count + 1e-8)
        orig_clean_recall = round_selected_original_clean_count / float(round_actual_clean_count + 1e-8)
        correction_total = round_corr_correct_count + round_corr_false_count
        correction_acc = round_corr_correct_count / float(correction_total + 1e-8)

        log_round = "\n==================== Round {:3d} ====================".format(epoch)
        log_metric = (
            "Global Acc: {:.2f}% | Global Loss: {:.4f} | Corr APA: {:.2f}% | Train Loss: {:.4f} | "
            "CoreClean: {:.3f} | RecoverClean: {:.3f} | Corrected: {:.3f} | Net1TrainRatio: {:.3f} | "
            "Net1Train: {}/{} | ActualTrainLabelCorrect: {:.3f} ({}/{}) | "
            "ActualCleanInRound: {} | OrigCleanRecall: {:.3f} | MissedClean: {} | FalseTrainLabel: {} | "
            "RecoverCleanCount: {} | CorrectedCount: {} | PersistUpdate: {} | CorrectionAcc: {:.3f} ({}/{}) | "
            "Net1Return: {} | AggWeight: {:.1f} | CorrectionTrigger: {:.0f}"
        ).format(
            global_test_acc,
            global_test_loss,
            corr_apa,
            train_loss,
            core_ratio,
            recover_clean_ratio,
            corrected_ratio,
            net1_train_ratio,
            round_net1_train_count,
            round_total_count,
            actual_label_correct_ratio,
            round_actual_label_correct_count,
            round_net1_train_count,
            round_actual_clean_count,
            orig_clean_recall,
            round_missed_clean_count,
            round_false_train_label_count,
            round_recover_clean_count,
            round_corrected_count,
            round_persistent_update_count,
            correction_acc,
            round_corr_correct_count,
            correction_total,
            round_net1_return_count,
            round_agg_weight,
            correction_trigger,
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
                f"{corr_apa:.6f},{corr_loss:.6f},{core_ratio:.6f},{recover_clean_ratio:.6f},{corrected_ratio:.6f},"
                f"{net1_train_ratio:.6f},{correction_trigger:.0f},{round_core_count},{round_recover_clean_count},"
                f"{round_corrected_count},{round_net1_return_count},{round_net1_train_count},{round_net2_train_count},"
                f"{round_total_count},{round_actual_label_correct_count},{round_actual_clean_count},{round_selected_original_clean_count},"
                f"{round_missed_clean_count},{round_false_train_label_count},{round_corr_correct_count},{round_corr_false_count},"
                f"{round_persistent_update_count},{correction_acc:.6f},{round_agg_weight:.6f},{actual_label_correct_ratio:.6f},{orig_clean_recall:.6f}\n"
            )

    print("Total time: {:.1f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
