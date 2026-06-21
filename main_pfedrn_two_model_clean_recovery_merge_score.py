"""
Two-model PFedRN clean-recovery main.

Design goal
-----------
Remove net2 from the previous three-model PFedRN-T1PR design and keep only:

1) net1: FedRN main/global model. It is uploaded and aggregated by FedAvg.
2) net3: personalized local model. It is NOT uploaded. Its local head only works as a
   conservative confirmer for clean-sample recovery from FedRN's noisy pool.

Important principles
--------------------
- No label correction. Never change labels.
- FedRN's p_rn alone decides the main core clean set S_core.
- S_recover is selected only from FedRN noisy pool and only when both net1 and net3
  strongly support the current training label.
- S_recover is now persistent at the client side: once a sample is recovered, it is saved
  into that client's recovery pool and can be reused when the client is selected again.
- Every round still re-scores all samples to form S_core and S_noisy; persistent S_recover
  is used as an extra training pool for net1 only.
- net1 trains on S_core union persistent S_recover with normal per-sample CE weight.
- net3 trains on S_core by default, so it does not self-reinforce on recovered samples.

Recommended first-run high-noise configuration:
    --recover_conf_main 0.90
    --recover_conf_personal 0.95
    --recover_min_p_rn 0.25
    --recover_memory_threshold 2
    --recover_max_ratio 0.05
    --personal_train_mode core
    --warmup_train_personal 0
    --reset_net3_after_warmup 1

This file is adapted from main_pfedrn_t1pr_three_model.py, but removes net2 from the
algorithmic path. If your project stores option parsers in utils/, copy
options_pfedrn_two_model_clean_recovery.py to utils/ as well.
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
    # Preferred parser for the persistent-recovery version.
    from utils.options_pfedrn_two_model_clean_recovery_persistent import args_parser
except Exception:
    try:
        # Backward compatible: also works if you overwrite the old option file.
        from utils.options_pfedrn_two_model_clean_recovery import args_parser
    except Exception:
        try:
            from utils.options_fedrn_t1pr_three_model import args_parser
        except Exception:
            from utils.options_fedrn_t1pr import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label
from models.nets_original_fedrn import get_model as get_global_model
# For the current alignment/debug run, keep net3 single-head as well.
# Later, if you want a dual-head personalized net, change only get_personal_model.
get_personal_model = get_global_model
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
    """Compatible with single-output models and dual-head CNN4Conv."""
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
    mode='global': sync global model, but preserve fc_local.
    mode='backbone_only': sync backbone only, preserve all heads including local head.
    """
    state = local_model.state_dict()
    for k, v in global_state.items():
        if k not in state:
            continue
        if mode == "global":
            if is_local_head_key(k):
                continue
            state[k] = v.clone()
        elif mode in ["backbone_only", "recovery"]:
            if is_head_key(k):
                continue
            state[k] = v.clone()
    local_model.load_state_dict(state)


def ensure_two_model_args(args):
    """Add defaults when running with an older options parser."""
    defaults = {
        "recover_interval": 1,
        "recover_max_ratio": 0.05,
        "recover_conf_main": 0.90,
        "recover_conf_personal": 0.95,
        "recover_min_p_rn": 0.25,
        "recover_memory_threshold": 2,
        "recover_memory_decay": 1,
        "recover_persistent": 1,
        "recover_pool_max_ratio": 1.0,
        "recover_clear_on_reset": 1,
        "recover_merge_into_core": 1,
        "personal_train_mode": "core",
        "pfl_personal_ep": 1,
        "personal_distill_weight": 0.0,
        "warmup_train_personal": 0,
        "reset_net3_after_warmup": 1,
        "neighbor_scope": "global",
        "num_edges": 1,
        "w_alpha": 0.6,
        "p_threshold": 0.5,
        "weight_decay": 0.0,
        "num_workers": 0,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


class LocalUpdateTwoModelCleanRecovery(LocalUpdatePFedRN):
    """
    Local update for FedRN + personalized conservative clean recovery.

    net1: main FedRN/global model, uploaded and aggregated.
    net3: personalized local model, not uploaded. It only confirms whether a noisy-pool
          sample's current label is likely clean.
    """

    def _ensure_recovery_memory(self):
        if not hasattr(self, "recovery_memory") or len(self.recovery_memory) != len(self.data_indices):
            self.recovery_memory = np.zeros(len(self.data_indices), dtype=np.int32)

    def _ensure_recovery_pool(self):
        """Persistent client-side recovered set, stored as global dataset indices."""
        local_set = set(int(x) for x in np.asarray(self.data_indices, dtype=np.int64))

        if not hasattr(self, "recovery_pool") or self.recovery_pool is None:
            self.recovery_pool = set()
        else:
            # Keep only samples belonging to this client, in case an old object was reused.
            self.recovery_pool = set(int(x) for x in self.recovery_pool if int(x) in local_set)

        if not hasattr(self, "recovery_pool_order") or self.recovery_pool_order is None:
            self.recovery_pool_order = []
        self.recovery_pool_order = [int(x) for x in self.recovery_pool_order if int(x) in self.recovery_pool]

    def _clear_recovery_pool(self):
        self.recovery_pool = set()
        self.recovery_pool_order = []
        self.last_recover_pool_indices = np.array([], dtype=np.int64)
        self.last_new_recover_indices = np.array([], dtype=np.int64)

    def _add_to_recovery_pool(self, new_recover_indices):
        """Accumulate newly recovered samples into the persistent per-client pool.

        Returns only the samples that were newly inserted into the pool this round.
        If a sample was already in the pool, it is not counted as a new addition.
        """
        self._ensure_recovery_pool()
        new_recover_indices = self._safe_indices(new_recover_indices)
        if len(new_recover_indices) == 0:
            return np.array([], dtype=np.int64)

        added = []
        local_set = set(int(x) for x in np.asarray(self.data_indices, dtype=np.int64))
        for idx in new_recover_indices:
            idx = int(idx)
            if idx not in local_set:
                continue
            if idx not in self.recovery_pool:
                self.recovery_pool.add(idx)
                self.recovery_pool_order.append(idx)
                added.append(idx)

        # Optional cap. Default 1.0 means at most all local samples can be retained.
        max_ratio = float(getattr(self.args, "recover_pool_max_ratio", 1.0))
        max_ratio = min(max(max_ratio, 0.0), 1.0)
        max_pool_size = int(max_ratio * len(self.data_indices))
        if max_ratio > 0.0:
            max_pool_size = max(1, max_pool_size)
        else:
            max_pool_size = 0

        # FIFO pruning if the pool is capped below the local dataset size.
        while len(self.recovery_pool_order) > max_pool_size:
            old_idx = int(self.recovery_pool_order.pop(0))
            self.recovery_pool.discard(old_idx)
            if old_idx in added:
                added.remove(old_idx)

        return self._safe_indices(added)

    def _get_recovery_pool_indices(self):
        self._ensure_recovery_pool()
        if len(self.recovery_pool_order) == 0:
            return np.array([], dtype=np.int64)
        return self._safe_indices(self.recovery_pool_order)

    def _get_recovery_pool_extra_indices(self, core_indices):
        """Return persistent recover samples that are not already in current S_core."""
        pool_idx = self._get_recovery_pool_indices()
        if len(pool_idx) == 0:
            return pool_idx
        core_set = set(int(x) for x in np.asarray(core_indices, dtype=np.int64))
        extra_idx = [int(x) for x in pool_idx if int(x) not in core_set]
        return self._safe_indices(extra_idx)

    def _safe_indices(self, arr):
        arr = np.array(arr, dtype=np.int64)
        if len(arr) == 0:
            return arr
        return np.unique(arr)

    def _select_core_and_noisy_by_prob(self, prob, threshold):
        prob = np.asarray(prob)
        core_mask = prob > threshold
        noisy_mask = ~core_mask

        # Prevent empty local training when GMM is overly strict.
        if core_mask.sum() == 0:
            keep_n = max(1, int(0.1 * len(prob)))
            top_pos = np.argsort(prob)[-keep_n:]
            core_mask[top_pos] = True
            noisy_mask = ~core_mask

        core_idx = self.data_indices[core_mask]
        noisy_idx = self.data_indices[noisy_mask]
        return self._safe_indices(core_idx), self._safe_indices(noisy_idx), core_mask, noisy_mask

    def _make_loader(self, indices, shuffle=True):
        return DataLoader(
            DatasetSplit(self.dataset, indices, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def _kl_soft(self, student_logits, teacher_logits):
        temp = float(getattr(self.args, "t1pr_mutual_temp", 1.0))
        temp = max(temp, 1e-6)
        return F.kl_div(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits.detach() / temp, dim=1),
            reduction="batchmean",
        ) * (temp * temp)

    def _get_local_noisy_labels(self):
        if hasattr(self.dataset, "targets"):
            labels = np.asarray(self.dataset.targets)
        else:
            labels = np.asarray(self.dataset.train_labels)
        return labels[np.asarray(self.data_indices, dtype=np.int64)]

    def _label_support(self, model, head="global"):
        """
        For all local data_indices, return:
        - pred: predicted class
        - conf_y: softmax confidence on the current training label y_i
        - max_conf: max softmax confidence
        All arrays are aligned with self.data_indices / p_rn positions.
        """
        model.eval()
        labels_local = self._get_local_noisy_labels()
        n = len(self.data_indices)
        pred = np.zeros(n, dtype=np.int64)
        conf_y = np.zeros(n, dtype=np.float64)
        max_conf = np.zeros(n, dtype=np.float64)
        loader = self._make_loader(self.data_indices, shuffle=False)

        with torch.no_grad():
            for inputs, targets, items, idxs in loader:
                inputs = inputs.to(self.args.device)
                logits = self._forward_logits(model, inputs, head=head)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                idxs_np = idxs.detach().cpu().numpy().astype(np.int64)

                # idxs returned by DatasetSplit are positions in self.data_indices in the original project.
                # If a customized DatasetSplit returns global indices, convert them back defensively.
                if np.max(idxs_np) >= n:
                    pos_map = {int(g): p for p, g in enumerate(self.data_indices)}
                    idxs_np = np.array([pos_map[int(g)] for g in idxs_np], dtype=np.int64)

                pred[idxs_np] = probs.argmax(axis=1)
                max_conf[idxs_np] = probs.max(axis=1)
                y_batch = labels_local[idxs_np].astype(np.int64)
                conf_y[idxs_np] = probs[np.arange(len(idxs_np)), y_batch]

        return pred, conf_y, max_conf

    def _select_clean_recovery_from_noisy(self, p_rn, noisy_mask, net_main, net_personal):
        """
        Conservative clean recovery from FedRN noisy pool.

        A sample can be recovered only if:
        - FedRN placed it in noisy pool;
        - p_rn is not too low, meaning FedRN does not strongly reject it;
        - net1 predicts the current training label with high confidence;
        - net3 predicts the current training label with high confidence;
        - it has satisfied the condition for enough rounds via recovery_memory.

        Note: labels are never changed.
        """
        self._ensure_recovery_memory()
        p_rn = np.asarray(p_rn, dtype=np.float64)
        noisy_mask = np.asarray(noisy_mask, dtype=bool)
        labels_local = self._get_local_noisy_labels().astype(np.int64)

        tau_main = float(getattr(self.args, "recover_conf_main", 0.90))
        tau_personal = float(getattr(self.args, "recover_conf_personal", 0.95))
        min_p_rn = float(getattr(self.args, "recover_min_p_rn", 0.25))
        memory_threshold = int(getattr(self.args, "recover_memory_threshold", 2))
        memory_decay = int(getattr(self.args, "recover_memory_decay", 1))
        max_ratio = float(getattr(self.args, "recover_max_ratio", 0.05))
        max_ratio = min(max(max_ratio, 0.0), 1.0)

        pred_main, conf_main_y, _ = self._label_support(net_main, head="global")
        pred_personal, conf_personal_y, _ = self._label_support(net_personal, head="local")

        cond = noisy_mask.copy()
        cond &= p_rn >= min_p_rn
        cond &= pred_main == labels_local
        cond &= pred_personal == labels_local
        cond &= conf_main_y >= tau_main
        cond &= conf_personal_y >= tau_personal

        # Update memory for all local samples. The memory gates new additions into the persistent recovery pool.
        self.recovery_memory[cond] += 1
        not_cond = ~cond
        if memory_decay > 0:
            self.recovery_memory[not_cond] = np.maximum(0, self.recovery_memory[not_cond] - memory_decay)

        stable_mask = noisy_mask & (self.recovery_memory >= memory_threshold) & cond
        candidate_pos = np.where(stable_mask)[0]
        if len(candidate_pos) == 0:
            self.last_recover_candidate_count = 0
            self.last_recover_precision_proxy = 0.0
            self.last_recover_mean_conf_main = 0.0
            self.last_recover_mean_conf_personal = 0.0
            return np.array([], dtype=np.int64)

        noisy_count = int(noisy_mask.sum())
        max_recover = int(max_ratio * noisy_count)
        if max_ratio > 0.0:
            max_recover = max(1, max_recover)
        else:
            max_recover = 0
        if max_recover <= 0:
            return np.array([], dtype=np.int64)

        # Rank candidates by conservative agreement score. Use min(conf_main_y, conf_personal_y)
        # so both models need to be confident.
        score = np.minimum(conf_main_y, conf_personal_y) + 0.10 * p_rn
        candidate_pos = candidate_pos[np.argsort(score[candidate_pos])[::-1]]
        selected_pos = candidate_pos[:max_recover]
        recover_idx = self.data_indices[selected_pos]

        self.last_recover_candidate_count = int(len(candidate_pos))
        self.last_recover_mean_conf_main = float(np.mean(conf_main_y[selected_pos])) if len(selected_pos) else 0.0
        self.last_recover_mean_conf_personal = float(np.mean(conf_personal_y[selected_pos])) if len(selected_pos) else 0.0
        return self._safe_indices(recover_idx)

    def train_net1_only(self, net_main, train_indices):
        train_indices = self._safe_indices(train_indices)
        if len(train_indices) == 0:
            train_indices = self.data_indices
        self.last_score_indices = self._safe_indices(train_indices)
        self.last_core_indices = self._safe_indices(train_indices)
        self.last_recover_indices = np.array([], dtype=np.int64)
        self.last_new_recover_indices = np.array([], dtype=np.int64)
        self.last_recover_pool_indices = self._get_recovery_pool_indices() if hasattr(self, "recovery_pool") else np.array([], dtype=np.int64)
        self.last_main_train_indices = self._safe_indices(train_indices)
        self.last_personal_train_indices = np.array([], dtype=np.int64)

        loader_main = self._make_loader(train_indices, shuffle=True)
        net_main.train()
        optimizer_main = torch.optim.SGD(
            net_main.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        losses = []
        local_ep_main = int(getattr(self.args, "local_ep", 5))
        for _ in range(local_ep_main):
            batch_losses = []
            for inputs, targets, items, idxs in loader_main:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_main.zero_grad()
                logits = self._forward_logits(net_main, inputs, head="global")
                loss = self.loss_func(logits, targets)
                loss.backward()
                optimizer_main.step()
                batch_losses.append(loss.item())
            if batch_losses:
                losses.append(float(np.mean(batch_losses)))

        self.net1.load_state_dict(net_main.state_dict())
        self.last_updated = self.args.g_epoch
        return net_main.state_dict(), float(np.mean(losses)) if losses else 0.0

    def train_two_models(self, net_main, net_personal, core_indices, recover_indices):
        """
        Train net1 and net3.

        net1: S_core union S_recover, same per-sample CE weight.
              In persistent mode, S_recover is the current client recovery pool
              excluding samples already covered by current S_core.
        net3: S_core only by default. Set --personal_train_mode core_recover or all if needed.
        """
        core_indices = self._safe_indices(core_indices)
        recover_indices = self._safe_indices(recover_indices)
        main_train_indices = self._safe_indices(np.concatenate([core_indices, recover_indices]))

        personal_train_mode = str(getattr(self.args, "personal_train_mode", "core"))
        if personal_train_mode == "all":
            personal_train_indices = self.data_indices
        elif personal_train_mode == "core_recover":
            personal_train_indices = main_train_indices
        else:
            personal_train_indices = core_indices

        if len(main_train_indices) == 0:
            main_train_indices = self.data_indices
        if len(personal_train_indices) == 0:
            personal_train_indices = core_indices if len(core_indices) > 0 else self.data_indices

        self.last_score_indices = self._safe_indices(core_indices)
        self.last_core_indices = self._safe_indices(core_indices)
        self.last_recover_indices = self._safe_indices(recover_indices)  # training recover part, not merely newly added samples
        if not hasattr(self, "last_new_recover_indices"):
            self.last_new_recover_indices = np.array([], dtype=np.int64)
        self.last_recover_pool_indices = self._get_recovery_pool_indices() if hasattr(self, "recovery_pool") else self._safe_indices(recover_indices)
        self.last_main_train_indices = self._safe_indices(main_train_indices)
        self.last_personal_train_indices = self._safe_indices(personal_train_indices)

        loader_main = self._make_loader(main_train_indices, shuffle=True)
        loader_personal = self._make_loader(personal_train_indices, shuffle=True)

        net_main.train()
        net_personal.train()

        optimizer_args = dict(
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        optimizer_main = torch.optim.SGD(net_main.parameters(), **optimizer_args)
        optimizer_personal = torch.optim.SGD(net_personal.parameters(), **optimizer_args)

        local_ep_main = int(getattr(self.args, "local_ep", 5))
        local_ep_personal = int(getattr(self.args, "pfl_personal_ep", 1))
        personal_distill_weight = float(getattr(self.args, "personal_distill_weight", 0.0))

        main_losses, personal_losses = [], []

        # net1: S_core + S_recover. Same per-sample CE weight because they are in one loader.
        for _ in range(local_ep_main):
            batch_losses = []
            for inputs, targets, items, idxs in loader_main:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_main.zero_grad()
                logits_main = self._forward_logits(net_main, inputs, head="global")
                loss = self.loss_func(logits_main, targets)
                loss.backward()
                optimizer_main.step()
                batch_losses.append(loss.item())
            if batch_losses:
                main_losses.append(float(np.mean(batch_losses)))

        # net3: local head. By default core only, to avoid self-reinforcing recovered samples.
        for _ in range(local_ep_personal):
            batch_losses = []
            for inputs, targets, items, idxs in loader_personal:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_personal.zero_grad()
                logits_personal = self._forward_logits(net_personal, inputs, head="local")
                loss = self.loss_func(logits_personal, targets)
                if personal_distill_weight > 0:
                    with torch.no_grad():
                        logits_teacher = self._forward_logits(net_main, inputs, head="global")
                    loss = loss + personal_distill_weight * self._kl_soft(logits_personal, logits_teacher)
                loss.backward()
                optimizer_personal.step()
                batch_losses.append(loss.item())
            if batch_losses:
                personal_losses.append(float(np.mean(batch_losses)))

        self.net1.load_state_dict(net_main.state_dict())
        self.net3.load_state_dict(net_personal.state_dict())
        self.last_updated = self.args.g_epoch

        loss_main = float(np.mean(main_losses)) if main_losses else 0.0
        loss_personal = float(np.mean(personal_losses)) if personal_losses else 0.0
        return net_main.state_dict(), loss_main, net_personal.state_dict(), loss_personal

    def train_two_models_merge_recover_into_score(self, net_main, net_personal, score_indices, recover_indices):
        """
        Variant: directly merge the persistent recover pool into S_core/score.

        After merging:
        - score_indices: original FedRN score set S_core, kept for logging only.
        - recover_indices: persistent recover-pool samples not already in original S_core, kept for logging.
        - merged_score_indices = S_core union recover_indices.

        Training behavior:
        - net1 trains on merged_score_indices.
        - net3 also trains on merged_score_indices when personal_train_mode='core', because recover is now treated as score/core.

        This is intentionally stronger than the previous setting where net3 saw only original S_core.
        """
        score_indices = self._safe_indices(score_indices)
        recover_indices = self._safe_indices(recover_indices)
        merged_score_indices = self._safe_indices(np.concatenate([score_indices, recover_indices]))

        if len(merged_score_indices) == 0:
            merged_score_indices = self.data_indices

        personal_train_mode = str(getattr(self.args, "personal_train_mode", "core"))
        if personal_train_mode == "all":
            personal_train_indices = self.data_indices
        elif personal_train_mode == "core_recover":
            personal_train_indices = merged_score_indices
        else:
            # In this variant, recover has been merged into score/core.
            personal_train_indices = merged_score_indices

        if len(personal_train_indices) == 0:
            personal_train_indices = merged_score_indices if len(merged_score_indices) > 0 else self.data_indices

        # Logging convention:
        # - last_score_indices: original score-only S_core.
        # - last_core_indices: merged score = original S_core + recover pool.
        # - last_recover_indices: recover part that was absorbed into score this round.
        self.last_score_indices = self._safe_indices(score_indices)
        self.last_core_indices = self._safe_indices(merged_score_indices)
        self.last_recover_indices = self._safe_indices(recover_indices)
        if not hasattr(self, "last_new_recover_indices"):
            self.last_new_recover_indices = np.array([], dtype=np.int64)
        self.last_recover_pool_indices = self._get_recovery_pool_indices() if hasattr(self, "recovery_pool") else self._safe_indices(recover_indices)
        self.last_main_train_indices = self._safe_indices(merged_score_indices)
        self.last_personal_train_indices = self._safe_indices(personal_train_indices)

        loader_main = self._make_loader(merged_score_indices, shuffle=True)
        loader_personal = self._make_loader(personal_train_indices, shuffle=True)

        net_main.train()
        net_personal.train()

        optimizer_args = dict(
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        optimizer_main = torch.optim.SGD(net_main.parameters(), **optimizer_args)
        optimizer_personal = torch.optim.SGD(net_personal.parameters(), **optimizer_args)

        local_ep_main = int(getattr(self.args, "local_ep", 5))
        local_ep_personal = int(getattr(self.args, "pfl_personal_ep", 1))
        personal_distill_weight = float(getattr(self.args, "personal_distill_weight", 0.0))

        main_losses, personal_losses = [], []

        for _ in range(local_ep_main):
            batch_losses = []
            for inputs, targets, items, idxs in loader_main:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_main.zero_grad()
                logits_main = self._forward_logits(net_main, inputs, head="global")
                loss = self.loss_func(logits_main, targets)
                loss.backward()
                optimizer_main.step()
                batch_losses.append(loss.item())
            if batch_losses:
                main_losses.append(float(np.mean(batch_losses)))

        for _ in range(local_ep_personal):
            batch_losses = []
            for inputs, targets, items, idxs in loader_personal:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                optimizer_personal.zero_grad()
                logits_personal = self._forward_logits(net_personal, inputs, head="local")
                loss = self.loss_func(logits_personal, targets)
                if personal_distill_weight > 0:
                    with torch.no_grad():
                        logits_teacher = self._forward_logits(net_main, inputs, head="global")
                    loss = loss + personal_distill_weight * self._kl_soft(logits_personal, logits_teacher)
                loss.backward()
                optimizer_personal.step()
                batch_losses.append(loss.item())
            if batch_losses:
                personal_losses.append(float(np.mean(batch_losses)))

        self.net1.load_state_dict(net_main.state_dict())
        self.net3.load_state_dict(net_personal.state_dict())
        self.last_updated = self.args.g_epoch

        loss_main = float(np.mean(main_losses)) if main_losses else 0.0
        loss_personal = float(np.mean(personal_losses)) if personal_losses else 0.0
        return net_main.state_dict(), loss_main, net_personal.state_dict(), loss_personal

    def train_phase1_two(self, net_main, net_personal):
        """Warmup stage. Default: train net1 with all noisy local data; do not train net3."""
        warmup_train_personal = int(getattr(self.args, "warmup_train_personal", 0))
        if warmup_train_personal:
            w_main, loss_main, w_personal, loss_personal = self.train_two_models(
                net_main,
                net_personal,
                core_indices=self.data_indices,
                recover_indices=np.array([], dtype=np.int64),
            )
        else:
            w_main, loss_main = self.train_net1_only(net_main, self.data_indices)
            w_personal = net_personal.state_dict()
            loss_personal = 0.0
            # Keep local object's net3 unchanged.
            if hasattr(self, "net3"):
                self.net3.load_state_dict(net_personal.state_dict())

        self.set_expertise()
        self.set_arbitrary_output()
        return w_main, loss_main, w_personal, loss_personal

    def train_phase2_clean_recovery(self, net_main, net_personal, self_score, neighbor_net, neighbor_score):
        """
        Phase2:
        1) net1 + top-1 reliable neighbor produce p_rn;
        2) p_rn alone selects S_core and FedRN noisy pool;
        3) net1 and net3 conservatively recover clean samples from noisy pool;
        4) net1 trains on S_core union S_recover; net3 trains on S_core by default.
        """
        # 1. Main/global self probability.
        p_self = self.fit_gmm(self.net1, head="global")
        init_clean_idx, _, _, _ = self._select_core_and_noisy_by_prob(
            p_self, threshold=float(getattr(self.args, "p_threshold", 0.5))
        )

        # 2. Top-1 reliable neighbor fine-tuning + probability.
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

        # 3. Core/noisy split only by FedRN p_rn.
        threshold = float(getattr(self.args, "p_threshold", 0.5))
        core_idx, noisy_idx, core_mask, noisy_mask = self._select_core_and_noisy_by_prob(p_rn, threshold)

        # 4. Clean recovery from FedRN noisy pool.
        #    Newly recovered samples are saved into a persistent client-side recover pool.
        #    net1 trains on current S_core plus the persistent recover pool; net3 still trains on S_core.
        recover_interval = int(getattr(self.args, "recover_interval", 1))
        recover_interval = max(recover_interval, 1)
        do_recover = (
            self.args.g_epoch >= self.args.warmup_epochs
            and ((self.args.g_epoch - self.args.warmup_epochs) % recover_interval == 0)
        )
        persistent_recover = int(getattr(self.args, "recover_persistent", 1))

        if do_recover:
            new_recover_idx = self._select_clean_recovery_from_noisy(p_rn, noisy_mask, net_main, net_personal)
        else:
            # Even if not recovering this round, softly decay memory. The persistent pool itself is kept.
            self._ensure_recovery_memory()
            decay = int(getattr(self.args, "recover_memory_decay", 1))
            if decay > 0:
                self.recovery_memory = np.maximum(0, self.recovery_memory - decay)
            new_recover_idx = np.array([], dtype=np.int64)

        if persistent_recover:
            actually_added_idx = self._add_to_recovery_pool(new_recover_idx)
            self.last_new_recover_indices = self._safe_indices(actually_added_idx)
            recover_idx = self._get_recovery_pool_extra_indices(core_idx)
        else:
            # Backward-compatible behavior: recover only this round.
            recover_idx = self._safe_indices(new_recover_idx)
            self.last_new_recover_indices = self._safe_indices(new_recover_idx)
            self.last_recover_pool_indices = recover_idx

        # Optional variant: absorb recover into score/core.
        # If enabled, net1 trains on merged score = S_core union recovery_pool,
        # and net3 also trains on merged score because recover is now treated as score/core.
        if int(getattr(self.args, "recover_merge_into_core", 1)):
            w_main, loss_main, w_personal, loss_personal = self.train_two_models_merge_recover_into_score(
                net_main,
                net_personal,
                score_indices=core_idx,
                recover_indices=recover_idx,
            )
        else:
            w_main, loss_main, w_personal, loss_personal = self.train_two_models(
                net_main,
                net_personal,
                core_indices=core_idx,
                recover_indices=recover_idx,
            )

        self.set_expertise()
        self.set_arbitrary_output()

        agreement = self.get_global_local_agreement(self.net1, self.net3)
        agreement_ratio = float(np.mean(agreement)) if len(agreement) > 0 else 0.0
        # core_ratio is the actually used score/core ratio. In merge mode this is S_core union recover_pool.
        core_ratio = float(len(getattr(self, "last_core_indices", core_idx))) / float(len(self.data_indices) + 1e-8)
        recover_ratio = float(len(recover_idx)) / float(len(self.data_indices) + 1e-8)
        train_ratio = float(len(getattr(self, "last_main_train_indices", []))) / float(len(self.data_indices) + 1e-8)

        return (
            w_main,
            loss_main,
            w_personal,
            loss_personal,
            core_ratio,
            recover_ratio,
            train_ratio,
            do_recover,
            agreement_ratio,
        )


# =============================================================================
# Evaluation helper: optional holdout evaluation for net3.
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
    args = ensure_two_model_args(args)

    args.method = "pfedrn_two_model_clean_recovery"
    args.exp_method = "PFedRN-2M-CCR-MergeScore"
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    args.schedule = [int(x) for x in args.schedule]

    # Keep top-1 reliable neighbor and one cloud/global scope as in your recent simplified setting.
    args.num_neighbors = 2
    args.send_2_models = False
    args.num_edges = 1
    args.neighbor_scope = "global"

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    classes_per_user = args.num_shards // args.num_users if args.partition == "shard" else "dirichlet"
    save_dir = os.path.join("resultdate", "PFedRN_TwoModel_CleanRecovery")
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(
        save_dir,
        f"详细模型_PFedRN-2M-CCR-MergeScore{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
    )
    metrics_log_filename = os.path.join(
        save_dir,
        f"总体模型_PFedRN-2M-CCR-MergeScore{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
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
    original_train_labels = labels.copy()
    args.img_size = int(dataset_train[0][0].shape[1])

    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == "shard":
        dict_users = sample_noniid_shard(labels=labels, num_users=args.num_users, num_shards=args.num_shards)
    else:
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)

    # Keep no local holdout, to be comparable with FedRN / PFedRN global results.
    dict_users_train, dict_users_test = {}, {}
    for i in range(args.num_users):
        idxs = list(copy.deepcopy(dict_users[i]))
        rng = np.random.RandomState(args.seed + i)
        rng.shuffle(idxs)
        dict_users_train[i] = idxs
        dict_users_test[i] = []

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

    net_glob = get_global_model(args).to(args.device)
    head_param_names = [name for name, _ in net_glob.named_parameters() if ("linear" in name or "fc_" in name or "classifier" in name)]
    print("[AlignCheck] net1/global model class:", net_glob.__class__.__name__)
    print("[AlignCheck] net1/global head parameters:", head_param_names)
    print("[AlignCheck] selected clients train from current net_glob; local.net1 cache is preserved after aggregation.")
    initial_state = copy.deepcopy(net_glob.state_dict())
    gaussian_noise = torch.randn(1, args.num_channels, args.img_size, args.img_size).to(args.device)

    local_objects = []
    for i in range(args.num_users):
        local = LocalUpdateTwoModelCleanRecovery(
            args=args,
            user_idx=i,
            dataset=dataset_train,
            idxs=dict_users_train[i],
            gaussian_noise=gaussian_noise,
        )
        # LocalUpdatePFedRN's BaseLocalUpdate may create models from models.nets (dual-head).
        # Overwrite them explicitly with the original FedRN single-head model, so the
        # degenerate test is really aligned with original FedRN.
        local.net1 = get_global_model(args).to(args.device)
        local.net1.load_state_dict(initial_state)
        local.net3 = get_personal_model(args).to(args.device)
        local.net3.load_state_dict(initial_state)
        local.recovery_memory = np.zeros(len(local.data_indices), dtype=np.int32)
        local.recovery_pool = set()
        local.recovery_pool_order = []
        local.net3_reset_done = False
        local_objects.append(local)

    num_edges = max(args.num_edges, 1)
    clients_per_edge = args.num_users // num_edges
    all_client_ids = list(range(args.num_users))
    edge_clients_map = {}
    for e in range(num_edges):
        s = e * clients_per_edge
        edge_clients_map[e] = all_client_ids[s:] if e == num_edges - 1 else all_client_ids[s:s + clients_per_edge]

    print("\nStructure: {} edge server(s), neighbor_scope={}".format(num_edges, args.neighbor_scope))
    print("PFedRN-2M-CCR-MergeScore: net1 FedRN global + net3 personalized confirmer. Only net1 is aggregated.")
    print(
        "Recovery: interval={}, max_ratio={}, conf_main={}, conf_personal={}, min_p_rn={}, memory_threshold={}, "
        "persistent={}, pool_max_ratio={}, merge_into_core={}".format(
            args.recover_interval,
            args.recover_max_ratio,
            args.recover_conf_main,
            args.recover_conf_personal,
            args.recover_min_p_rn,
            args.recover_memory_threshold,
            getattr(args, "recover_persistent", 1),
            getattr(args, "recover_pool_max_ratio", 1.0),
            getattr(args, "recover_merge_into_core", 1),
        )
    )
    print("Net3: personal_train_mode={}, warmup_train_personal={}, reset_after_warmup={}\n".format(
        args.personal_train_mode,
        args.warmup_train_personal,
        args.reset_net3_after_warmup,
    ))

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Experiment Start: {timestamp}\n")
        f.write(f"Args: {args}\n")
        f.write(msg_actual_clean + "\n")
        f.write("=" * 80 + "\n")
    with open(metrics_log_filename, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,global_test_acc,global_test_loss,personal_apa,personal_loss,"
            "core_clean_ratio,recover_ratio,train_ratio,agreement_ratio,recovery_trigger,"
            "score_count,core_count,recover_count,new_recover_count,recover_pool_count,train_count,total_count,actual_train_clean_count,"
            "actual_round_clean_count,missed_clean_count,false_clean_count,total_wrong_count,"
            "actual_train_clean_ratio,clean_recall,recover_true_clean_count,recover_false_clean_count,recover_precision,"
            "new_recover_true_clean_count,new_recover_false_clean_count,new_recover_precision,recover_pool_true_clean_count,recover_pool_false_clean_count,recover_pool_precision\n"
        )

    # ---------------------------------------------------------------------
    # Federated training loop
    # ---------------------------------------------------------------------
    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}: {} => {}".format(epoch + 1, args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay
        args.g_epoch = epoch

        # Optional: reset net3 right after warmup, using the current net1 state as initialization.
        if epoch == args.warmup_epochs and int(getattr(args, "reset_net3_after_warmup", 1)):
            print("[Info] Resetting net3 from net1 at the start of phase2.")
            for local in local_objects:
                local.net3.load_state_dict(copy.deepcopy(local.net1.state_dict()), strict=False)
                local.recovery_memory = np.zeros(len(local.data_indices), dtype=np.int32)
                if int(getattr(args, "recover_clear_on_reset", 1)) and hasattr(local, "_clear_recovery_pool"):
                    local._clear_recovery_pool()
                local.net3_reset_done = True

        edge_weights, edge_samples = [], []
        edge_logs = []
        round_losses, round_core, round_recover, round_train_ratio = [], [], [], []
        round_recovery_trigger, round_agree = [], []
        round_score_count, round_core_count, round_recover_count, round_train_count = 0, 0, 0, 0
        round_total_count, round_actual_train_clean_count = 0, 0
        round_actual_clean_count, round_missed_clean_count = 0, 0
        round_false_clean_count, round_total_wrong_count = 0, 0
        round_recover_true_clean_count, round_recover_false_clean_count = 0, 0
        round_new_recover_count, round_new_recover_true_clean_count, round_new_recover_false_clean_count = 0, 0, 0
        round_recover_pool_count, round_recover_pool_true_clean_count, round_recover_pool_false_clean_count = 0, 0, 0

        for edge_id, current_edge_clients in edge_clients_map.items():
            m = max(int(args.frac * len(current_edge_clients)), 1)
            selected_clients = np.random.choice(current_edge_clients, m, replace=False)

            client_weights, client_samples = [], []
            client_losses, client_core, client_recover, client_train_ratio = [], [], [], []
            client_recovery_trigger, client_agree = [], []
            edge_score_count, edge_core_count, edge_recover_count, edge_train_count = 0, 0, 0, 0
            edge_total_count, edge_actual_train_clean_count = 0, 0
            edge_actual_clean_count, edge_missed_clean_count = 0, 0
            edge_false_clean_count, edge_total_wrong_count = 0, 0
            edge_recover_true_clean_count, edge_recover_false_clean_count = 0, 0
            edge_new_recover_count, edge_new_recover_true_clean_count, edge_new_recover_false_clean_count = 0, 0, 0
            edge_recover_pool_count, edge_recover_pool_true_clean_count, edge_recover_pool_false_clean_count = 0, 0, 0

            for client_idx in selected_clients:
                local = local_objects[client_idx]
                local.args = args

                # Original FedRN trains each selected client from the current global model,
                # while self.net1 remains the client's previous local model for GMM/expertise.
                # Do NOT start local training from local.net1 here.
                net_main = copy.deepcopy(net_glob).to(args.device)
                net_personal = copy.deepcopy(local.net3).to(args.device)
                sample_size = len(dict_users_train[client_idx])

                if epoch < args.warmup_epochs:
                    w_main, loss_main, w_personal, loss_personal = local.train_phase1_two(
                        net_main,
                        net_personal,
                    )
                    core_ratio = 1.0
                    recover_ratio = 0.0
                    train_ratio = 1.0
                    do_recover = False
                    agreement_ratio = 0.0
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

                    (
                        w_main,
                        loss_main,
                        w_personal,
                        loss_personal,
                        core_ratio,
                        recover_ratio,
                        train_ratio,
                        do_recover,
                        agreement_ratio,
                    ) = local.train_phase2_clean_recovery(
                        net_main,
                        net_personal,
                        self_score,
                        top1_neighbor,
                        top1_score,
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

                # Analysis-only sample-selection statistics.
                local_data_indices = np.asarray(local.data_indices, dtype=np.int64)
                main_train_indices = np.asarray(
                    getattr(local, "last_main_train_indices", local_data_indices), dtype=np.int64
                )
                core_indices_logged = np.asarray(
                    getattr(local, "last_core_indices", local_data_indices if epoch < args.warmup_epochs else []), dtype=np.int64
                )
                score_indices_logged = np.asarray(getattr(local, "last_score_indices", core_indices_logged), dtype=np.int64)
                recover_indices_logged = np.asarray(getattr(local, "last_recover_indices", []), dtype=np.int64)
                new_recover_indices_logged = np.asarray(getattr(local, "last_new_recover_indices", []), dtype=np.int64)
                recover_pool_indices_logged = np.asarray(getattr(local, "last_recover_pool_indices", []), dtype=np.int64)

                local_actual_clean_count = int(real_clean_mask[local_data_indices].sum()) if len(local_data_indices) else 0
                train_count_i = int(len(main_train_indices))
                actual_train_clean_i = int(real_clean_mask[main_train_indices].sum()) if train_count_i > 0 else 0
                false_clean_i = max(train_count_i - actual_train_clean_i, 0)
                missed_clean_i = max(local_actual_clean_count - actual_train_clean_i, 0)
                total_wrong_i = false_clean_i + missed_clean_i

                recover_count_i = int(len(recover_indices_logged))
                recover_true_clean_i = int(real_clean_mask[recover_indices_logged].sum()) if recover_count_i > 0 else 0
                recover_false_clean_i = max(recover_count_i - recover_true_clean_i, 0)

                new_recover_count_i = int(len(new_recover_indices_logged))
                new_recover_true_clean_i = int(real_clean_mask[new_recover_indices_logged].sum()) if new_recover_count_i > 0 else 0
                new_recover_false_clean_i = max(new_recover_count_i - new_recover_true_clean_i, 0)

                recover_pool_count_i = int(len(recover_pool_indices_logged))
                recover_pool_true_clean_i = int(real_clean_mask[recover_pool_indices_logged].sum()) if recover_pool_count_i > 0 else 0
                recover_pool_false_clean_i = max(recover_pool_count_i - recover_pool_true_clean_i, 0)

                edge_score_count += int(len(score_indices_logged))
                edge_core_count += int(len(core_indices_logged))
                edge_recover_count += recover_count_i
                edge_train_count += train_count_i
                edge_total_count += sample_size
                edge_actual_train_clean_count += actual_train_clean_i
                edge_actual_clean_count += local_actual_clean_count
                edge_missed_clean_count += missed_clean_i
                edge_false_clean_count += false_clean_i
                edge_total_wrong_count += total_wrong_i
                edge_recover_true_clean_count += recover_true_clean_i
                edge_recover_false_clean_count += recover_false_clean_i
                edge_new_recover_count += new_recover_count_i
                edge_new_recover_true_clean_count += new_recover_true_clean_i
                edge_new_recover_false_clean_count += new_recover_false_clean_i
                edge_recover_pool_count += recover_pool_count_i
                edge_recover_pool_true_clean_count += recover_pool_true_clean_i
                edge_recover_pool_false_clean_count += recover_pool_false_clean_i

            if len(client_weights) > 0:
                w_edge = fedavg(client_weights, client_samples, skip_local_head=True)
                edge_weights.append(w_edge)
                edge_samples.append(sum(client_samples))
                edge_actual_train_clean_ratio = edge_actual_train_clean_count / float(edge_train_count + 1e-8)
                edge_clean_recall = edge_actual_train_clean_count / float(edge_actual_clean_count + 1e-8)
                edge_recover_precision = edge_recover_true_clean_count / float(edge_recover_count + 1e-8)
                edge_new_recover_precision = edge_new_recover_true_clean_count / float(edge_new_recover_count + 1e-8)
                edge_recover_pool_precision = edge_recover_pool_true_clean_count / float(edge_recover_pool_count + 1e-8)
                edge_logs.append(
                    "  --> [Edge {:02d}] clients={} loss={:.4f} score={:.3f} core={:.3f} recover_train={:.3f} train={:.3f} agree={:.3f} "
                    "ScoreCount={} CleanCount={}/{} ActualTrainClean={:.3f} CleanRecall={:.3f} MissedClean={} FalseClean={} TotalWrong={} "
                    "RecoverTrainClean={}/{} RecoverTrainFalse={} RecoverTrainPrecision={:.3f} "
                    "NewRecoverClean={}/{} NewRecoverFalse={} NewRecoverPrecision={:.3f} "
                    "RecoverPoolClean={}/{} RecoverPoolFalse={} RecoverPoolPrecision={:.3f} trigger={:.0f}".format(
                        edge_id + 1,
                        len(client_weights),
                        float(np.mean(client_losses)),
                        edge_score_count / float(edge_total_count + 1e-8),
                        float(np.mean(client_core)),
                        float(np.mean(client_recover)),
                        float(np.mean(client_train_ratio)),
                        float(np.mean(client_agree)) if client_agree else 0.0,
                        edge_score_count,
                        edge_train_count,
                        edge_total_count,
                        edge_actual_train_clean_ratio,
                        edge_clean_recall,
                        edge_missed_clean_count,
                        edge_false_clean_count,
                        edge_total_wrong_count,
                        edge_recover_true_clean_count,
                        edge_recover_count,
                        edge_recover_false_clean_count,
                        edge_recover_precision,
                        edge_new_recover_true_clean_count,
                        edge_new_recover_count,
                        edge_new_recover_false_clean_count,
                        edge_new_recover_precision,
                        edge_recover_pool_true_clean_count,
                        edge_recover_pool_count,
                        edge_recover_pool_false_clean_count,
                        edge_recover_pool_precision,
                        float(np.max(client_recovery_trigger)) if client_recovery_trigger else 0.0,
                    )
                )
                round_losses.extend(client_losses)
                round_core.extend(client_core)
                round_recover.extend(client_recover)
                round_train_ratio.extend(client_train_ratio)
                round_recovery_trigger.extend(client_recovery_trigger)
                round_agree.extend(client_agree)
                round_score_count += edge_score_count
                round_core_count += edge_core_count
                round_recover_count += edge_recover_count
                round_train_count += edge_train_count
                round_total_count += edge_total_count
                round_actual_train_clean_count += edge_actual_train_clean_count
                round_actual_clean_count += edge_actual_clean_count
                round_missed_clean_count += edge_missed_clean_count
                round_false_clean_count += edge_false_clean_count
                round_total_wrong_count += edge_total_wrong_count
                round_recover_true_clean_count += edge_recover_true_clean_count
                round_recover_false_clean_count += edge_recover_false_clean_count
                round_new_recover_count += edge_new_recover_count
                round_new_recover_true_clean_count += edge_new_recover_true_clean_count
                round_new_recover_false_clean_count += edge_new_recover_false_clean_count
                round_recover_pool_count += edge_recover_pool_count
                round_recover_pool_true_clean_count += edge_recover_pool_true_clean_count
                round_recover_pool_false_clean_count += edge_recover_pool_false_clean_count

        if len(edge_weights) > 0:
            w_glob = fedavg(edge_weights, edge_samples, skip_local_head=True)
            net_glob.load_state_dict(w_glob, strict=False)

            # Keep local.net1 as the client's cached local model, exactly like original FedRN.
            # Do NOT overwrite local.net1 with the aggregated global model here; otherwise
            # fit_gmm(self.net1) uses a global model while expertise/arbitrary_output came
            # from a previous local model, which breaks FedRN's reliability logic.
            for local in local_objects:
                sync_global_to_client(local.net3, w_glob, mode="backbone_only")

        global_train_acc, global_train_loss = test_img(net_glob, log_train_loader, args)
        global_test_acc, global_test_loss = test_img(net_glob, log_test_loader, args)
        personal_apa, personal_loss = evaluate_local_model(
            local_objects, dataset_train, dict_users_test, args, model_name="net3", head="local"
        )

        train_loss = float(np.mean(round_losses)) if round_losses else 0.0
        core_ratio = float(np.mean(round_core)) if round_core else 0.0
        recover_ratio = float(np.mean(round_recover)) if round_recover else 0.0
        train_ratio = float(np.mean(round_train_ratio)) if round_train_ratio else 0.0
        recovery_trigger = float(np.max(round_recovery_trigger)) if round_recovery_trigger else 0.0
        agree_ratio = float(np.mean(round_agree)) if round_agree else 0.0
        actual_train_clean_ratio = round_actual_train_clean_count / float(round_train_count + 1e-8)
        clean_recall = round_actual_train_clean_count / float(round_actual_clean_count + 1e-8)
        recover_precision = round_recover_true_clean_count / float(round_recover_count + 1e-8)
        new_recover_precision = round_new_recover_true_clean_count / float(round_new_recover_count + 1e-8)
        recover_pool_precision = round_recover_pool_true_clean_count / float(round_recover_pool_count + 1e-8)

        log_round = "\n==================== Round {:3d} ====================".format(epoch)
        log_metric = (
            "Global Acc: {:.2f}% | Global Loss: {:.4f} | Personal APA: {:.2f}% | "
            "Train Loss: {:.4f} | ScoreClean: {:.3f} | CoreClean: {:.3f} | RecoverTrain: {:.3f} | TrainRatio: {:.3f} | Agree: {:.3f} | "
            "ScoreCount: {} | CleanCount: {}/{} | ActualTrainClean: {:.3f} ({}/{}) | "
            "ActualCleanInRound: {} | CleanRecall: {:.3f} | MissedClean: {} | FalseClean: {} | TotalWrong: {} | "
            "RecoverTrainClean: {}/{} | RecoverTrainFalse: {} | RecoverTrainPrecision: {:.3f} | "
            "NewRecoverClean: {}/{} | NewRecoverFalse: {} | NewRecoverPrecision: {:.3f} | "
            "RecoverPoolClean: {}/{} | RecoverPoolFalse: {} | RecoverPoolPrecision: {:.3f} | RecoveryTrigger: {:.0f}"
        ).format(
            global_test_acc,
            global_test_loss,
            personal_apa,
            train_loss,
            round_score_count / float(round_total_count + 1e-8),
            core_ratio,
            recover_ratio,
            train_ratio,
            agree_ratio,
            round_score_count,
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
            round_recover_true_clean_count,
            round_recover_count,
            round_recover_false_clean_count,
            recover_precision,
            round_new_recover_true_clean_count,
            round_new_recover_count,
            round_new_recover_false_clean_count,
            new_recover_precision,
            round_recover_pool_true_clean_count,
            round_recover_pool_count,
            round_recover_pool_false_clean_count,
            recover_pool_precision,
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
                f"{personal_apa:.6f},{personal_loss:.6f},"
                f"{core_ratio:.6f},{recover_ratio:.6f},{train_ratio:.6f},{agree_ratio:.6f},{recovery_trigger:.0f},"
                f"{round_score_count},{round_core_count},{round_recover_count},{round_new_recover_count},{round_recover_pool_count},{round_train_count},{round_total_count},"
                f"{round_actual_train_clean_count},{round_actual_clean_count},{round_missed_clean_count},"
                f"{round_false_clean_count},{round_total_wrong_count},{actual_train_clean_ratio:.6f},{clean_recall:.6f},"
                f"{round_recover_true_clean_count},{round_recover_false_clean_count},{recover_precision:.6f},"
                f"{round_new_recover_true_clean_count},{round_new_recover_false_clean_count},{new_recover_precision:.6f},"
                f"{round_recover_pool_true_clean_count},{round_recover_pool_false_clean_count},{recover_pool_precision:.6f}\n"
            )

    print("Total time: {:.1f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()