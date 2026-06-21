#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedRN score baseline with clean local holdout.

This diagnostic script keeps the original FedRN training path, but changes the
data preparation:
1. Partition the clean dataset into clients.
2. Split each client's clean data into 80% train and 20% holdout.
3. Build a clean holdout evaluation view with original labels and deterministic
   validation transforms.
4. Inject label noise only into the 80% train split.
5. Run original FedRN on the noisy train split only.
6. Log FedRN score/core quality and evaluate the global model on the clean
   local holdout split.

No net2 and no recovery are used here. The goal is to decide whether FedRN's
score set is a good future training source for a local confirmation model.
"""

import copy
import datetime
import os
import random
import sys
import time

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
import torchvision
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from utils import load_dataset
from utils.options_fedrn_score_holdout_baseline import args_parser
from utils.sampling import sample_dirichlet, sample_iid, sample_noniid_shard
from utils.utils import noisify_label

from models.fed import LocalModelWeights
from models.nets_original_fedrn import get_model
from models.test import test_img
from models.update_original_fedrn import DatasetSplit, LocalUpdateFedRN


def _get_train_labels(dataset):
    if hasattr(dataset, "train_labels"):
        return dataset.train_labels
    if hasattr(dataset, "targets"):
        return dataset.targets
    raise AttributeError("dataset has neither train_labels nor targets")


def _set_train_label(dataset, idx, value):
    if hasattr(dataset, "train_labels"):
        dataset.train_labels[idx] = int(value)
    if hasattr(dataset, "targets"):
        dataset.targets[idx] = int(value)


def _ensure_label_alias(dataset):
    if hasattr(dataset, "targets") and not hasattr(dataset, "train_labels"):
        dataset.train_labels = dataset.targets
    if hasattr(dataset, "train_labels") and not hasattr(dataset, "targets"):
        dataset.targets = dataset.train_labels


def make_clean_eval_dataset(dataset_train, dataset_test, original_labels):
    dataset_eval = copy.copy(dataset_train)
    clean_labels = [int(x) for x in original_labels]
    if hasattr(dataset_eval, "train_labels"):
        dataset_eval.train_labels = clean_labels
    if hasattr(dataset_eval, "targets"):
        dataset_eval.targets = clean_labels
    if hasattr(dataset_test, "transform"):
        dataset_eval.transform = dataset_test.transform
    return dataset_eval


def _safe_minmax(values):
    values = list(values)
    v_min, v_max = min(values), max(values)
    denom = v_max - v_min
    if abs(denom) < 1e-12:
        return [0.0 for _ in values]
    return [(v - v_min) / denom for v in values]


def _safe_indices(indices):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return indices
    return np.unique(indices)


def _flatten_user_dict(user_dict):
    all_indices = []
    for idxs in user_dict.values():
        all_indices.extend(list(idxs))
    return _safe_indices(all_indices)


def split_clients_before_noise(dict_users, train_ratio, seed):
    train_ratio = min(max(float(train_ratio), 0.0), 1.0)
    dict_users_train, dict_users_holdout = {}, {}
    for user, idxs in dict_users.items():
        idxs = list(idxs)
        rng = np.random.RandomState(seed + int(user))
        rng.shuffle(idxs)
        if len(idxs) <= 1:
            train_n = len(idxs)
        else:
            train_n = int(round(len(idxs) * train_ratio))
            train_n = min(max(train_n, 1), len(idxs) - 1)
        dict_users_train[user] = list(idxs[:train_n])
        dict_users_holdout[user] = list(idxs[train_n:])
    return dict_users_train, dict_users_holdout


class LocalUpdateFedRNScoreHoldout(LocalUpdateFedRN):
    """Original LocalUpdateFedRN plus score-quality logging."""

    def __init__(self, *args, real_clean_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_clean_mask = real_clean_mask
        self.last_stats = self._empty_stats(stage="init")
        self.last_score_indices = np.array([], dtype=np.int64)

    def _empty_stats(self, stage="unknown"):
        total = int(len(self.data_indices)) if hasattr(self, "data_indices") else 0
        return dict(
            user_idx=int(self.user_idx) if self.user_idx is not None else -1,
            stage=stage,
            total_count=total,
            actual_round_clean_count=0,
            actual_round_noisy_count=0,
            fedrn_clean_count=0,
            fedrn_noisy_count=0,
            train_count=0,
            actual_train_clean_count=0,
            actual_train_noisy_count=0,
            false_clean_count=0,
            missed_clean_count=0,
            correctly_excluded_noisy_count=0,
            actual_train_clean_ratio=0.0,
            clean_recall=0.0,
            train_ratio=0.0,
            fedrn_clean_ratio=0.0,
            fedrn_noisy_ratio=0.0,
            initial_fedrn_clean_count=0,
            initial_fedrn_noisy_count=0,
            holdout_acc=-1.0,
            holdout_loss=-1.0,
            holdout_count=0,
        )

    def _compute_selection_stats(self, train_indices, noisy_indices=None, stage="phase2"):
        data_indices = np.asarray(self.data_indices, dtype=np.int64)
        train_indices = np.asarray(train_indices, dtype=np.int64)

        if noisy_indices is None:
            train_set = set(int(x) for x in train_indices)
            noisy_indices = np.asarray([idx for idx in data_indices if int(idx) not in train_set], dtype=np.int64)
        else:
            noisy_indices = np.asarray(noisy_indices, dtype=np.int64)

        total_count = int(len(data_indices))
        fedrn_clean_count = int(len(train_indices))
        fedrn_noisy_count = int(len(noisy_indices))

        if self.real_clean_mask is None:
            stats = self._empty_stats(stage=stage)
            stats.update(
                fedrn_clean_count=fedrn_clean_count,
                fedrn_noisy_count=fedrn_noisy_count,
                train_count=fedrn_clean_count,
                train_ratio=fedrn_clean_count / float(max(total_count, 1)),
                fedrn_clean_ratio=fedrn_clean_count / float(max(total_count, 1)),
                fedrn_noisy_ratio=fedrn_noisy_count / float(max(total_count, 1)),
            )
            return stats

        real_clean_mask = np.asarray(self.real_clean_mask)
        actual_round_clean_count = int(real_clean_mask[data_indices].sum()) if total_count else 0
        actual_round_noisy_count = int(total_count - actual_round_clean_count)
        actual_train_clean_count = int(real_clean_mask[train_indices].sum()) if fedrn_clean_count else 0
        actual_train_noisy_count = int(fedrn_clean_count - actual_train_clean_count)
        missed_clean_count = int(real_clean_mask[noisy_indices].sum()) if fedrn_noisy_count else 0
        correctly_excluded_noisy_count = int(fedrn_noisy_count - missed_clean_count)

        return dict(
            user_idx=int(self.user_idx),
            stage=stage,
            total_count=total_count,
            actual_round_clean_count=actual_round_clean_count,
            actual_round_noisy_count=actual_round_noisy_count,
            fedrn_clean_count=fedrn_clean_count,
            fedrn_noisy_count=fedrn_noisy_count,
            train_count=fedrn_clean_count,
            actual_train_clean_count=actual_train_clean_count,
            actual_train_noisy_count=actual_train_noisy_count,
            false_clean_count=actual_train_noisy_count,
            missed_clean_count=missed_clean_count,
            correctly_excluded_noisy_count=correctly_excluded_noisy_count,
            actual_train_clean_ratio=actual_train_clean_count / float(max(fedrn_clean_count, 1)),
            clean_recall=actual_train_clean_count / float(max(actual_round_clean_count, 1)),
            train_ratio=fedrn_clean_count / float(max(total_count, 1)),
            fedrn_clean_ratio=fedrn_clean_count / float(max(total_count, 1)),
            fedrn_noisy_ratio=fedrn_noisy_count / float(max(total_count, 1)),
            initial_fedrn_clean_count=0,
            initial_fedrn_noisy_count=0,
            holdout_acc=-1.0,
            holdout_loss=-1.0,
            holdout_count=0,
        )

    def fit_gmm(self, net):
        losses = []
        net.eval()
        with torch.no_grad():
            for inputs, targets, items, idxs in self.ldr_eval:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = net(inputs)
                loss = self.CE(outputs, targets)
                losses.append(loss)

        losses = torch.cat(losses).detach().cpu().numpy()
        denom = losses.max() - losses.min()
        if abs(float(denom)) < 1e-12:
            losses = np.zeros_like(losses, dtype=np.float64)
        else:
            losses = (losses - losses.min()) / denom

        input_loss = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        return prob[:, gmm.means_.argmin()]

    def train_phase1(self, net):
        w, loss = self.train_single_model(net)
        self.set_expertise()
        self.set_arbitrary_output()
        self.last_score_indices = np.asarray(self.data_indices, dtype=np.int64)
        self.last_stats = self._compute_selection_stats(
            train_indices=self.data_indices,
            noisy_indices=np.array([], dtype=np.int64),
            stage="warmup_all_data",
        )
        return w, loss

    def train_phase2(self, net, prev_score, neighbor_list, neighbor_score_list):
        prob = self.fit_gmm(self.net1)
        pred_clean_idx, pred_noisy_idx = self.get_clean_idx(prob)

        prob_list = [prob]
        neighbor_list = self.finetune_head(neighbor_list, pred_clean_idx)
        for neighbor_net in neighbor_list:
            neighbor_prob = self.fit_gmm(neighbor_net)
            prob_list.append(neighbor_prob)

        score_list = [float(prev_score)] + [float(s) for s in neighbor_score_list]
        score_sum = float(sum(score_list)) + 1e-12
        score_list = [score / score_sum for score in score_list]

        final_prob = np.zeros(len(prob), dtype=np.float64)
        for p, score in zip(prob_list, score_list):
            final_prob = np.add(final_prob, np.multiply(p, score))

        final_clean_idx, final_noisy_idx = self.get_clean_idx(final_prob)
        self.last_score_indices = _safe_indices(final_clean_idx)
        self.last_stats = self._compute_selection_stats(
            train_indices=final_clean_idx,
            noisy_indices=final_noisy_idx,
            stage="fedrn_selection",
        )
        self.last_stats["initial_fedrn_clean_count"] = int(len(pred_clean_idx))
        self.last_stats["initial_fedrn_noisy_count"] = int(len(pred_noisy_idx))

        self.ldr_train = DataLoader(
            DatasetSplit(self.dataset, final_clean_idx, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        w, loss = self.train_single_model(net)
        self.set_expertise()
        self.set_arbitrary_output()
        return w, loss


def get_local_update_objects(args, dataset_train, dict_users_train, gaussian_noise, real_clean_mask):
    local_update_objects = []
    for idx in range(args.num_users):
        local_update_object = LocalUpdateFedRNScoreHoldout(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users_train[idx],
            gaussian_noise=gaussian_noise,
            real_clean_mask=real_clean_mask,
        )
        local_update_objects.append(local_update_object)
    return local_update_objects


def _sum_stats(stats_list):
    keys = [
        "total_count",
        "actual_round_clean_count",
        "actual_round_noisy_count",
        "fedrn_clean_count",
        "fedrn_noisy_count",
        "train_count",
        "actual_train_clean_count",
        "actual_train_noisy_count",
        "false_clean_count",
        "missed_clean_count",
        "correctly_excluded_noisy_count",
        "holdout_count",
    ]
    agg = {k: int(sum(int(s.get(k, 0)) for s in stats_list)) for k in keys}
    agg["train_ratio"] = agg["train_count"] / float(max(agg["total_count"], 1))
    agg["fedrn_clean_ratio"] = agg["fedrn_clean_count"] / float(max(agg["total_count"], 1))
    agg["fedrn_noisy_ratio"] = agg["fedrn_noisy_count"] / float(max(agg["total_count"], 1))
    agg["actual_train_clean_ratio"] = agg["actual_train_clean_count"] / float(max(agg["train_count"], 1))
    agg["clean_recall"] = agg["actual_train_clean_count"] / float(max(agg["actual_round_clean_count"], 1))
    agg["actual_round_clean_ratio"] = agg["actual_round_clean_count"] / float(max(agg["total_count"], 1))
    agg["actual_round_noisy_ratio"] = agg["actual_round_noisy_count"] / float(max(agg["total_count"], 1))
    return agg


def _format_round_stats(agg):
    return (
        "FedRNClean={fedrn_clean_count}/{total_count}({fedrn_clean_ratio:.3f}) | "
        "FedRNNoisy={fedrn_noisy_count}/{total_count}({fedrn_noisy_ratio:.3f}) | "
        "ActualRoundClean={actual_round_clean_count}/{total_count}({actual_round_clean_ratio:.3f}) | "
        "ActualRoundNoisy={actual_round_noisy_count}/{total_count}({actual_round_noisy_ratio:.3f}) | "
        "ActualTrainClean={actual_train_clean_ratio:.3f}({actual_train_clean_count}/{train_count}) | "
        "CleanRecall={clean_recall:.3f} | "
        "MissedClean={missed_clean_count} | "
        "FalseClean={false_clean_count}"
    ).format(**agg)


def _client_stats_line(stats):
    return (
        "    [Client {user_idx:03d}] stage={stage} "
        "Train={train_count}/{total_count} FedRNClean={fedrn_clean_count} FedRNNoisy={fedrn_noisy_count} "
        "ActualRoundClean={actual_round_clean_count} ActualRoundNoisy={actual_round_noisy_count} "
        "ActualTrainClean={actual_train_clean_ratio:.3f}({actual_train_clean_count}/{train_count}) "
        "CleanRecall={clean_recall:.3f} MissedClean={missed_clean_count} FalseClean={false_clean_count} "
        "HoldoutAcc={holdout_acc:.2f}% HoldoutN={holdout_count}"
    ).format(**stats)


def evaluate_subset(net, dataset, indices, args):
    indices = list(indices)
    if len(indices) == 0:
        return -1.0, -1.0
    loader = DataLoader(
        DatasetSplit(dataset, indices),
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return test_img(net, loader, args)


def annotate_selected_holdout(stats_list, selected_clients, net, dataset, dict_users_holdout, args):
    stats_by_client = {int(s["user_idx"]): s for s in stats_list}
    for client_idx in selected_clients:
        holdout_indices = dict_users_holdout[int(client_idx)]
        acc, loss = evaluate_subset(net, dataset, holdout_indices, args)
        stats_by_client[int(client_idx)]["holdout_acc"] = acc
        stats_by_client[int(client_idx)]["holdout_loss"] = loss
        stats_by_client[int(client_idx)]["holdout_count"] = int(len(holdout_indices))


def get_fedrn_score_indices(local, user_idx, epoch, args, local_update_objects, cosine_similarity):
    prob = local.fit_gmm(local.net1)
    self_clean_idx, _ = local.get_clean_idx(prob)
    final_clean_idx = self_clean_idx
    score_stage = "self_gmm"

    if int(args.snapshot_with_neighbors) and epoch >= int(args.warmup_epochs):
        sim_list, exp_list = [], []
        for other in local_update_objects:
            sim = cosine_similarity(
                local.arbitrary_output.to(args.device),
                other.arbitrary_output.to(args.device),
            ).item()
            sim_list.append(sim)
            exp_list.append(other.expertise)

        sim_list = _safe_minmax(sim_list)
        exp_list = _safe_minmax(exp_list)
        prev_score = args.w_alpha * exp_list[int(user_idx)] + (1.0 - args.w_alpha)

        neighbor_scores = []
        for neighbor_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
            if neighbor_idx == int(user_idx):
                continue
            score = args.w_alpha * exp + (1.0 - args.w_alpha) * sim
            neighbor_scores.append((score, neighbor_idx))
        neighbor_scores.sort(key=lambda x: x[0], reverse=True)

        prob_list = [prob]
        score_list = [float(prev_score)]
        for score, neighbor_idx in neighbor_scores[: args.num_neighbors]:
            neighbor_net = copy.deepcopy(local_update_objects[neighbor_idx].net1)
            neighbor_net = local.finetune_head([neighbor_net], self_clean_idx)[0]
            prob_list.append(local.fit_gmm(neighbor_net))
            score_list.append(float(score))

        score_sum = float(sum(score_list)) + 1e-12
        final_prob = np.zeros(len(prob), dtype=np.float64)
        for p, score in zip(prob_list, score_list):
            final_prob = np.add(final_prob, np.multiply(p, score / score_sum))
        final_clean_idx, _ = local.get_clean_idx(final_prob)
        score_stage = "fedrn_neighbor_fusion"

    return _safe_indices(self_clean_idx), _safe_indices(final_clean_idx), score_stage


def export_score_snapshot(
    snapshot_path,
    epoch,
    args,
    local_update_objects,
    original_labels,
    noisy_labels,
    real_clean_mask,
    cosine_similarity,
):
    with open(snapshot_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,user_idx,global_idx,self_gmm_clean,fedrn_final_clean,is_actual_clean,"
            "original_label,noisy_label,score_stage\n"
        )
        for user_idx, local in enumerate(local_update_objects):
            self_clean_idx, final_clean_idx, score_stage = get_fedrn_score_indices(
                local=local,
                user_idx=user_idx,
                epoch=epoch,
                args=args,
                local_update_objects=local_update_objects,
                cosine_similarity=cosine_similarity,
            )
            self_clean_set = set(int(x) for x in self_clean_idx)
            final_clean_set = set(int(x) for x in final_clean_idx)

            for idx in local.data_indices:
                idx = int(idx)
                f.write(
                    "{},{},{},{},{},{},{},{},{}\n".format(
                        epoch,
                        user_idx,
                        idx,
                        int(idx in self_clean_set),
                        int(idx in final_clean_set),
                        int(bool(real_clean_mask[idx])),
                        int(original_labels[idx]),
                        int(noisy_labels[idx]),
                        score_stage,
                    )
                )


def train_net2_on_score_set(net2, dataset_train, score_indices, args):
    score_indices = list(score_indices)
    if len(score_indices) == 0:
        return 0.0

    loader = DataLoader(
        DatasetSplit(dataset_train, score_indices, real_idx_return=True),
        batch_size=args.local_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    net2.train()
    optimizer = torch.optim.SGD(
        net2.parameters(),
        lr=float(getattr(args, "net2_lr", args.lr)),
        momentum=float(args.momentum),
        weight_decay=float(getattr(args, "net2_weight_decay", args.weight_decay)),
    )
    loss_func = torch.nn.CrossEntropyLoss()
    losses = []
    local_ep = int(getattr(args, "_active_net2_ep", getattr(args, "net2_local_ep", 1)))
    for _ in range(local_ep):
        batch_losses = []
        for inputs, targets, _items, _idxs in loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            net2.zero_grad()
            outputs = net2(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        if batch_losses:
            losses.append(float(np.mean(batch_losses)))
    return float(np.mean(losses)) if losses else 0.0


def init_personal_net2(args, net_glob, local):
    net2 = get_model(args).to(args.device)
    init_mode = str(getattr(args, "net2_init", "global"))
    if init_mode == "local_net1":
        net2.load_state_dict(copy.deepcopy(local.net1.state_dict()), strict=False)
    elif init_mode == "global":
        net2.load_state_dict(copy.deepcopy(net_glob.state_dict()), strict=False)
    return net2


def update_continuous_net2_for_client(
    local,
    net2,
    dataset_train,
    dataset_holdout_eval,
    holdout_indices,
    real_clean_mask,
    args,
):
    score_indices = _safe_indices(getattr(local, "last_score_indices", np.array([], dtype=np.int64)))
    train_loss = train_net2_on_score_set(net2, dataset_train, score_indices, args)
    holdout_acc, holdout_loss = evaluate_subset(net2, dataset_holdout_eval, holdout_indices, args)

    score_count = int(len(score_indices))
    score_clean = int(np.asarray(real_clean_mask)[score_indices].sum()) if score_count else 0
    local_actual_clean = int(np.asarray(real_clean_mask)[local.data_indices].sum())
    score_precision = score_clean / float(max(score_count, 1))
    score_recall = score_clean / float(max(local_actual_clean, 1))
    return dict(
        user_idx=int(local.user_idx),
        score_count=score_count,
        score_actual_clean_count=score_clean,
        score_actual_noisy_count=int(score_count - score_clean),
        score_precision=score_precision,
        score_recall=score_recall,
        holdout_count=int(len(holdout_indices)),
        net2_holdout_acc=float(holdout_acc),
        net2_holdout_loss=float(holdout_loss),
        net2_train_loss=float(train_loss),
    )


def summarize_net2_round(rows):
    if not rows:
        return {}
    return dict(
        clients=len(rows),
        avg_score_count=float(np.mean([r["score_count"] for r in rows])),
        avg_score_precision=float(np.mean([r["score_precision"] for r in rows])),
        avg_score_recall=float(np.mean([r["score_recall"] for r in rows])),
        avg_holdout_acc=float(np.mean([r["net2_holdout_acc"] for r in rows])),
        avg_holdout_loss=float(np.mean([r["net2_holdout_loss"] for r in rows])),
        avg_train_loss=float(np.mean([r["net2_train_loss"] for r in rows])),
    )


def append_net2_round_rows(path, epoch, rows):
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(
                "epoch,user_idx,score_count,score_actual_clean_count,score_actual_noisy_count,"
                "score_precision,score_recall,holdout_count,net2_holdout_acc,"
                "net2_holdout_loss,net2_train_loss\n"
            )
        for row in rows:
            f.write(
                "{},{user_idx},{score_count},{score_actual_clean_count},{score_actual_noisy_count},"
                "{score_precision:.6f},{score_recall:.6f},{holdout_count},"
                "{net2_holdout_acc:.6f},{net2_holdout_loss:.6f},{net2_train_loss:.6f}\n".format(
                    epoch,
                    **row,
                )
            )


def run_net2_score_eval(
    epoch,
    args,
    save_dir,
    timestamp,
    net_glob,
    dataset_train,
    dataset_holdout_eval,
    dict_users_holdout,
    local_update_objects,
    real_clean_mask,
    cosine_similarity,
):
    client_ids = list(range(args.num_users))
    if int(args.net2_eval_max_clients) > 0:
        client_ids = client_ids[: int(args.net2_eval_max_clients)]

    result_path = os.path.join(save_dir, "net2_score_eval_round{}_{}.csv".format(epoch, timestamp))
    rows = []
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,user_idx,score_stage,score_count,score_actual_clean_count,score_actual_noisy_count,"
            "score_precision,score_recall,holdout_count,net2_holdout_acc,net2_holdout_loss,net2_train_loss\n"
        )
        for user_idx in client_ids:
            local = local_update_objects[int(user_idx)]
            _self_idx, score_indices, score_stage = get_fedrn_score_indices(
                local=local,
                user_idx=user_idx,
                epoch=epoch,
                args=args,
                local_update_objects=local_update_objects,
                cosine_similarity=cosine_similarity,
            )
            score_indices = _safe_indices(score_indices)
            holdout_indices = list(dict_users_holdout[int(user_idx)])

            if str(args.net2_eval_init) == "local_net1":
                net2 = copy.deepcopy(local.net1).to(args.device)
            else:
                net2 = get_model(args).to(args.device)
                if str(args.net2_eval_init) == "global":
                    net2.load_state_dict(copy.deepcopy(net_glob.state_dict()), strict=False)

            old_active_ep = getattr(args, "_active_net2_ep", None)
            args._active_net2_ep = int(getattr(args, "net2_eval_ep", getattr(args, "net2_local_ep", 1)))
            net2_train_loss = train_net2_on_score_set(net2, dataset_train, score_indices, args)
            if old_active_ep is None:
                delattr(args, "_active_net2_ep")
            else:
                args._active_net2_ep = old_active_ep
            holdout_acc, holdout_loss = evaluate_subset(
                net2,
                dataset_holdout_eval,
                holdout_indices,
                args,
            )

            score_count = int(len(score_indices))
            actual_clean_count = int(np.asarray(real_clean_mask)[score_indices].sum()) if score_count else 0
            actual_noisy_count = int(score_count - actual_clean_count)
            local_actual_clean = int(np.asarray(real_clean_mask)[local.data_indices].sum())
            score_precision = actual_clean_count / float(max(score_count, 1))
            score_recall = actual_clean_count / float(max(local_actual_clean, 1))
            row = dict(
                epoch=int(epoch),
                user_idx=int(user_idx),
                score_stage=score_stage,
                score_count=score_count,
                score_actual_clean_count=actual_clean_count,
                score_actual_noisy_count=actual_noisy_count,
                score_precision=score_precision,
                score_recall=score_recall,
                holdout_count=int(len(holdout_indices)),
                net2_holdout_acc=float(holdout_acc),
                net2_holdout_loss=float(holdout_loss),
                net2_train_loss=float(net2_train_loss),
            )
            rows.append(row)
            f.write(
                "{epoch},{user_idx},{score_stage},{score_count},{score_actual_clean_count},"
                "{score_actual_noisy_count},{score_precision:.6f},{score_recall:.6f},"
                "{holdout_count},{net2_holdout_acc:.6f},{net2_holdout_loss:.6f},{net2_train_loss:.6f}\n".format(
                    **row
                )
            )

    if not rows:
        return result_path, {}

    summary = dict(
        clients=len(rows),
        avg_score_count=float(np.mean([r["score_count"] for r in rows])),
        avg_score_precision=float(np.mean([r["score_precision"] for r in rows])),
        avg_score_recall=float(np.mean([r["score_recall"] for r in rows])),
        avg_holdout_acc=float(np.mean([r["net2_holdout_acc"] for r in rows])),
        avg_train_loss=float(np.mean([r["net2_train_loss"] for r in rows])),
    )
    return result_path, summary


def build_noise_schedule(args):
    if sum(args.noise_group_num) != args.num_users:
        raise ValueError("sum(args.noise_group_num) must equal args.num_users")

    if len(args.group_noise_rate) == 1:
        args.group_noise_rate = args.group_noise_rate * 2

    args.group_noise_rate = [
        (args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
        for i in range(len(args.group_noise_rate) // 2)
    ]

    if len(args.group_noise_rate) != len(args.noise_group_num):
        raise ValueError("group_noise_rate must provide min/max for every noise group")

    user_noise_type_rates = []
    for num_users_in_group, noise_type, (min_r, max_r) in zip(
        args.noise_group_num, args.noise_type_lst, args.group_noise_rate
    ):
        step = (max_r - min_r) / max(num_users_in_group, 1)
        rates = np.array(range(num_users_in_group)) * step + min_r
        user_noise_type_rates += list(zip([noise_type] * num_users_in_group, rates))
    return user_noise_type_rates


def inject_noise_on_train_split(args, dataset_train, dict_users_train, user_noise_type_rates):
    for user, (noise_type, noise_rate) in enumerate(user_noise_type_rates):
        if noise_type == "clean":
            continue

        data_indices = list(copy.deepcopy(dict_users_train[user]))
        if args.noise_seed_mode == "client":
            random.seed(args.seed + int(user))
        else:
            random.seed(args.seed)
        random.shuffle(data_indices)
        noise_num = int(len(data_indices) * noise_rate)

        for d_idx in data_indices[:noise_num]:
            true_label = int(_get_train_labels(dataset_train)[d_idx])
            noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=noise_type)
            _set_train_label(dataset_train, d_idx, noisy_label)


def main():
    start = time.time()
    args = args_parser()
    args.method = "fedrn"
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    args.schedule = [int(x) for x in args.schedule]
    args.send_2_models = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print("FedRN score holdout baseline")
    for item in vars(args).items():
        print(item)
    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    _ensure_label_alias(dataset_train)
    _ensure_label_alias(dataset_test)

    original_train_labels = np.array(copy.deepcopy(list(_get_train_labels(dataset_train))), dtype=np.int64)
    labels = np.array(_get_train_labels(dataset_train), dtype=np.int64)
    args.img_size = int(dataset_train[0][0].shape[1])

    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == "shard":
        dict_users = sample_noniid_shard(labels=labels, num_users=args.num_users, num_shards=args.num_shards)
    elif args.partition == "dirichlet":
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)
    else:
        raise ValueError("Unsupported partition: {}".format(args.partition))

    dict_users_train, dict_users_holdout = split_clients_before_noise(
        dict_users=dict_users,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    all_train_indices = _flatten_user_dict(dict_users_train)
    all_holdout_indices = _flatten_user_dict(dict_users_holdout)

    user_noise_type_rates = build_noise_schedule(args)
    inject_noise_on_train_split(args, dataset_train, dict_users_train, user_noise_type_rates)

    noisy_train_labels = np.array(copy.deepcopy(list(_get_train_labels(dataset_train))), dtype=np.int64)
    real_clean_mask = original_train_labels == noisy_train_labels

    train_actual_clean = int(real_clean_mask[all_train_indices].sum())
    train_total = int(len(all_train_indices))
    train_actual_noisy = train_total - train_actual_clean
    holdout_clean = int(real_clean_mask[all_holdout_indices].sum()) if len(all_holdout_indices) else 0

    classes_per_user = args.num_shards // args.num_users if args.partition == "shard" else "dirichlet"
    log_filename = os.path.join(
        save_dir,
        "detail_FedRN_score_holdout{}_{}_{}_[{}]_{}.txt".format(
            args.epochs,
            args.dataset,
            classes_per_user,
            ",".join("{}-{}".format(a, b) for a, b in args.group_noise_rate),
            timestamp,
        ),
    )
    metrics_log_filename = os.path.join(
        save_dir,
        "metrics_FedRN_score_holdout{}_{}_{}_[{}]_{}.txt".format(
            args.epochs,
            args.dataset,
            classes_per_user,
            ",".join("{}-{}".format(a, b) for a, b in args.group_noise_rate),
            timestamp,
        ),
    )
    summary_table_filename = os.path.join(
        save_dir,
        "summary_table_FedRN_net2_score_holdout{}_{}_{}_[{}]_{}.csv".format(
            args.epochs,
            args.dataset,
            classes_per_user,
            ",".join("{}-{}".format(a, b) for a, b in args.group_noise_rate),
            timestamp,
        ),
    )

    print("Results will be saved to:", log_filename)
    print("Metrics ONLY will be saved to:", metrics_log_filename)
    print("Summary table will be saved to:", summary_table_filename)
    print(
        "Noisy train clean labels: {}/{} ({:.3f}); noisy train noisy labels: {} ({:.3f}); "
        "clean holdout labels unchanged: {}/{}".format(
            train_actual_clean,
            train_total,
            train_actual_clean / float(max(train_total, 1)),
            train_actual_noisy,
            train_actual_noisy / float(max(train_total, 1)),
            holdout_clean,
            len(all_holdout_indices),
        )
    )

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("Experiment Start: {}\n".format(timestamp))
        f.write("Args: {}\n".format(args))
        f.write("Mode: original FedRN score baseline with clean pre-noise local holdout\n")
        f.write(
            "Noisy train clean labels: {}/{} ({:.6f}); noisy train noisy labels: {} ({:.6f}); "
            "clean holdout labels unchanged: {}/{}\n".format(
                train_actual_clean,
                train_total,
                train_actual_clean / float(max(train_total, 1)),
                train_actual_noisy,
                train_actual_noisy / float(max(train_total, 1)),
                holdout_clean,
                len(all_holdout_indices),
            )
        )
        f.write("=" * 100 + "\n")

    with open(metrics_log_filename, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_acc,train_loss,clean_train_acc,clean_train_loss,test_acc,test_loss,"
            "global_holdout_acc,global_holdout_loss,"
            "selected_holdout_acc,selected_holdout_loss,local_train_loss,selected_clients,total_count,"
            "actual_round_clean_count,actual_round_noisy_count,fedrn_clean_count,fedrn_noisy_count,"
            "train_count,actual_train_clean_count,actual_train_noisy_count,missed_clean_count,"
            "false_clean_count,correctly_excluded_noisy_count,actual_round_clean_ratio,"
            "fedrn_clean_ratio,fedrn_noisy_ratio,actual_train_clean_ratio,clean_recall,train_ratio\n"
        )

    with open(summary_table_filename, "w", encoding="utf-8") as f:
        f.write(
            "epoch,net1_cifar10_acc,net1_cifar10_loss,net1_clean_train_acc,global_holdout_acc,"
            "selected_clients,score_count,score_ratio,score_precision,score_recall,"
            "missed_clean,false_clean,net2_clients,net2_avg_score_count,net2_score_precision,"
            "net2_score_recall,net2_holdout_acc,net2_train_loss,net2_result_file\n"
        )

    loader_args = dict(batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dataset_holdout_eval = make_clean_eval_dataset(dataset_train, dataset_test, original_train_labels)

    log_train_loader = DataLoader(DatasetSplit(dataset_train, all_train_indices), **loader_args)
    clean_train_loader = DataLoader(DatasetSplit(dataset_holdout_eval, all_train_indices), **loader_args)
    log_test_loader = DataLoader(dataset_test, **loader_args)

    gaussian_noise = torch.randn(1, args.num_channels, args.img_size, args.img_size).to(args.device)
    net_glob = get_model(args).to(args.device)
    cosine_similarity = torch.nn.CosineSimilarity()

    fed_args = dict(
        all_clients=args.all_clients,
        num_users=args.num_users,
        method=args.fed_method,
        dict_users=dict_users_train,
    )
    local_weights = LocalModelWeights(net_glob=net_glob, **fed_args)

    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users_train=dict_users_train,
        gaussian_noise=gaussian_noise,
        real_clean_mask=real_clean_mask,
    )
    for local in local_update_objects:
        local.weight = copy.deepcopy(net_glob.state_dict())

    personal_net2_objects = [None for _ in range(args.num_users)]
    net2_round_log_filename = os.path.join(
        save_dir,
        "continuous_net2_score_holdout{}_{}_{}_[{}]_{}.csv".format(
            args.epochs,
            args.dataset,
            classes_per_user,
            ",".join("{}-{}".format(a, b) for a, b in args.group_noise_rate),
            timestamp,
        ),
    )

    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}: {} => {}".format(epoch + 1, args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        args.g_epoch = epoch
        local_losses = []
        client_stats = []
        net2_round_rows = []

        m = max(int(args.frac * args.num_users), 1)
        selected_clients = np.random.choice(range(args.num_users), m, replace=False)

        for idx in selected_clients:
            local = local_update_objects[int(idx)]
            local.args = args

            if epoch < args.warmup_epochs:
                w, loss = local.train_phase1(copy.deepcopy(net_glob).to(args.device))
            else:
                sim_list, exp_list = [], []
                for user in range(args.num_users):
                    sim = cosine_similarity(
                        local.arbitrary_output.to(args.device),
                        local_update_objects[user].arbitrary_output.to(args.device),
                    ).item()
                    sim_list.append(sim)
                    exp_list.append(local_update_objects[user].expertise)

                sim_list = _safe_minmax(sim_list)
                exp_list = _safe_minmax(exp_list)
                prev_score = args.w_alpha * exp_list[int(idx)] + (1.0 - args.w_alpha)

                score_list = []
                for neighbor_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
                    if neighbor_idx == int(idx):
                        continue
                    score = args.w_alpha * exp + (1.0 - args.w_alpha) * sim
                    score_list.append((score, neighbor_idx))
                score_list.sort(key=lambda x: x[0], reverse=True)

                neighbor_list, neighbor_score_list = [], []
                for k in range(min(args.num_neighbors, len(score_list))):
                    neighbor_score, neighbor_idx = score_list[k]
                    neighbor_net = copy.deepcopy(local_update_objects[neighbor_idx].net1)
                    neighbor_list.append(neighbor_net)
                    neighbor_score_list.append(neighbor_score)

                w, loss = local.train_phase2(
                    copy.deepcopy(net_glob).to(args.device),
                    prev_score,
                    neighbor_list,
                    neighbor_score_list,
                )

            local_weights.update(int(idx), w)
            local_losses.append(copy.deepcopy(loss))
            client_stats.append(copy.deepcopy(local.last_stats))

            if int(getattr(args, "train_net2_after_warmup", 1)) and epoch >= int(args.warmup_epochs):
                if personal_net2_objects[int(idx)] is None:
                    personal_net2_objects[int(idx)] = init_personal_net2(args, net_glob, local)
                net2_row = update_continuous_net2_for_client(
                    local=local,
                    net2=personal_net2_objects[int(idx)],
                    dataset_train=dataset_train,
                    dataset_holdout_eval=dataset_holdout_eval,
                    holdout_indices=dict_users_holdout[int(idx)],
                    real_clean_mask=real_clean_mask,
                    args=args,
                )
                net2_round_rows.append(net2_row)

        w_glob = local_weights.average()
        net_glob.load_state_dict(w_glob, strict=False)
        local_weights.init()

        if args.client_holdout_detail == "selected" and int(args.log_client_stats):
            annotate_selected_holdout(
                client_stats,
                selected_clients,
                net_glob,
                dataset_holdout_eval,
                dict_users_holdout,
                args,
            )

        train_acc, train_loss = test_img(net_glob, log_train_loader, args)
        clean_train_acc, clean_train_loss = test_img(net_glob, clean_train_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_loader, args)

        do_holdout_eval = args.holdout_eval_every > 0 and (epoch % args.holdout_eval_every == 0)
        if do_holdout_eval:
            global_holdout_acc, global_holdout_loss = evaluate_subset(
                net_glob, dataset_holdout_eval, all_holdout_indices, args
            )
            selected_holdout_indices = _flatten_user_dict(
                {int(i): dict_users_holdout[int(i)] for i in selected_clients}
            )
            selected_holdout_acc, selected_holdout_loss = evaluate_subset(
                net_glob, dataset_holdout_eval, selected_holdout_indices, args
            )
        else:
            global_holdout_acc, global_holdout_loss = -1.0, -1.0
            selected_holdout_acc, selected_holdout_loss = -1.0, -1.0

        local_train_loss = float(np.mean(local_losses)) if local_losses else 0.0
        agg = _sum_stats(client_stats)
        net2_summary = {}
        if net2_round_rows and (
            int(getattr(args, "net2_eval_every", 1)) > 0
            and epoch % int(getattr(args, "net2_eval_every", 1)) == 0
        ):
            net2_summary = summarize_net2_round(net2_round_rows)
            append_net2_round_rows(net2_round_log_filename, epoch, net2_round_rows)

        log_round = "\n==================== Round {:3d} ====================".format(epoch)
        log_selected = "  --> selected_clients={} local_loss={:.4f} | {}".format(
            len(selected_clients),
            local_train_loss,
            _format_round_stats(agg),
        )
        log_metric = (
            "Train Acc: {:.2f}% | Train Loss: {:.4f} | Clean Train Acc: {:.2f}% | "
            "Clean Train Loss: {:.4f} | Global Acc: {:.2f}% | Global Loss: {:.4f} | "
            "Global Holdout Acc: {:.2f}% | Selected Holdout Acc: {:.2f}% | "
            "Local Train Loss: {:.4f} | Net2 Holdout Acc: {:.2f}% | {}"
        ).format(
            train_acc,
            train_loss,
            clean_train_acc,
            clean_train_loss,
            test_acc,
            test_loss,
            global_holdout_acc,
            selected_holdout_acc,
            local_train_loss,
            net2_summary.get("avg_holdout_acc", -1.0),
            _format_round_stats(agg),
        )

        print(log_round)
        print(log_selected)
        print(log_metric)

        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(log_round + "\n")
            f.write(log_selected + "\n")
            if int(args.log_client_stats):
                for stats in client_stats:
                    f.write(_client_stats_line(stats) + "\n")
            f.write(log_metric + "\n")

        with open(metrics_log_filename, "a", encoding="utf-8") as f:
            f.write(
                "{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},"
                "{:.6f},{},{},{},{},{},{},{},{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                    epoch,
                    train_acc,
                    train_loss,
                    clean_train_acc,
                    clean_train_loss,
                    test_acc,
                    test_loss,
                    global_holdout_acc,
                    global_holdout_loss,
                    selected_holdout_acc,
                    selected_holdout_loss,
                    local_train_loss,
                    len(selected_clients),
                    agg["total_count"],
                    agg["actual_round_clean_count"],
                    agg["actual_round_noisy_count"],
                    agg["fedrn_clean_count"],
                    agg["fedrn_noisy_count"],
                    agg["train_count"],
                    agg["actual_train_clean_count"],
                    agg["actual_train_noisy_count"],
                    agg["missed_clean_count"],
                    agg["false_clean_count"],
                    agg["correctly_excluded_noisy_count"],
                    agg["actual_round_clean_ratio"],
                    agg["fedrn_clean_ratio"],
                    agg["fedrn_noisy_ratio"],
                    agg["actual_train_clean_ratio"],
                    agg["clean_recall"],
                    agg["train_ratio"],
                )
            )

        net2_result_path = net2_round_log_filename if net2_summary else ""
        if int(args.save_score_snapshots) and epoch in set(int(x) for x in args.snapshot_rounds):
            snapshot_path = os.path.join(
                save_dir,
                "score_snapshot_round{}_{}.csv".format(epoch, timestamp),
            )
            export_score_snapshot(
                snapshot_path=snapshot_path,
                epoch=epoch,
                args=args,
                local_update_objects=local_update_objects,
                original_labels=original_train_labels,
                noisy_labels=noisy_train_labels,
                real_clean_mask=real_clean_mask,
                cosine_similarity=cosine_similarity,
            )
            print("Score snapshot saved to:", snapshot_path)
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write("Score snapshot saved to: {}\n".format(snapshot_path))

        if int(args.run_net2_score_eval) and epoch in set(int(x) for x in args.net2_eval_rounds):
            snapshot_net2_result_path, snapshot_net2_summary = run_net2_score_eval(
                epoch=epoch,
                args=args,
                save_dir=save_dir,
                timestamp=timestamp,
                net_glob=net_glob,
                dataset_train=dataset_train,
                dataset_holdout_eval=dataset_holdout_eval,
                dict_users_holdout=dict_users_holdout,
                local_update_objects=local_update_objects,
                real_clean_mask=real_clean_mask,
                cosine_similarity=cosine_similarity,
            )
            if snapshot_net2_summary:
                net2_line = (
                    "Net2 score eval saved to: {} | clients={} avg_score_count={:.2f} "
                    "avg_score_precision={:.4f} avg_score_recall={:.4f} "
                    "avg_holdout_acc={:.2f}% avg_train_loss={:.4f}"
                ).format(
                    snapshot_net2_result_path,
                    snapshot_net2_summary["clients"],
                    snapshot_net2_summary["avg_score_count"],
                    snapshot_net2_summary["avg_score_precision"],
                    snapshot_net2_summary["avg_score_recall"],
                    snapshot_net2_summary["avg_holdout_acc"],
                    snapshot_net2_summary["avg_train_loss"],
                )
            else:
                net2_line = "Net2 score eval saved to: {} | no clients evaluated".format(snapshot_net2_result_path)
            print(net2_line)
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(net2_line + "\n")

        with open(summary_table_filename, "a", encoding="utf-8") as f:
            f.write(
                "{},{:.6f},{:.6f},{:.6f},{:.6f},{},{},{:.6f},{:.6f},{:.6f},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}\n".format(
                    epoch,
                    test_acc,
                    test_loss,
                    clean_train_acc,
                    global_holdout_acc,
                    len(selected_clients),
                    agg["fedrn_clean_count"],
                    agg["fedrn_clean_ratio"],
                    agg["actual_train_clean_ratio"],
                    agg["clean_recall"],
                    agg["missed_clean_count"],
                    agg["false_clean_count"],
                    net2_summary.get("clients", 0),
                    net2_summary.get("avg_score_count", -1.0),
                    net2_summary.get("avg_score_precision", -1.0),
                    net2_summary.get("avg_score_recall", -1.0),
                    net2_summary.get("avg_holdout_acc", -1.0),
                    net2_summary.get("avg_train_loss", -1.0),
                    net2_result_path,
                )
            )

    print("Total time: {:.1f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
