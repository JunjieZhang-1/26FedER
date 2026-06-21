#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Original FedRN with detailed clean/noisy logging.

Purpose:
- Keep the original FedRN training logic unchanged as much as possible.
- Add detailed statistics for every selected round/client:
  * how many samples participate in local training / total local samples
  * how many local samples are actually clean / noisy after synthetic noise injection
  * how many samples FedRN selects as clean / noisy
  * among FedRN-selected clean samples, how many are actually clean / actually noisy
  * how many actually-clean samples are mistakenly excluded as noisy
  * precision / recall of FedRN clean selection

Usage example:
python main_fedrn_original_detailed_logging.py --method fedrn --dataset cifar10 --epochs 500 \
  --warmup_epochs 100 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 \
  --num_shards 200 --group_noise_rate 0 0.4 --noise_group_num 100 \
  --num_neighbors 2 --w_alpha 0.6 --p_threshold 0.5
"""

import os
import sys

# =============================================================================
# 🌟 Windows 环境兼容性修复 (DLL Load Failed Fix)
# =============================================================================
conda_path = r"C:\Users\25839\.conda\envs\improve-FedRN-main\Library\bin"
if os.path.exists(conda_path):
    os.environ['PATH'] = conda_path + os.pathsep + os.environ['PATH']
else:
    pass
#
# import os
# import sys
#
# # -----------------------------------------------------------------------------
# # Windows + Conda DLL path fix
# # 必须放在 import numpy / import torch 之前。
# # 否则在某些 PyCharm/Windows 环境中，NumPy 的 MKL DLL 会找不到。
# # -----------------------------------------------------------------------------
# if sys.platform.startswith("win"):
#     conda_prefix = os.environ.get("CONDA_PREFIX")
#     if not conda_prefix:
#         # 例如：C:\Users\25839\.conda\envs\improve-FedRN-main\python.exe
#         # 推回环境根目录：C:\Users\25839\.conda\envs\improve-FedRN-main
#         conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
#
#     dll_dirs = [
#         os.path.join(conda_prefix, "Library", "bin"),
#         os.path.join(conda_prefix, "DLLs"),
#         os.path.join(conda_prefix, "libs"),
#     ]
#     for dll_dir in dll_dirs:
#         if os.path.isdir(dll_dir) and dll_dir not in os.environ.get("PATH", ""):
#             os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

import copy
import datetime
import random
import time

import numpy as np
import torch
import torchvision
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from utils import load_dataset
from utils.options_original_fedrn import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label

from models.fed import LocalModelWeights
from models.nets_original_fedrn import get_model
from models.test import test_img
from models.update_original_fedrn import LocalUpdateFedRN, DatasetSplit


def _get_train_labels(dataset):
    """Return train labels as a mutable list-like object when possible."""
    if hasattr(dataset, "train_labels"):
        return dataset.train_labels
    if hasattr(dataset, "targets"):
        return dataset.targets
    raise AttributeError("dataset has neither train_labels nor targets")


def _set_train_label(dataset, idx, value):
    """Set a label while keeping torchvision train_labels / targets aliases consistent."""
    if hasattr(dataset, "train_labels"):
        dataset.train_labels[idx] = int(value)
    if hasattr(dataset, "targets"):
        dataset.targets[idx] = int(value)


def _ensure_cifar_label_alias(dataset):
    """Compatibility for different torchvision versions."""
    if hasattr(dataset, "targets") and not hasattr(dataset, "train_labels"):
        dataset.train_labels = dataset.targets
    if hasattr(dataset, "train_labels") and not hasattr(dataset, "targets"):
        dataset.targets = dataset.train_labels


def _safe_minmax(values):
    values = list(values)
    v_min, v_max = min(values), max(values)
    denom = v_max - v_min
    if abs(denom) < 1e-12:
        return [0.0 for _ in values]
    return [(v - v_min) / denom for v in values]


class LocalUpdateFedRNDetailed(LocalUpdateFedRN):
    """Original LocalUpdateFedRN plus detailed clean/noisy statistics."""

    def __init__(self, *args, real_clean_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_clean_mask = real_clean_mask
        self.last_stats = self._empty_stats(stage="init")

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
        )

    def _compute_selection_stats(self, train_indices, noisy_indices=None, stage="phase2"):
        """
        train_indices: samples used for local training, i.e. FedRN-selected clean set.
        noisy_indices: samples excluded by FedRN, i.e. FedRN-predicted noisy set.
        """
        data_indices = np.asarray(self.data_indices, dtype=np.int64)
        train_indices = np.asarray(train_indices, dtype=np.int64)

        if noisy_indices is None:
            train_set = set(train_indices.tolist())
            noisy_indices = np.asarray([idx for idx in data_indices if idx not in train_set], dtype=np.int64)
        else:
            noisy_indices = np.asarray(noisy_indices, dtype=np.int64)

        total_count = int(len(data_indices))
        fedrn_clean_count = int(len(train_indices))
        fedrn_noisy_count = int(len(noisy_indices))

        if self.real_clean_mask is None:
            # If true labels are unavailable, still log FedRN selection size.
            return dict(
                user_idx=int(self.user_idx),
                stage=stage,
                total_count=total_count,
                actual_round_clean_count=0,
                actual_round_noisy_count=0,
                fedrn_clean_count=fedrn_clean_count,
                fedrn_noisy_count=fedrn_noisy_count,
                train_count=fedrn_clean_count,
                actual_train_clean_count=0,
                actual_train_noisy_count=0,
                false_clean_count=0,
                missed_clean_count=0,
                correctly_excluded_noisy_count=0,
                actual_train_clean_ratio=0.0,
                clean_recall=0.0,
                train_ratio=float(fedrn_clean_count) / max(total_count, 1),
                fedrn_clean_ratio=float(fedrn_clean_count) / max(total_count, 1),
                fedrn_noisy_ratio=float(fedrn_noisy_count) / max(total_count, 1),
            )

        actual_round_clean_count = int(np.asarray(self.real_clean_mask)[data_indices].sum())
        actual_round_noisy_count = int(total_count - actual_round_clean_count)

        actual_train_clean_count = int(np.asarray(self.real_clean_mask)[train_indices].sum()) if fedrn_clean_count > 0 else 0
        actual_train_noisy_count = int(fedrn_clean_count - actual_train_clean_count)

        missed_clean_count = int(np.asarray(self.real_clean_mask)[noisy_indices].sum()) if fedrn_noisy_count > 0 else 0
        correctly_excluded_noisy_count = int(fedrn_noisy_count - missed_clean_count)

        # false_clean_count = FedRN thinks clean but actually noisy.
        false_clean_count = actual_train_noisy_count

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
            false_clean_count=false_clean_count,
            missed_clean_count=missed_clean_count,
            correctly_excluded_noisy_count=correctly_excluded_noisy_count,
            actual_train_clean_ratio=float(actual_train_clean_count) / max(fedrn_clean_count, 1),
            clean_recall=float(actual_train_clean_count) / max(actual_round_clean_count, 1),
            train_ratio=float(fedrn_clean_count) / max(total_count, 1),
            fedrn_clean_ratio=float(fedrn_clean_count) / max(total_count, 1),
            fedrn_noisy_ratio=float(fedrn_noisy_count) / max(total_count, 1),
        )

    def fit_gmm(self, net):
        """Same as original FedRN, with a numerical guard for constant loss."""
        losses = []
        net.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = net(inputs)
                loss = self.CE(outputs, targets)
                losses.append(loss)

        losses = torch.cat(losses).cpu().numpy()
        denom = losses.max() - losses.min()
        if abs(float(denom)) < 1e-12:
            losses = np.zeros_like(losses, dtype=np.float64)
        else:
            losses = (losses - losses.min()) / denom

        input_loss = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return prob

    def train_phase1(self, net):
        # Warmup: all local samples participate in training. FedRN has not started filtering.
        w, loss = self.train_single_model(net)
        self.set_expertise()
        self.set_arbitrary_output()
        self.last_stats = self._compute_selection_stats(
            train_indices=self.data_indices,
            noisy_indices=np.array([], dtype=np.int64),
            stage="warmup_all_data",
        )
        return w, loss

    def train_phase2(self, net, prev_score, neighbor_list, neighbor_score_list):
        # 1. Target model GMM preliminary clean set.
        prob = self.fit_gmm(self.net1)
        pred_clean_idx, pred_noisy_idx = self.get_clean_idx(prob)

        # 2. Neighbor head fine-tuning and neighbor GMMs.
        prob_list = [prob]
        neighbor_list = self.finetune_head(neighbor_list, pred_clean_idx)
        for neighbor_net in neighbor_list:
            neighbor_prob = self.fit_gmm(neighbor_net)
            prob_list.append(neighbor_prob)

        # 3. Reliability-score weighted probability fusion.
        score_list = [float(prev_score)] + [float(s) for s in neighbor_score_list]
        score_sum = float(sum(score_list)) + 1e-12
        score_list = [score / score_sum for score in score_list]

        final_prob = np.zeros(len(prob), dtype=np.float64)
        for p, score in zip(prob_list, score_list):
            final_prob = np.add(final_prob, np.multiply(p, score))

        # 4. FedRN final clean/noisy decision.
        final_clean_idx, final_noisy_idx = self.get_clean_idx(final_prob)

        # 5. Detailed logging stats before training.
        self.last_stats = self._compute_selection_stats(
            train_indices=final_clean_idx,
            noisy_indices=final_noisy_idx,
            stage="fedrn_selection",
        )
        self.last_stats["initial_fedrn_clean_count"] = int(len(pred_clean_idx))
        self.last_stats["initial_fedrn_noisy_count"] = int(len(pred_noisy_idx))

        # 6. Original FedRN local training on final clean set.
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


def get_local_update_objects_detailed(args, dataset_train, dict_users, noise_rates, gaussian_noise, real_clean_mask):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        if args.method != "fedrn":
            raise ValueError("This detailed logger is intended for original FedRN only. Please use --method fedrn.")
        local_update_object = LocalUpdateFedRNDetailed(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
            gaussian_noise=gaussian_noise,
            real_clean_mask=real_clean_mask,
        )
        local_update_objects.append(local_update_object)
    return local_update_objects


def _sum_stats(stats_list):
    """Aggregate count-based stats across selected clients for one round."""
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
    ]
    agg = {k: int(sum(int(s.get(k, 0)) for s in stats_list)) for k in keys}
    agg["train_ratio"] = agg["train_count"] / max(agg["total_count"], 1)
    agg["fedrn_clean_ratio"] = agg["fedrn_clean_count"] / max(agg["total_count"], 1)
    agg["fedrn_noisy_ratio"] = agg["fedrn_noisy_count"] / max(agg["total_count"], 1)
    agg["actual_train_clean_ratio"] = agg["actual_train_clean_count"] / max(agg["train_count"], 1)
    agg["clean_recall"] = agg["actual_train_clean_count"] / max(agg["actual_round_clean_count"], 1)
    agg["actual_round_clean_ratio"] = agg["actual_round_clean_count"] / max(agg["total_count"], 1)
    agg["actual_round_noisy_ratio"] = agg["actual_round_noisy_count"] / max(agg["total_count"], 1)
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


def _client_stats_line(s):
    return (
        "    [Client {user_idx:03d}] stage={stage} "
        "Train={train_count}/{total_count} FedRNClean={fedrn_clean_count} FedRNNoisy={fedrn_noisy_count} "
        "ActualRoundClean={actual_round_clean_count} ActualRoundNoisy={actual_round_noisy_count} "
        "ActualTrainClean={actual_train_clean_ratio:.3f}({actual_train_clean_count}/{train_count}) "
        "CleanRecall={clean_recall:.3f} MissedClean={missed_clean_count} FalseClean={false_clean_count}"
    ).format(**s)


def main():
    start = time.time()
    args = args_parser()

    # Force the original FedRN method for this diagnostic script.
    args.method = "fedrn"
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    args.schedule = [int(x) for x in args.schedule]
    args.send_2_models = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("resultdate", "FedRN_original_detailed")
    os.makedirs(save_dir, exist_ok=True)

    print("FedRN original detailed logging run")
    for x in vars(args).items():
        print(x)

    if not torch.cuda.is_available() and args.gpu != -1:
        raise RuntimeError("Cuda is not available. Use --gpu -1 for CPU debugging.")
    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Arbitrary gaussian noise for similarity indicator.
    gaussian_noise = torch.randn(1, 3, 32, 32)

    # -------------------------------------------------------------------------
    # Load dataset and split users.
    # -------------------------------------------------------------------------
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    _ensure_cifar_label_alias(dataset_train)
    _ensure_cifar_label_alias(dataset_test)

    original_train_labels = np.array(copy.deepcopy(list(_get_train_labels(dataset_train))), dtype=np.int64)
    labels = np.array(_get_train_labels(dataset_train), dtype=np.int64)
    img_size = dataset_train[0][0].shape
    args.img_size = int(img_size[1])

    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == "shard":
        dict_users = sample_noniid_shard(labels=labels, num_users=args.num_users, num_shards=args.num_shards)
    elif args.partition == "dirichlet":
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)
    else:
        raise ValueError("Unsupported partition: {}".format(args.partition))

    # -------------------------------------------------------------------------
    # Add label noise to training data.
    # -------------------------------------------------------------------------
    if sum(args.noise_group_num) != args.num_users:
        raise ValueError("sum(args.noise_group_num) must equal args.num_users")

    if len(args.group_noise_rate) == 1:
        args.group_noise_rate = args.group_noise_rate * 2

    if not len(args.noise_group_num) == len(args.group_noise_rate) and \
            len(args.group_noise_rate) * 2 == len(args.noise_type_lst):
        raise ValueError("The noise input is invalid")

    args.group_noise_rate = [
        (args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
        for i in range(len(args.group_noise_rate) // 2)
    ]

    user_noise_type_rates = []
    for num_users_in_group, noise_type, (min_group_noise_rate, max_group_noise_rate) in zip(
        args.noise_group_num, args.noise_type_lst, args.group_noise_rate
    ):
        step = (max_group_noise_rate - min_group_noise_rate) / max(num_users_in_group, 1)
        noise_rates = np.array(range(num_users_in_group)) * step + min_group_noise_rate
        user_noise_type_rates += [*zip([noise_type] * num_users_in_group, noise_rates)]

    for user, (user_noise_type, user_noise_rate) in enumerate(user_noise_type_rates):
        if user_noise_type != "clean":
            data_indices = list(copy.deepcopy(dict_users[user]))
            random.seed(args.seed)  # keep original FedRN behavior
            random.shuffle(data_indices)
            noise_index = int(len(data_indices) * user_noise_rate)

            for d_idx in data_indices[:noise_index]:
                true_label = int(_get_train_labels(dataset_train)[d_idx])
                noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
                _set_train_label(dataset_train, d_idx, noisy_label)

    noisy_train_labels = np.array(copy.deepcopy(list(_get_train_labels(dataset_train))), dtype=np.int64)
    real_clean_mask = (original_train_labels == noisy_train_labels)
    global_actual_clean = int(real_clean_mask.sum())
    global_total = int(len(real_clean_mask))
    global_actual_noisy = global_total - global_actual_clean

    classes_per_user = args.num_shards // args.num_users if args.partition == "shard" else "dirichlet"
    log_filename = os.path.join(
        save_dir,
        f"详细模型_FedRN_original_detailed{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
    )
    metrics_log_filename = os.path.join(
        save_dir,
        f"总体模型_FedRN_original_detailed{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
    )

    print("Results will be saved to:", log_filename)
    print("Metrics ONLY will be saved to:", metrics_log_filename)
    print(
        "Actual clean labels in noisy train set: {}/{} ({:.3f}); Actual noisy labels: {} ({:.3f})".format(
            global_actual_clean,
            global_total,
            global_actual_clean / max(global_total, 1),
            global_actual_noisy,
            global_actual_noisy / max(global_total, 1),
        )
    )

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Experiment Start: {timestamp}\n")
        f.write(f"Args: {args}\n")
        f.write(
            "Actual clean labels in noisy train set: {}/{} ({:.6f}); Actual noisy labels: {} ({:.6f})\n".format(
                global_actual_clean,
                global_total,
                global_actual_clean / max(global_total, 1),
                global_actual_noisy,
                global_actual_noisy / max(global_total, 1),
            )
        )
        f.write("=" * 100 + "\n")

    with open(metrics_log_filename, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_acc,train_loss,test_acc,test_loss,local_train_loss,"
            "selected_clients,total_count,actual_round_clean_count,actual_round_noisy_count,"
            "fedrn_clean_count,fedrn_noisy_count,train_count,actual_train_clean_count,actual_train_noisy_count,"
            "missed_clean_count,false_clean_count,correctly_excluded_noisy_count,"
            "actual_round_clean_ratio,fedrn_clean_ratio,fedrn_noisy_ratio,actual_train_clean_ratio,clean_recall,train_ratio\n"
        )

    # -------------------------------------------------------------------------
    # Dataloaders for global logging.
    # -------------------------------------------------------------------------
    logging_args = dict(batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, **logging_args)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, **logging_args)

    # -------------------------------------------------------------------------
    # Build model and local objects.
    # -------------------------------------------------------------------------
    net_glob = get_model(args).to(args.device)
    CosineSimilarity = torch.nn.CosineSimilarity()

    pred_user_noise_rates = [args.forget_rate] * args.num_users
    fed_args = dict(
        all_clients=args.all_clients,
        num_users=args.num_users,
        method=args.fed_method,
        dict_users=dict_users,
    )
    local_weights = LocalModelWeights(net_glob=net_glob, **fed_args)

    local_update_objects = get_local_update_objects_detailed(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        noise_rates=pred_user_noise_rates,
        gaussian_noise=gaussian_noise,
        real_clean_mask=real_clean_mask,
    )
    for i in range(args.num_users):
        local = local_update_objects[i]
        local.weight = copy.deepcopy(net_glob.state_dict())

    # -------------------------------------------------------------------------
    # Federated training loop.
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        local_losses = []
        client_stats = []
        args.g_epoch = epoch

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Update.
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args

            if epoch < args.warmup_epochs:
                w, loss = local.train_phase1(copy.deepcopy(net_glob).to(args.device))
            else:
                # Get similarity and expertise values.
                sim_list = []
                exp_list = []
                for user in range(args.num_users):
                    sim = CosineSimilarity(
                        local.arbitrary_output.to(args.device),
                        local_update_objects[user].arbitrary_output.to(args.device),
                    ).item()
                    exp = local_update_objects[user].expertise
                    sim_list.append(sim)
                    exp_list.append(exp)

                # Normalize similarity & expertise values with numerical guards.
                sim_list = _safe_minmax(sim_list)
                exp_list = _safe_minmax(exp_list)

                # Compute and sort reliability scores.
                prev_score = args.w_alpha * exp_list[idx] + (1 - args.w_alpha)

                score_list = []
                for neighbor_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
                    if neighbor_idx != idx:
                        score = args.w_alpha * exp + (1 - args.w_alpha) * sim
                        score_list.append([score, neighbor_idx])
                score_list.sort(key=lambda x: x[0], reverse=True)

                # Get top-k reliable neighbors.
                neighbor_list = []
                neighbor_score_list = []
                for k in range(args.num_neighbors):
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

            local_weights.update(idx, w)
            local_losses.append(copy.deepcopy(loss))
            client_stats.append(copy.deepcopy(local.last_stats))

        # Global aggregation.
        w_glob = local_weights.average()
        net_glob.load_state_dict(w_glob, strict=False)
        local_weights.init()

        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
        local_train_loss = float(np.mean(local_losses)) if local_losses else 0.0
        agg = _sum_stats(client_stats)

        log_round = "\n==================== Round {:3d} ====================".format(epoch)
        log_selected = "  --> selected_clients={} local_loss={:.4f} | {}".format(
            len(idxs_users),
            local_train_loss,
            _format_round_stats(agg),
        )
        log_metric = (
            "Train Acc: {:.2f}% | Train Loss: {:.4f} | Global Acc: {:.2f}% | Global Loss: {:.4f} | "
            "Local Train Loss: {:.4f} | {}"
        ).format(train_acc, train_loss, test_acc, test_loss, local_train_loss, _format_round_stats(agg))

        print(log_round)
        print(log_selected)
        print(log_metric)

        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(log_round + "\n")
            f.write(log_selected + "\n")
            for s in client_stats:
                f.write(_client_stats_line(s) + "\n")
            f.write(log_metric + "\n")

        with open(metrics_log_filename, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_acc:.6f},{train_loss:.6f},{test_acc:.6f},{test_loss:.6f},{local_train_loss:.6f},"
                f"{len(idxs_users)},{agg['total_count']},{agg['actual_round_clean_count']},{agg['actual_round_noisy_count']},"
                f"{agg['fedrn_clean_count']},{agg['fedrn_noisy_count']},{agg['train_count']},"
                f"{agg['actual_train_clean_count']},{agg['actual_train_noisy_count']},"
                f"{agg['missed_clean_count']},{agg['false_clean_count']},{agg['correctly_excluded_noisy_count']},"
                f"{agg['actual_round_clean_ratio']:.6f},{agg['fedrn_clean_ratio']:.6f},{agg['fedrn_noisy_ratio']:.6f},"
                f"{agg['actual_train_clean_ratio']:.6f},{agg['clean_recall']:.6f},{agg['train_ratio']:.6f}\n"
            )

    print("Total time: {:.1f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Original FedRN with detailed clean/noisy logging.

Purpose:
- Keep the original FedRN training logic unchanged as much as possible.
- Add detailed statistics for every selected round/client:
  * how many samples participate in local training / total local samples
  * how many local samples are actually clean / noisy after synthetic noise injection
  * how many samples FedRN selects as clean / noisy
  * among FedRN-selected clean samples, how many are actually clean / actually noisy
  * how many actually-clean samples are mistakenly excluded as noisy
  * precision / recall of FedRN clean selection

Usage example:
python main_fedrn_original_detailed_logging.py --method fedrn --dataset cifar10 --epochs 500 \
  --warmup_epochs 100 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 \
  --num_shards 200 --group_noise_rate 0 0.4 --noise_group_num 100 \
  --num_neighbors 2 --w_alpha 0.6 --p_threshold 0.5
"""

import os
import sys

# =============================================================================
# 🌟 Windows 环境兼容性修复 (DLL Load Failed Fix)
# =============================================================================
conda_path = r"C:\Users\25839\.conda\envs\improve-FedRN-main\Library\bin"
if os.path.exists(conda_path):
    os.environ['PATH'] = conda_path + os.pathsep + os.environ['PATH']
else:
    pass
#
# import os
# import sys
#
# # -----------------------------------------------------------------------------
# # Windows + Conda DLL path fix
# # 必须放在 import numpy / import torch 之前。
# # 否则在某些 PyCharm/Windows 环境中，NumPy 的 MKL DLL 会找不到。
# # -----------------------------------------------------------------------------
# if sys.platform.startswith("win"):
#     conda_prefix = os.environ.get("CONDA_PREFIX")
#     if not conda_prefix:
#         # 例如：C:\Users\25839\.conda\envs\improve-FedRN-main\python.exe
#         # 推回环境根目录：C:\Users\25839\.conda\envs\improve-FedRN-main
#         conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
#
#     dll_dirs = [
#         os.path.join(conda_prefix, "Library", "bin"),
#         os.path.join(conda_prefix, "DLLs"),
#         os.path.join(conda_prefix, "libs"),
#     ]
#     for dll_dir in dll_dirs:
#         if os.path.isdir(dll_dir) and dll_dir not in os.environ.get("PATH", ""):
#             os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

import copy
import datetime
import random
import time

import numpy as np
import torch
import torchvision
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from utils import load_dataset
from utils.options_original_fedrn import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label

from models.fed import LocalModelWeights
from models.nets_original_fedrn import get_model
from models.test import test_img
from models.update_original_fedrn import LocalUpdateFedRN, DatasetSplit


def _get_train_labels(dataset):
    """Return train labels as a mutable list-like object when possible."""
    if hasattr(dataset, "train_labels"):
        return dataset.train_labels
    if hasattr(dataset, "targets"):
        return dataset.targets
    raise AttributeError("dataset has neither train_labels nor targets")


def _set_train_label(dataset, idx, value):
    """Set a label while keeping torchvision train_labels / targets aliases consistent."""
    if hasattr(dataset, "train_labels"):
        dataset.train_labels[idx] = int(value)
    if hasattr(dataset, "targets"):
        dataset.targets[idx] = int(value)


def _ensure_cifar_label_alias(dataset):
    """Compatibility for different torchvision versions."""
    if hasattr(dataset, "targets") and not hasattr(dataset, "train_labels"):
        dataset.train_labels = dataset.targets
    if hasattr(dataset, "train_labels") and not hasattr(dataset, "targets"):
        dataset.targets = dataset.train_labels


def _safe_minmax(values):
    values = list(values)
    v_min, v_max = min(values), max(values)
    denom = v_max - v_min
    if abs(denom) < 1e-12:
        return [0.0 for _ in values]
    return [(v - v_min) / denom for v in values]


class LocalUpdateFedRNDetailed(LocalUpdateFedRN):
    """Original LocalUpdateFedRN plus detailed clean/noisy statistics."""

    def __init__(self, *args, real_clean_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_clean_mask = real_clean_mask
        self.last_stats = self._empty_stats(stage="init")

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
        )

    def _compute_selection_stats(self, train_indices, noisy_indices=None, stage="phase2"):
        """
        train_indices: samples used for local training, i.e. FedRN-selected clean set.
        noisy_indices: samples excluded by FedRN, i.e. FedRN-predicted noisy set.
        """
        data_indices = np.asarray(self.data_indices, dtype=np.int64)
        train_indices = np.asarray(train_indices, dtype=np.int64)

        if noisy_indices is None:
            train_set = set(train_indices.tolist())
            noisy_indices = np.asarray([idx for idx in data_indices if idx not in train_set], dtype=np.int64)
        else:
            noisy_indices = np.asarray(noisy_indices, dtype=np.int64)

        total_count = int(len(data_indices))
        fedrn_clean_count = int(len(train_indices))
        fedrn_noisy_count = int(len(noisy_indices))

        if self.real_clean_mask is None:
            # If true labels are unavailable, still log FedRN selection size.
            return dict(
                user_idx=int(self.user_idx),
                stage=stage,
                total_count=total_count,
                actual_round_clean_count=0,
                actual_round_noisy_count=0,
                fedrn_clean_count=fedrn_clean_count,
                fedrn_noisy_count=fedrn_noisy_count,
                train_count=fedrn_clean_count,
                actual_train_clean_count=0,
                actual_train_noisy_count=0,
                false_clean_count=0,
                missed_clean_count=0,
                correctly_excluded_noisy_count=0,
                actual_train_clean_ratio=0.0,
                clean_recall=0.0,
                train_ratio=float(fedrn_clean_count) / max(total_count, 1),
                fedrn_clean_ratio=float(fedrn_clean_count) / max(total_count, 1),
                fedrn_noisy_ratio=float(fedrn_noisy_count) / max(total_count, 1),
            )

        actual_round_clean_count = int(np.asarray(self.real_clean_mask)[data_indices].sum())
        actual_round_noisy_count = int(total_count - actual_round_clean_count)

        actual_train_clean_count = int(np.asarray(self.real_clean_mask)[train_indices].sum()) if fedrn_clean_count > 0 else 0
        actual_train_noisy_count = int(fedrn_clean_count - actual_train_clean_count)

        missed_clean_count = int(np.asarray(self.real_clean_mask)[noisy_indices].sum()) if fedrn_noisy_count > 0 else 0
        correctly_excluded_noisy_count = int(fedrn_noisy_count - missed_clean_count)

        # false_clean_count = FedRN thinks clean but actually noisy.
        false_clean_count = actual_train_noisy_count

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
            false_clean_count=false_clean_count,
            missed_clean_count=missed_clean_count,
            correctly_excluded_noisy_count=correctly_excluded_noisy_count,
            actual_train_clean_ratio=float(actual_train_clean_count) / max(fedrn_clean_count, 1),
            clean_recall=float(actual_train_clean_count) / max(actual_round_clean_count, 1),
            train_ratio=float(fedrn_clean_count) / max(total_count, 1),
            fedrn_clean_ratio=float(fedrn_clean_count) / max(total_count, 1),
            fedrn_noisy_ratio=float(fedrn_noisy_count) / max(total_count, 1),
        )

    def fit_gmm(self, net):
        """Same as original FedRN, with a numerical guard for constant loss."""
        losses = []
        net.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = net(inputs)
                loss = self.CE(outputs, targets)
                losses.append(loss)

        losses = torch.cat(losses).cpu().numpy()
        denom = losses.max() - losses.min()
        if abs(float(denom)) < 1e-12:
            losses = np.zeros_like(losses, dtype=np.float64)
        else:
            losses = (losses - losses.min()) / denom

        input_loss = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return prob

    def train_phase1(self, net):
        # Warmup: all local samples participate in training. FedRN has not started filtering.
        w, loss = self.train_single_model(net)
        self.set_expertise()
        self.set_arbitrary_output()
        self.last_stats = self._compute_selection_stats(
            train_indices=self.data_indices,
            noisy_indices=np.array([], dtype=np.int64),
            stage="warmup_all_data",
        )
        return w, loss

    def train_phase2(self, net, prev_score, neighbor_list, neighbor_score_list):
        # 1. Target model GMM preliminary clean set.
        prob = self.fit_gmm(self.net1)
        pred_clean_idx, pred_noisy_idx = self.get_clean_idx(prob)

        # 2. Neighbor head fine-tuning and neighbor GMMs.
        prob_list = [prob]
        neighbor_list = self.finetune_head(neighbor_list, pred_clean_idx)
        for neighbor_net in neighbor_list:
            neighbor_prob = self.fit_gmm(neighbor_net)
            prob_list.append(neighbor_prob)

        # 3. Reliability-score weighted probability fusion.
        score_list = [float(prev_score)] + [float(s) for s in neighbor_score_list]
        score_sum = float(sum(score_list)) + 1e-12
        score_list = [score / score_sum for score in score_list]

        final_prob = np.zeros(len(prob), dtype=np.float64)
        for p, score in zip(prob_list, score_list):
            final_prob = np.add(final_prob, np.multiply(p, score))

        # 4. FedRN final clean/noisy decision.
        final_clean_idx, final_noisy_idx = self.get_clean_idx(final_prob)

        # 5. Detailed logging stats before training.
        self.last_stats = self._compute_selection_stats(
            train_indices=final_clean_idx,
            noisy_indices=final_noisy_idx,
            stage="fedrn_selection",
        )
        self.last_stats["initial_fedrn_clean_count"] = int(len(pred_clean_idx))
        self.last_stats["initial_fedrn_noisy_count"] = int(len(pred_noisy_idx))

        # 6. Original FedRN local training on final clean set.
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


def get_local_update_objects_detailed(args, dataset_train, dict_users, noise_rates, gaussian_noise, real_clean_mask):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        if args.method != "fedrn":
            raise ValueError("This detailed logger is intended for original FedRN only. Please use --method fedrn.")
        local_update_object = LocalUpdateFedRNDetailed(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
            gaussian_noise=gaussian_noise,
            real_clean_mask=real_clean_mask,
        )
        local_update_objects.append(local_update_object)
    return local_update_objects


def _sum_stats(stats_list):
    """Aggregate count-based stats across selected clients for one round."""
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
    ]
    agg = {k: int(sum(int(s.get(k, 0)) for s in stats_list)) for k in keys}
    agg["train_ratio"] = agg["train_count"] / max(agg["total_count"], 1)
    agg["fedrn_clean_ratio"] = agg["fedrn_clean_count"] / max(agg["total_count"], 1)
    agg["fedrn_noisy_ratio"] = agg["fedrn_noisy_count"] / max(agg["total_count"], 1)
    agg["actual_train_clean_ratio"] = agg["actual_train_clean_count"] / max(agg["train_count"], 1)
    agg["clean_recall"] = agg["actual_train_clean_count"] / max(agg["actual_round_clean_count"], 1)
    agg["actual_round_clean_ratio"] = agg["actual_round_clean_count"] / max(agg["total_count"], 1)
    agg["actual_round_noisy_ratio"] = agg["actual_round_noisy_count"] / max(agg["total_count"], 1)
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


def _client_stats_line(s):
    return (
        "    [Client {user_idx:03d}] stage={stage} "
        "Train={train_count}/{total_count} FedRNClean={fedrn_clean_count} FedRNNoisy={fedrn_noisy_count} "
        "ActualRoundClean={actual_round_clean_count} ActualRoundNoisy={actual_round_noisy_count} "
        "ActualTrainClean={actual_train_clean_ratio:.3f}({actual_train_clean_count}/{train_count}) "
        "CleanRecall={clean_recall:.3f} MissedClean={missed_clean_count} FalseClean={false_clean_count}"
    ).format(**s)


def main():
    start = time.time()
    args = args_parser()

    # Force the original FedRN method for this diagnostic script.
    args.method = "fedrn"
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    args.schedule = [int(x) for x in args.schedule]
    args.send_2_models = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("resultdate", "FedRN_original_detailed")
    os.makedirs(save_dir, exist_ok=True)

    print("FedRN original detailed logging run")
    for x in vars(args).items():
        print(x)

    if not torch.cuda.is_available() and args.gpu != -1:
        raise RuntimeError("Cuda is not available. Use --gpu -1 for CPU debugging.")
    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Arbitrary gaussian noise for similarity indicator.
    gaussian_noise = torch.randn(1, 3, 32, 32)

    # -------------------------------------------------------------------------
    # Load dataset and split users.
    # -------------------------------------------------------------------------
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    _ensure_cifar_label_alias(dataset_train)
    _ensure_cifar_label_alias(dataset_test)

    original_train_labels = np.array(copy.deepcopy(list(_get_train_labels(dataset_train))), dtype=np.int64)
    labels = np.array(_get_train_labels(dataset_train), dtype=np.int64)
    img_size = dataset_train[0][0].shape
    args.img_size = int(img_size[1])

    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == "shard":
        dict_users = sample_noniid_shard(labels=labels, num_users=args.num_users, num_shards=args.num_shards)
    elif args.partition == "dirichlet":
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)
    else:
        raise ValueError("Unsupported partition: {}".format(args.partition))

    # -------------------------------------------------------------------------
    # Add label noise to training data.
    # -------------------------------------------------------------------------
    if sum(args.noise_group_num) != args.num_users:
        raise ValueError("sum(args.noise_group_num) must equal args.num_users")

    if len(args.group_noise_rate) == 1:
        args.group_noise_rate = args.group_noise_rate * 2

    if not len(args.noise_group_num) == len(args.group_noise_rate) and \
            len(args.group_noise_rate) * 2 == len(args.noise_type_lst):
        raise ValueError("The noise input is invalid")

    args.group_noise_rate = [
        (args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
        for i in range(len(args.group_noise_rate) // 2)
    ]

    user_noise_type_rates = []
    for num_users_in_group, noise_type, (min_group_noise_rate, max_group_noise_rate) in zip(
        args.noise_group_num, args.noise_type_lst, args.group_noise_rate
    ):
        step = (max_group_noise_rate - min_group_noise_rate) / max(num_users_in_group, 1)
        noise_rates = np.array(range(num_users_in_group)) * step + min_group_noise_rate
        user_noise_type_rates += [*zip([noise_type] * num_users_in_group, noise_rates)]

    for user, (user_noise_type, user_noise_rate) in enumerate(user_noise_type_rates):
        if user_noise_type != "clean":
            data_indices = list(copy.deepcopy(dict_users[user]))
            random.seed(args.seed)  # keep original FedRN behavior
            random.shuffle(data_indices)
            noise_index = int(len(data_indices) * user_noise_rate)

            for d_idx in data_indices[:noise_index]:
                true_label = int(_get_train_labels(dataset_train)[d_idx])
                noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
                _set_train_label(dataset_train, d_idx, noisy_label)

    noisy_train_labels = np.array(copy.deepcopy(list(_get_train_labels(dataset_train))), dtype=np.int64)
    real_clean_mask = (original_train_labels == noisy_train_labels)
    global_actual_clean = int(real_clean_mask.sum())
    global_total = int(len(real_clean_mask))
    global_actual_noisy = global_total - global_actual_clean

    classes_per_user = args.num_shards // args.num_users if args.partition == "shard" else "dirichlet"
    log_filename = os.path.join(
        save_dir,
        f"详细模型_FedRN_original_detailed{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
    )
    metrics_log_filename = os.path.join(
        save_dir,
        f"总体模型_FedRN_original_detailed{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
    )

    print("Results will be saved to:", log_filename)
    print("Metrics ONLY will be saved to:", metrics_log_filename)
    print(
        "Actual clean labels in noisy train set: {}/{} ({:.3f}); Actual noisy labels: {} ({:.3f})".format(
            global_actual_clean,
            global_total,
            global_actual_clean / max(global_total, 1),
            global_actual_noisy,
            global_actual_noisy / max(global_total, 1),
        )
    )

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Experiment Start: {timestamp}\n")
        f.write(f"Args: {args}\n")
        f.write(
            "Actual clean labels in noisy train set: {}/{} ({:.6f}); Actual noisy labels: {} ({:.6f})\n".format(
                global_actual_clean,
                global_total,
                global_actual_clean / max(global_total, 1),
                global_actual_noisy,
                global_actual_noisy / max(global_total, 1),
            )
        )
        f.write("=" * 100 + "\n")

    with open(metrics_log_filename, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_acc,train_loss,test_acc,test_loss,local_train_loss,"
            "selected_clients,total_count,actual_round_clean_count,actual_round_noisy_count,"
            "fedrn_clean_count,fedrn_noisy_count,train_count,actual_train_clean_count,actual_train_noisy_count,"
            "missed_clean_count,false_clean_count,correctly_excluded_noisy_count,"
            "actual_round_clean_ratio,fedrn_clean_ratio,fedrn_noisy_ratio,actual_train_clean_ratio,clean_recall,train_ratio\n"
        )

    # -------------------------------------------------------------------------
    # Dataloaders for global logging.
    # -------------------------------------------------------------------------
    logging_args = dict(batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, **logging_args)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, **logging_args)

    # -------------------------------------------------------------------------
    # Build model and local objects.
    # -------------------------------------------------------------------------
    net_glob = get_model(args).to(args.device)
    CosineSimilarity = torch.nn.CosineSimilarity()

    pred_user_noise_rates = [args.forget_rate] * args.num_users
    fed_args = dict(
        all_clients=args.all_clients,
        num_users=args.num_users,
        method=args.fed_method,
        dict_users=dict_users,
    )
    local_weights = LocalModelWeights(net_glob=net_glob, **fed_args)

    local_update_objects = get_local_update_objects_detailed(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        noise_rates=pred_user_noise_rates,
        gaussian_noise=gaussian_noise,
        real_clean_mask=real_clean_mask,
    )
    for i in range(args.num_users):
        local = local_update_objects[i]
        local.weight = copy.deepcopy(net_glob.state_dict())

    # -------------------------------------------------------------------------
    # Federated training loop.
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        local_losses = []
        client_stats = []
        args.g_epoch = epoch

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Update.
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args

            if epoch < args.warmup_epochs:
                w, loss = local.train_phase1(copy.deepcopy(net_glob).to(args.device))
            else:
                # Get similarity and expertise values.
                sim_list = []
                exp_list = []
                for user in range(args.num_users):
                    sim = CosineSimilarity(
                        local.arbitrary_output.to(args.device),
                        local_update_objects[user].arbitrary_output.to(args.device),
                    ).item()
                    exp = local_update_objects[user].expertise
                    sim_list.append(sim)
                    exp_list.append(exp)

                # Normalize similarity & expertise values with numerical guards.
                sim_list = _safe_minmax(sim_list)
                exp_list = _safe_minmax(exp_list)

                # Compute and sort reliability scores.
                prev_score = args.w_alpha * exp_list[idx] + (1 - args.w_alpha)

                score_list = []
                for neighbor_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
                    if neighbor_idx != idx:
                        score = args.w_alpha * exp + (1 - args.w_alpha) * sim
                        score_list.append([score, neighbor_idx])
                score_list.sort(key=lambda x: x[0], reverse=True)

                # Get top-k reliable neighbors.
                neighbor_list = []
                neighbor_score_list = []
                for k in range(args.num_neighbors):
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

            local_weights.update(idx, w)
            local_losses.append(copy.deepcopy(loss))
            client_stats.append(copy.deepcopy(local.last_stats))

        # Global aggregation.
        w_glob = local_weights.average()
        net_glob.load_state_dict(w_glob, strict=False)
        local_weights.init()

        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
        local_train_loss = float(np.mean(local_losses)) if local_losses else 0.0
        agg = _sum_stats(client_stats)

        log_round = "\n==================== Round {:3d} ====================".format(epoch)
        log_selected = "  --> selected_clients={} local_loss={:.4f} | {}".format(
            len(idxs_users),
            local_train_loss,
            _format_round_stats(agg),
        )
        log_metric = (
            "Train Acc: {:.2f}% | Train Loss: {:.4f} | Global Acc: {:.2f}% | Global Loss: {:.4f} | "
            "Local Train Loss: {:.4f} | {}"
        ).format(train_acc, train_loss, test_acc, test_loss, local_train_loss, _format_round_stats(agg))

        print(log_round)
        print(log_selected)
        print(log_metric)

        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(log_round + "\n")
            f.write(log_selected + "\n")
            for s in client_stats:
                f.write(_client_stats_line(s) + "\n")
            f.write(log_metric + "\n")

        with open(metrics_log_filename, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_acc:.6f},{train_loss:.6f},{test_acc:.6f},{test_loss:.6f},{local_train_loss:.6f},"
                f"{len(idxs_users)},{agg['total_count']},{agg['actual_round_clean_count']},{agg['actual_round_noisy_count']},"
                f"{agg['fedrn_clean_count']},{agg['fedrn_noisy_count']},{agg['train_count']},"
                f"{agg['actual_train_clean_count']},{agg['actual_train_noisy_count']},"
                f"{agg['missed_clean_count']},{agg['false_clean_count']},{agg['correctly_excluded_noisy_count']},"
                f"{agg['actual_round_clean_ratio']:.6f},{agg['fedrn_clean_ratio']:.6f},{agg['fedrn_noisy_ratio']:.6f},"
                f"{agg['actual_train_clean_ratio']:.6f},{agg['clean_recall']:.6f},{agg['train_ratio']:.6f}\n"
            )

    print("Total time: {:.1f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
