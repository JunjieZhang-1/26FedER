#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_net3_personalized_only.py

用途：
    单独评估三模型中的 net3 personalized local model。

重点不看全局 CIFAR-10 test acc，而是只看 net3 在每个客户端本地数据上的效果：
    1. local train 上拟合带噪标签的能力；
    2. local train 上预测真实干净标签的能力；
    3. 高置信样本是否可靠；
    4. 当 net3 预测与带噪标签不一致时，是否具有纠错价值；
    5. 默认 local_eval：80% 本地训练、20% 本地 holdout，用本地留出干净集评估个性化泛化能力。

和 main_net2_personalized_only.py 的关系：
    - 这个脚本可以看作 net2-only 的 net3 版本；
    - 默认使用 models.nets 中的 PFedRN 双头模型，并优先使用 local head；
    - 如果你的模型不是双头，脚本也能兼容，但会提醒你检查。

建议放置：
    项目根目录，与 main_pfedrn.py / main_pfedrn_t1pr_three_model.py 同级运行。
"""

import os
import sys
import copy
import time
import random
import datetime
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import load_dataset
from utils.options import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label

# net3 在三模型里是 personalized local model，通常需要 models.nets 的双头网络。
# 如果你的项目里 models.nets 不是双头，也可以跑，但 local head 会退化为普通输出。
try:
    from models.nets import get_model
except Exception:
    from models.nets_original_fedrn import get_model


class DatasetSplitWithIndex(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        real_idx = int(self.idxs[int(item)])
        image, label = self.dataset[real_idx]
        return image, label, real_idx


class CleanTargetDataset(Dataset):
    """用于 local_holdout：用原始干净标签评估，而不是被噪声污染后的 dataset.targets。"""
    def __init__(self, dataset, idxs, clean_targets):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.clean_targets = clean_targets

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        real_idx = int(self.idxs[int(item)])
        image, _ = self.dataset[real_idx]
        label = int(self.clean_targets[real_idx])
        return image, label, real_idx


def strip_custom_args():
    """支持本脚本自定义参数，同时不修改 utils/options.py。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="完全本地训练：不聚合 backbone。更接近纯 net3 local-only。",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="每隔多少轮评估一次。local train 评估比较耗时，默认 5。",
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="local_eval",
        choices=["local_train", "local_holdout", "local_eval"],
        help=(
            "local_train：看本地训练集效果；local_holdout：本地留出干净集；"
            "local_eval：同时输出本地训练集分析和本地 holdout，不做全局测试。"
        ),
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="本地训练集比例；默认 0.8，即每个客户端 80% 训练、20% 本地 holdout 测试。",
    )
    parser.add_argument(
        "--eval_scope",
        type=str,
        default="all_clients",
        choices=["all_clients", "selected_clients"],
        help="local_train 统计范围：所有客户端或本轮选中客户端。",
    )
    parser.add_argument(
        "--noise_seed_mode",
        type=str,
        default="fedrn",
        choices=["fedrn", "per_user"],
        help="噪声注入随机种子：fedrn 尽量对齐 FedRN；per_user 每个客户端不同 seed。",
    )
    parser.add_argument(
        "--conf_thresholds",
        type=float,
        nargs="+",
        default=[0.6, 0.7, 0.8, 0.9],
        help="高置信统计阈值列表。",
    )
    parser.add_argument(
        "--aggregate_head",
        action="store_true",
        help="默认只聚合 shared backbone，不聚合分类头；打开后也聚合头，不建议。",
    )
    parser.add_argument(
        "--use_clean_holdout_label",
        type=int,
        default=1,
        help="local_holdout 是否用原始干净标签评估，默认 1。",
    )
    custom_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return custom_args


def get_targets(dataset):
    if hasattr(dataset, "targets"):
        return dataset.targets
    if hasattr(dataset, "train_labels"):
        return dataset.train_labels
    raise AttributeError("dataset has neither targets nor train_labels")


def set_target(dataset, idx, value):
    if hasattr(dataset, "targets"):
        dataset.targets[idx] = int(value)
        if hasattr(dataset, "train_labels"):
            dataset.train_labels[idx] = int(value)
    else:
        dataset.train_labels[idx] = int(value)


def get_logits(output, head="local"):
    """兼容双头模型和单头模型。双头时 head='local' 使用第二个输出。"""
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


def check_net3_model(model):
    names = [name for name, _ in model.named_parameters()]
    has_local_head = any("fc_local" in n for n in names)
    has_global_head = any("fc_global" in n for n in names)
    if has_local_head:
        print("[OK] 检测到 fc_local，net3 将使用 local head。")
    else:
        print("[Warning] 没检测到 fc_local。当前模型可能是单头，net3 会退化为普通本地模型。")
    if has_global_head:
        print("[Info] 检测到 fc_global，但本脚本训练/评估默认使用 local head。")


def fedavg_shared_backbone(state_list, weight_list, aggregate_head=False):
    """PFL 式聚合：默认只聚合共享 backbone，不聚合分类头。"""
    if len(state_list) == 0:
        return None
    total = float(sum(weight_list))
    w_avg = copy.deepcopy(state_list[0])

    for k in w_avg.keys():
        if (not aggregate_head) and is_head_key(k):
            continue
        w_avg[k] = w_avg[k] * weight_list[0]
        for i in range(1, len(state_list)):
            w_avg[k] += state_list[i][k] * weight_list[i]
        w_avg[k] = torch.div(w_avg[k], total)
    return w_avg


def sync_shared_backbone(client_model, global_state, aggregate_head=False):
    """把聚合后的 backbone 同步给客户端，默认保留客户端自己的 local head。"""
    state = client_model.state_dict()
    for k, v in global_state.items():
        if k not in state:
            continue
        if (not aggregate_head) and is_head_key(k):
            continue
        state[k] = v.clone()
    client_model.load_state_dict(state, strict=True)


def local_train_net3(model, dataset, idxs, args):
    """net3 只使用 local head 训练。"""
    model.train()
    loader = DataLoader(
        DatasetSplitWithIndex(dataset, idxs),
        batch_size=args.local_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    losses = []
    for _ in range(args.local_ep):
        for images, labels, _ in loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            logits = get_logits(model(images), head="local")
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return copy.deepcopy(model.state_dict()), float(np.mean(losses)) if losses else 0.0


def init_counter(thresholds):
    d = {
        "total": 0,
        "loss_sum": 0.0,
        "correct_noisy": 0,
        "correct_true": 0,
        "label_clean": 0,
    }
    for t in thresholds:
        key = f"{t:.2f}"
        d[f"hc_total_{key}"] = 0
        d[f"hc_true_{key}"] = 0
        d[f"hc_noisy_{key}"] = 0
        d[f"hc_label_clean_{key}"] = 0
        d[f"disagree_total_{key}"] = 0
        d[f"disagree_true_{key}"] = 0
        d[f"disagree_on_noisy_label_{key}"] = 0
        d[f"agree_total_{key}"] = 0
        d[f"agree_label_clean_{key}"] = 0
    return d


def update_counter(counter, logits, noisy_labels, true_labels, thresholds):
    probs = F.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    loss_vec = F.cross_entropy(logits, noisy_labels, reduction="none")

    total = noisy_labels.size(0)
    label_clean_mask = noisy_labels.eq(true_labels)

    counter["total"] += int(total)
    counter["loss_sum"] += float(loss_vec.sum().item())
    counter["correct_noisy"] += int(preds.eq(noisy_labels).sum().item())
    counter["correct_true"] += int(preds.eq(true_labels).sum().item())
    counter["label_clean"] += int(label_clean_mask.sum().item())

    for t in thresholds:
        key = f"{t:.2f}"
        hc = confs.ge(float(t))
        if hc.any():
            counter[f"hc_total_{key}"] += int(hc.sum().item())
            counter[f"hc_true_{key}"] += int(preds[hc].eq(true_labels[hc]).sum().item())
            counter[f"hc_noisy_{key}"] += int(preds[hc].eq(noisy_labels[hc]).sum().item())
            counter[f"hc_label_clean_{key}"] += int(label_clean_mask[hc].sum().item())

        disagree = hc & (~preds.eq(noisy_labels))
        if disagree.any():
            counter[f"disagree_total_{key}"] += int(disagree.sum().item())
            counter[f"disagree_true_{key}"] += int(preds[disagree].eq(true_labels[disagree]).sum().item())
            # 这个统计看：高置信且不同意 noisy label 的样本里，有多少本来就是噪声标签。
            counter[f"disagree_on_noisy_label_{key}"] += int((~label_clean_mask[disagree]).sum().item())

        agree = hc & preds.eq(noisy_labels)
        if agree.any():
            counter[f"agree_total_{key}"] += int(agree.sum().item())
            counter[f"agree_label_clean_{key}"] += int(label_clean_mask[agree].sum().item())


def finalize_counter(counter, thresholds, prefix="train"):
    total = max(counter["total"], 1)
    result = {
        f"{prefix}_loss_noisy": counter["loss_sum"] / total,
        f"{prefix}_acc_noisy": 100.0 * counter["correct_noisy"] / total,
        f"{prefix}_acc_true": 100.0 * counter["correct_true"] / total,
        f"{prefix}_label_clean_ratio": counter["label_clean"] / total,
        f"{prefix}_total": counter["total"],
    }
    for t in thresholds:
        key = f"{t:.2f}"
        hc_total = counter[f"hc_total_{key}"]
        dis_total = counter[f"disagree_total_{key}"]
        agr_total = counter[f"agree_total_{key}"]
        result[f"{prefix}_hc_cov_{key}"] = hc_total / total if total else 0.0
        result[f"{prefix}_hc_acc_true_{key}"] = 100.0 * counter[f"hc_true_{key}"] / hc_total if hc_total else 0.0
        result[f"{prefix}_hc_acc_noisy_{key}"] = 100.0 * counter[f"hc_noisy_{key}"] / hc_total if hc_total else 0.0
        result[f"{prefix}_hc_label_clean_{key}"] = counter[f"hc_label_clean_{key}"] / hc_total if hc_total else 0.0
        result[f"{prefix}_disagree_count_{key}"] = dis_total
        result[f"{prefix}_disagree_corr_acc_{key}"] = 100.0 * counter[f"disagree_true_{key}"] / dis_total if dis_total else 0.0
        result[f"{prefix}_disagree_noise_precision_{key}"] = counter[f"disagree_on_noisy_label_{key}"] / dis_total if dis_total else 0.0
        result[f"{prefix}_agree_count_{key}"] = agr_total
        result[f"{prefix}_agree_label_precision_{key}"] = counter[f"agree_label_clean_{key}"] / agr_total if agr_total else 0.0
    return result


def evaluate_net3_local_train(client_models, dataset_train, dict_users_train, clean_targets, args, client_ids=None):
    """在本地训练集上评估：同时看 noisy label 和 true label。"""
    thresholds = args.conf_thresholds
    counter = init_counter(thresholds)
    if client_ids is None:
        client_ids = list(range(len(client_models)))

    for cid in client_ids:
        idxs = dict_users_train[cid]
        if len(idxs) == 0:
            continue
        model = client_models[cid].to(args.device)
        model.eval()
        loader = DataLoader(
            DatasetSplitWithIndex(dataset_train, idxs),
            batch_size=args.local_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        with torch.no_grad():
            for images, noisy_labels, real_idxs in loader:
                images = images.to(args.device)
                noisy_labels = noisy_labels.to(args.device)
                true_labels = torch.as_tensor(
                    [int(clean_targets[int(i)]) for i in real_idxs],
                    dtype=torch.long,
                    device=args.device,
                )
                logits = get_logits(model(images), head="local")
                update_counter(counter, logits, noisy_labels, true_labels, thresholds)
        model.cpu()
    return finalize_counter(counter, thresholds, prefix="train")


def evaluate_net3_local_holdout(client_models, dataset_train, dict_users_test, clean_targets, args):
    """在本地 holdout 上评估。默认 holdout 用 clean_targets。"""
    acc_sum, loss_sum, valid = 0.0, 0.0, 0
    for cid, model in enumerate(client_models):
        idxs = dict_users_test.get(cid, [])
        if len(idxs) == 0:
            continue
        model = model.to(args.device)
        model.eval()
        if int(args.use_clean_holdout_label) == 1:
            ds = CleanTargetDataset(dataset_train, idxs, clean_targets)
        else:
            ds = DatasetSplitWithIndex(dataset_train, idxs)
        loader = DataLoader(
            ds,
            batch_size=args.local_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        correct, total, loss_total = 0, 0, 0.0
        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                logits = get_logits(model(images), head="local")
                loss_total += F.cross_entropy(logits, labels, reduction="sum").item()
                pred = logits.argmax(dim=1)
                correct += int(pred.eq(labels).sum().item())
                total += labels.size(0)
        model.cpu()
        if total > 0:
            acc_sum += 100.0 * correct / total
            loss_sum += loss_total / total
            valid += 1
    if valid == 0:
        return 0.0, 0.0
    return acc_sum / valid, loss_sum / valid


def evaluate_net3_global_test(client_models, dataset_test, args):
    """每个客户端模型都在全局干净测试集上测试，然后取平均。极端 shard 下仅作参考。"""
    loader = DataLoader(
        dataset_test,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    acc_sum, loss_sum, valid = 0.0, 0.0, 0
    for model in client_models:
        model = model.to(args.device)
        model.eval()
        correct, total, loss_total = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                logits = get_logits(model(images), head="local")
                loss_total += F.cross_entropy(logits, labels, reduction="sum").item()
                pred = logits.argmax(dim=1)
                correct += int(pred.eq(labels).sum().item())
                total += labels.size(0)
        model.cpu()
        if total > 0:
            acc_sum += 100.0 * correct / total
            loss_sum += loss_total / total
            valid += 1
    if valid == 0:
        return 0.0, 0.0
    return acc_sum / valid, loss_sum / valid


def compute_label_correct_stats(dataset_train, clean_targets, dict_users_train, selected_clients):
    targets = get_targets(dataset_train)
    total, correct = 0, 0
    for cid in selected_clients:
        for idx in dict_users_train[cid]:
            total += 1
            if int(targets[idx]) == int(clean_targets[idx]):
                correct += 1
    return total, correct, (correct / total if total else 0.0)


def main():
    start = time.time()
    custom = strip_custom_args()
    args = args_parser()

    args.method = "net3_personalized_only"
    args.send_2_models = False
    args.local_only = bool(custom.local_only)
    args.eval_every = max(int(custom.eval_every), 1)
    args.test_mode = custom.test_mode
    args.train_ratio = float(custom.train_ratio)
    args.eval_scope = custom.eval_scope
    args.noise_seed_mode = custom.noise_seed_mode
    args.conf_thresholds = [float(x) for x in custom.conf_thresholds]
    args.aggregate_head = bool(custom.aggregate_head)
    args.use_clean_holdout_label = int(custom.use_clean_holdout_label)

    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    args.schedule = [int(x) for x in args.schedule]

    if not torch.cuda.is_available() and args.gpu != -1:
        raise RuntimeError("CUDA 不可用。调试可加 --gpu -1 使用 CPU。")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    clean_targets_before_noise = copy.deepcopy(list(get_targets(dataset_train)))
    labels = np.array(clean_targets_before_noise)
    args.img_size = int(dataset_train[0][0].shape[1])

    # 用户划分
    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
        split_tag = "iid"
    elif args.partition == "shard":
        dict_users = sample_noniid_shard(
            labels=labels,
            num_users=args.num_users,
            num_shards=args.num_shards,
        )
        split_tag = f"{args.num_shards // args.num_users}shards_per_client"
    elif args.partition == "dirichlet":
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)
        split_tag = f"dirichlet_alpha{args.dd_alpha}"
    else:
        raise ValueError(f"Unsupported partition: {args.partition}")

    # 本地 train / holdout 划分
    dict_users_train, dict_users_test = {}, {}
    per_client_unique_labels = []
    need_holdout = args.test_mode in ["local_holdout", "local_eval"] and args.train_ratio < 1.0

    for user in range(args.num_users):
        idxs = list(copy.deepcopy(dict_users[user]))
        per_client_unique_labels.append(len(set(labels[idxs].tolist())))
        if need_holdout:
            rng = np.random.RandomState(args.seed + user)
            rng.shuffle(idxs)
            split_point = int(len(idxs) * args.train_ratio)
            dict_users_train[user] = idxs[:split_point]
            dict_users_test[user] = idxs[split_point:]
        else:
            dict_users_train[user] = idxs
            dict_users_test[user] = []

    # 解析噪声率
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

    # 标签噪声：只加到客户端训练集。holdout 默认保持干净。
    total_noisy = 0
    for user, (noise_type, noise_rate) in enumerate(user_noise_type_rates):
        if noise_type == "clean":
            continue
        data_indices = list(copy.deepcopy(dict_users_train[user]))
        if args.noise_seed_mode == "fedrn":
            random.seed(args.seed)
        else:
            random.seed(args.seed + user)
        random.shuffle(data_indices)
        noise_num = int(len(data_indices) * noise_rate)
        total_noisy += noise_num
        for d_idx in data_indices[:noise_num]:
            y = get_targets(dataset_train)[d_idx]
            noisy_y = noisify_label(y, num_classes=args.num_classes, noise_type=noise_type)
            set_target(dataset_train, d_idx, noisy_y)

    # 初始化 net3 客户端模型
    global_model = get_model(args).to(args.device)
    check_net3_model(global_model)
    initial_state = copy.deepcopy(global_model.state_dict())

    client_models = []
    for _ in range(args.num_users):
        model = get_model(args)
        model.load_state_dict(initial_state)
        model.cpu()
        client_models.append(model)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mode_tag = "localonly" if args.local_only else "sharedbackbone"
    save_dir = os.path.join("resultdate", "Net3_personalized_only")
    os.makedirs(save_dir, exist_ok=True)

    detail_path = os.path.join(
        save_dir,
        f"详细模型_Net3PersonalOnly_{mode_tag}_{args.epochs}_{args.dataset}_{split_tag}_{args.group_noise_rate}_{timestamp}.txt",
    )
    metric_path = os.path.join(
        save_dir,
        f"总体模型_Net3PersonalOnly_{mode_tag}_{args.epochs}_{args.dataset}_{split_tag}_{args.group_noise_rate}_{timestamp}.txt",
    )

    print("Results will be saved to:", detail_path)
    print("Metrics ONLY will be saved to:", metric_path)
    print("Mode:", "Local-only，net3 完全本地" if args.local_only else "PFL，共享 backbone，保留本地 local head")
    print("Eval mode:", args.test_mode)
    print("Eval scope:", args.eval_scope)
    print("Noise seed mode:", args.noise_seed_mode)
    print("Total intended noisy labels:", total_noisy)
    print("Average unique labels/client: {:.2f}, min={}, max={}".format(
        float(np.mean(per_client_unique_labels)),
        int(np.min(per_client_unique_labels)),
        int(np.max(per_client_unique_labels)),
    ))
    print("Args:", args)

    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment Start: {timestamp}\n")
        f.write(f"Mode: {'local_only' if args.local_only else 'shared_backbone_pfl'}\n")
        f.write("Model role: net3 personalized local model, local head only\n")
        f.write(f"Eval mode: {args.test_mode}\n")
        f.write(f"Eval scope: {args.eval_scope}\n")
        f.write(f"Noise seed mode: {args.noise_seed_mode}\n")
        f.write(f"Args: {args}\n")
        f.write(f"Average unique labels/client: {np.mean(per_client_unique_labels):.4f}\n")
        f.write(f"Unique labels/client min/max: {np.min(per_client_unique_labels)} / {np.max(per_client_unique_labels)}\n")
        f.write(f"Total intended noisy labels: {total_noisy}\n")
        f.write("=" * 100 + "\n")

    # CSV 表头：保留最重要指标 + 0.8 阈值指标，详细阈值都写入 detail log。
    with open(metric_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,selected_clients,selected_train_label_correct,"
            "local_train_acc_noisy,local_train_acc_true,local_train_loss_noisy,"
            "hc_cov_0.80,hc_acc_true_0.80,disagree_count_0.80,disagree_corr_acc_0.80,"
            "agree_count_0.80,agree_label_precision_0.80,"
            "local_holdout_acc,local_holdout_loss\n"
        )

    all_client_ids = list(range(args.num_users))

    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print(f"Learning Rate Decay Epoch {epoch + 1}: {args.lr} => {args.lr * args.lr_decay}")
            args.lr *= args.lr_decay

        selected_num = max(int(args.frac * args.num_users), 1)
        selected_clients = np.random.choice(all_client_ids, selected_num, replace=False)

        selected_states, selected_sizes, selected_losses = [], [], []

        for client_id in selected_clients:
            model = copy.deepcopy(client_models[client_id]).to(args.device)
            state, loss = local_train_net3(model, dataset_train, dict_users_train[client_id], args)

            client_models[client_id].load_state_dict(state, strict=True)
            client_models[client_id].cpu()

            selected_states.append({k: v.detach().cpu() for k, v in state.items()})
            selected_sizes.append(len(dict_users_train[client_id]))
            selected_losses.append(loss)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 标准 PFL：只聚合共享 backbone，同步给所有客户端；local head 保持本地。
        if (not args.local_only) and len(selected_states) > 0:
            global_state = fedavg_shared_backbone(
                selected_states,
                selected_sizes,
                aggregate_head=args.aggregate_head,
            )
            global_model.load_state_dict(global_state, strict=True)
            global_model.cpu()
            for client_model in client_models:
                sync_shared_backbone(client_model, global_state, aggregate_head=args.aggregate_head)

        train_loss = float(np.mean(selected_losses)) if selected_losses else 0.0
        train_total, train_correct, train_correct_ratio = compute_label_correct_stats(
            dataset_train, clean_targets_before_noise, dict_users_train, selected_clients
        )

        # 默认不每轮全评估，避免太慢。
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs - 1)
        local_stats = None
        local_holdout_acc, local_holdout_loss = -1.0, -1.0
        if do_eval:
            eval_client_ids = None
            if args.eval_scope == "selected_clients":
                eval_client_ids = list(map(int, selected_clients))

            if args.test_mode in ["local_train", "local_eval"]:
                local_stats = evaluate_net3_local_train(
                    client_models,
                    dataset_train,
                    dict_users_train,
                    clean_targets_before_noise,
                    args,
                    client_ids=eval_client_ids,
                )
            if args.test_mode in ["local_holdout", "local_eval"]:
                local_holdout_acc, local_holdout_loss = evaluate_net3_local_holdout(
                    client_models,
                    dataset_train,
                    dict_users_test,
                    clean_targets_before_noise,
                    args,
                )
        if local_stats is None:
            local_stats = {
                "train_acc_noisy": -1.0,
                "train_acc_true": -1.0,
                "train_loss_noisy": -1.0,
                "train_total": 0,
                "train_label_clean_ratio": -1.0,
            }
            for t in args.conf_thresholds:
                key = f"{t:.2f}"
                local_stats[f"train_hc_cov_{key}"] = -1.0
                local_stats[f"train_hc_acc_true_{key}"] = -1.0
                local_stats[f"train_hc_acc_noisy_{key}"] = -1.0
                local_stats[f"train_hc_label_clean_{key}"] = -1.0
                local_stats[f"train_disagree_count_{key}"] = 0
                local_stats[f"train_disagree_corr_acc_{key}"] = -1.0
                local_stats[f"train_disagree_noise_precision_{key}"] = -1.0
                local_stats[f"train_agree_count_{key}"] = 0
                local_stats[f"train_agree_label_precision_{key}"] = -1.0

        log_round = f"\n==================== Round {epoch:3d} ===================="
        log_metric = (
            f"Net3 TrainLoss: {train_loss:.4f} | Selected Clients: {selected_num} | "
            f"SelectedTrainLabelCorrect: {train_correct_ratio:.3f} ({train_correct}/{train_total}) | "
            f"LocalTrain Acc(noisy): {local_stats['train_acc_noisy']:.2f}% | "
            f"LocalTrain Acc(true): {local_stats['train_acc_true']:.2f}% | "
            f"LocalTrain Loss(noisy): {local_stats['train_loss_noisy']:.4f} | "
            f"Holdout Acc: {local_holdout_acc:.2f}%"
        )
        print(log_round)
        print(log_metric)

        # 重点输出 0.8 阈值；其它阈值写到 detail。
        k80 = "0.80"
        log_hc80 = (
            f"  [conf>=0.80] HC_Cov={local_stats.get('train_hc_cov_' + k80, -1):.3f} | "
            f"HC_AccTrue={local_stats.get('train_hc_acc_true_' + k80, -1):.2f}% | "
            f"DisagreeCount={local_stats.get('train_disagree_count_' + k80, 0)} | "
            f"DisagreeCorrAcc={local_stats.get('train_disagree_corr_acc_' + k80, -1):.2f}% | "
            f"DisagreeNoisePrecision={local_stats.get('train_disagree_noise_precision_' + k80, -1):.3f} | "
            f"AgreeCount={local_stats.get('train_agree_count_' + k80, 0)} | "
            f"AgreeLabelPrecision={local_stats.get('train_agree_label_precision_' + k80, -1):.3f}"
        )
        print(log_hc80)

        with open(detail_path, "a", encoding="utf-8") as f:
            f.write(log_round + "\n")
            f.write(log_metric + "\n")
            f.write(log_hc80 + "\n")
            if do_eval:
                for t in args.conf_thresholds:
                    key = f"{t:.2f}"
                    f.write(
                        f"  [conf>={key}] "
                        f"HC_Cov={local_stats.get('train_hc_cov_' + key, -1):.4f} "
                        f"HC_AccTrue={local_stats.get('train_hc_acc_true_' + key, -1):.2f}% "
                        f"HC_AccNoisy={local_stats.get('train_hc_acc_noisy_' + key, -1):.2f}% "
                        f"HC_LabelClean={local_stats.get('train_hc_label_clean_' + key, -1):.4f} "
                        f"DisagreeCount={local_stats.get('train_disagree_count_' + key, 0)} "
                        f"DisagreeCorrAcc={local_stats.get('train_disagree_corr_acc_' + key, -1):.2f}% "
                        f"DisagreeNoisePrecision={local_stats.get('train_disagree_noise_precision_' + key, -1):.4f} "
                        f"AgreeCount={local_stats.get('train_agree_count_' + key, 0)} "
                        f"AgreeLabelPrecision={local_stats.get('train_agree_label_precision_' + key, -1):.4f}\n"
                    )

        with open(metric_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{selected_num},{train_correct_ratio:.6f},"
                f"{local_stats['train_acc_noisy']:.6f},{local_stats['train_acc_true']:.6f},{local_stats['train_loss_noisy']:.6f},"
                f"{local_stats.get('train_hc_cov_' + k80, -1):.6f},"
                f"{local_stats.get('train_hc_acc_true_' + k80, -1):.6f},"
                f"{local_stats.get('train_disagree_count_' + k80, 0)},"
                f"{local_stats.get('train_disagree_corr_acc_' + k80, -1):.6f},"
                f"{local_stats.get('train_agree_count_' + k80, 0)},"
                f"{local_stats.get('train_agree_label_precision_' + k80, -1):.6f},"
                f"{local_holdout_acc:.6f},{local_holdout_loss:.6f}\n"
            )

    print("Total time: {:.1f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
