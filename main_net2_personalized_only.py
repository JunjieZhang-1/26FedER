#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_net2_personalized_only.py

用途：
    单独评估“本地个性化 net2”的效果，用来判断：
    1. net2 这个本地/个性化模型在当前 CIFAR-10 + shard + 高噪声下到底能达到什么精度；
    2. net2 是否真的比全局模型更适合做本地 CNLL / 标签纠错；
    3. shared-backbone PFL 与完全 local-only 的差异。

和 main_pfl_only_shard_effect.py 的区别：
    - 本脚本默认使用每个客户端的全部本地训练数据，不再做 80/20 holdout。
    - 标签噪声注入到全部本地训练数据。
    - 评估时默认在全局干净 dataset_test 上测试每个客户端模型，然后取客户端平均 APA。
    - 这样更接近你当前 FedRN/PFedRN 主实验：客户端训练使用全部本地数据，全局测试集保持干净。
    - 聚合方式：只聚合 shared backbone，不聚合 private head，相当于 PFL 中 net2 的单独效果。
    - 如果加 --local_only，则完全不聚合 backbone，变成每个客户端只训练自己的本地模型。

建议放置：
    项目根目录，与 main_pfedrn.py 同级运行。
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

# 强制优先使用原始 FedRN 单头模型，避免误用 PFedRN 双头网络。
try:
    from models.nets_original_fedrn import get_model
except Exception:
    from models.nets import get_model


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[int(item)]]
        return image, label


def strip_custom_args():
    """支持本脚本自定义参数，同时不修改 utils/options.py。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--local_only", action="store_true",
                        help="完全本地训练：不聚合 backbone，不同步全局模型。")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="每隔多少轮评估一次 APA。默认每轮评估。")
    parser.add_argument("--test_mode", type=str, default="global",
                        choices=["global", "local_holdout"],
                        help=(
                            "评估方式：global 表示每个客户端模型都在全局干净 dataset_test 上测；"
                            "local_holdout 表示每个客户端留 20% 本地干净 holdout 测。"
                        ))
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="local_holdout 模式下使用的本地训练比例；global 模式默认使用全部本地数据。")
    parser.add_argument("--noise_seed_mode", type=str, default="fedrn",
                        choices=["fedrn", "per_user"],
                        help="噪声注入随机种子：fedrn 使用 random.seed(args.seed)，per_user 使用 args.seed + user。")
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
        dataset.targets[idx] = value
        if hasattr(dataset, "train_labels"):
            dataset.train_labels[idx] = value
    else:
        dataset.train_labels[idx] = value


def is_private_head_key(key):
    """
    PFL 私有分类头判定。
    CIFAR 原始 CNN4Conv: linear.*
    MNIST 常见分类头: fc2.*
    其它模型常见分类器: classifier.*
    """
    return (
            "linear" in key
            or key.startswith("fc2")
            or "classifier" in key
    )


def fedavg_pfl(state_list, weight_list):
    """标准 PFL 聚合：只聚合共享 backbone，不聚合私有分类头。"""
    if len(state_list) == 0:
        return None

    total = float(sum(weight_list))
    w_avg = copy.deepcopy(state_list[0])

    for k in w_avg.keys():
        if is_private_head_key(k):
            continue
        w_avg[k] = w_avg[k] * weight_list[0]
        for i in range(1, len(state_list)):
            w_avg[k] += state_list[i][k] * weight_list[i]
        w_avg[k] = torch.div(w_avg[k], total)

    return w_avg


def sync_shared_backbone(client_model, global_state):
    """只同步共享 backbone，不覆盖客户端私有分类头。"""
    state = client_model.state_dict()
    for k, v in global_state.items():
        if k in state and not is_private_head_key(k):
            state[k] = v.clone()
    client_model.load_state_dict(state, strict=True)


def get_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def check_single_head_model(model):
    names = [name for name, _ in model.named_parameters()]
    if any("fc_global" in n or "fc_local" in n for n in names):
        raise RuntimeError(
            "检测到当前模型是 PFedRN 双头模型(fc_global/fc_local)。\n"
            "本脚本用于单独评估 net2 / 个性化 PFL，请使用单头模型：models/nets_original_fedrn.py。"
        )
    if not any(is_private_head_key(n) for n in names):
        print("[Warning] 没检测到 linear/fc2/classifier 分类头，确认 is_private_head_key 是否需要改。")


def local_train(model, dataset, idxs, args):
    model.train()
    loader = DataLoader(
        DatasetSplit(dataset, idxs),
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
        for images, labels in loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            logits = get_logits(model(images))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return copy.deepcopy(model.state_dict()), float(np.mean(losses)) if losses else 0.0


def evaluate_personalized_global_test(client_models, dataset_test, args):
    """
    每个客户端模型都在同一个全局干净 test set 上评估，然后取客户端平均。
    这个指标更适合判断 net2 的分类泛化能力。
    """
    acc_sum, loss_sum, valid_clients = 0.0, 0.0, 0

    loader = DataLoader(
        dataset_test,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    for model in client_models:
        model = model.to(args.device)
        model.eval()

        correct, total, loss_total = 0.0, 0, 0.0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                logits = get_logits(model(images))
                loss_total += F.cross_entropy(logits, labels, reduction="sum").item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).float().sum().item()
                total += labels.size(0)

        model.cpu()
        if total > 0:
            acc_sum += 100.0 * correct / total
            loss_sum += loss_total / total
            valid_clients += 1

    if valid_clients == 0:
        return 0.0, 0.0
    return acc_sum / valid_clients, loss_sum / valid_clients


def evaluate_personalized_local_holdout(client_models, dataset_train, dict_users_test, args):
    """每个客户端模型在自己的本地 holdout 上测试，然后取客户端平均 APA。"""
    acc_sum, loss_sum, valid_clients = 0.0, 0.0, 0

    for client_id, model in enumerate(client_models):
        idxs = dict_users_test.get(client_id, [])
        if len(idxs) == 0:
            continue

        model = model.to(args.device)
        model.eval()
        loader = DataLoader(
            DatasetSplit(dataset_train, idxs),
            batch_size=args.local_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        correct, total, loss_total = 0.0, 0, 0.0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                logits = get_logits(model(images))
                loss_total += F.cross_entropy(logits, labels, reduction="sum").item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).float().sum().item()
                total += labels.size(0)

        model.cpu()

        if total > 0:
            acc_sum += 100.0 * correct / total
            loss_sum += loss_total / total
            valid_clients += 1

    if valid_clients == 0:
        return 0.0, 0.0
    return acc_sum / valid_clients, loss_sum / valid_clients


def compute_label_correct_stats(dataset_train, original_clean_targets, dict_users_train, selected_clients):
    """统计选中客户端训练标签和真实标签一致的比例，仅用于日志分析。"""
    targets = get_targets(dataset_train)
    total, correct = 0, 0
    for cid in selected_clients:
        for idx in dict_users_train[cid]:
            total += 1
            if int(targets[idx]) == int(original_clean_targets[idx]):
                correct += 1
    if total == 0:
        return 0, 0, 0.0
    return total, correct, correct / total


def main():
    start = time.time()
    custom = strip_custom_args()
    args = args_parser()

    args.method = "net2_personalized_only"
    args.send_2_models = False
    args.local_only = bool(custom.local_only)
    args.eval_every = max(int(custom.eval_every), 1)
    args.test_mode = custom.test_mode
    args.train_ratio = float(custom.train_ratio)
    args.noise_seed_mode = custom.noise_seed_mode

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

    # 训练/测试划分
    dict_users_train, dict_users_test = {}, {}
    per_client_unique_labels = []

    for user in range(args.num_users):
        idxs = list(copy.deepcopy(dict_users[user]))
        per_client_unique_labels.append(len(set(labels[idxs].tolist())))

        if args.test_mode == "local_holdout":
            rng = np.random.RandomState(args.seed + user)
            rng.shuffle(idxs)
            split_point = int(len(idxs) * args.train_ratio)
            dict_users_train[user] = idxs[:split_point]
            dict_users_test[user] = idxs[split_point:]
        else:
            # global test 模式下，客户端本地全部数据用于训练，全局 dataset_test 用于评估。
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

    # 标签噪声：只加到客户端训练集。
    total_noisy = 0
    for user, (noise_type, noise_rate) in enumerate(user_noise_type_rates):
        if noise_type == "clean":
            continue
        data_indices = list(copy.deepcopy(dict_users_train[user]))

        # fedrn 模式：和 FedRN original 对齐；per_user 模式：每个客户端不同 seed。
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

    # 初始化全局模型和客户端个性化模型
    global_model = get_model(args).to(args.device)
    check_single_head_model(global_model)
    initial_state = copy.deepcopy(global_model.state_dict())

    client_models = []
    for _ in range(args.num_users):
        model = get_model(args)
        model.load_state_dict(initial_state)
        model.cpu()
        client_models.append(model)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mode_tag = "localonly" if args.local_only else "sharedbackbone"
    save_dir = os.path.join("resultdate", "Net2_personalized_only")
    os.makedirs(save_dir, exist_ok=True)

    detail_path = os.path.join(
        save_dir,
        f"详细模型_Net2PersonalOnly_{mode_tag}_{args.epochs}_{args.dataset}_{split_tag}_{args.group_noise_rate}_{timestamp}.txt",
    )
    metric_path = os.path.join(
        save_dir,
        f"总体模型_Net2PersonalOnly_{mode_tag}_{args.epochs}_{args.dataset}_{split_tag}_{args.group_noise_rate}_{timestamp}.txt",
    )

    print("Results will be saved to:", detail_path)
    print("Metrics ONLY will be saved to:", metric_path)
    print("Mode:", "Local-only，本地模型完全独立" if args.local_only else "Standard PFL，共享 backbone，私有 head")
    print("Eval mode:", args.test_mode)
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
        f.write(f"Eval mode: {args.test_mode}\n")
        f.write(f"Noise seed mode: {args.noise_seed_mode}\n")
        f.write(f"Args: {args}\n")
        f.write(f"Average unique labels/client: {np.mean(per_client_unique_labels):.4f}\n")
        f.write(
            f"Unique labels/client min/max: {np.min(per_client_unique_labels)} / {np.max(per_client_unique_labels)}\n")
        f.write(f"Total intended noisy labels: {total_noisy}\n")
        f.write("=" * 80 + "\n")

    with open(metric_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,personal_acc,personal_loss,selected_clients,"
            "avg_unique_labels_per_client,selected_train_label_correct\n"
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
            state, loss = local_train(model, dataset_train, dict_users_train[client_id], args)

            # 保存客户端自己的完整模型，包括私有分类头。
            client_models[client_id].load_state_dict(state, strict=True)
            client_models[client_id].cpu()

            selected_states.append({k: v.detach().cpu() for k, v in state.items()})
            selected_sizes.append(len(dict_users_train[client_id]))
            selected_losses.append(loss)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 标准 PFL：聚合共享 backbone，同步给所有客户端；私有 head 保持本地。
        if (not args.local_only) and len(selected_states) > 0:
            global_state = fedavg_pfl(selected_states, selected_sizes)
            global_model.load_state_dict(global_state, strict=True)
            global_model.cpu()
            for client_model in client_models:
                sync_shared_backbone(client_model, global_state)

        train_loss = float(np.mean(selected_losses)) if selected_losses else 0.0
        train_total, train_correct, train_correct_ratio = compute_label_correct_stats(
            dataset_train, clean_targets_before_noise, dict_users_train, selected_clients
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            if args.test_mode == "local_holdout":
                personal_acc, personal_loss = evaluate_personalized_local_holdout(
                    client_models, dataset_train, dict_users_test, args
                )
            else:
                personal_acc, personal_loss = evaluate_personalized_global_test(
                    client_models, dataset_test, args
                )
        else:
            personal_acc, personal_loss = -1.0, -1.0

        log_round = f"\n==================== Round {epoch:3d} ===================="
        log_metric = (
            f"Net2/PFL Acc: {personal_acc:.2f}% | Personal Loss: {personal_loss:.4f} | "
            f"Train Loss: {train_loss:.4f} | Selected Clients: {selected_num} | "
            f"TrainLabelCorrect: {train_correct_ratio:.3f} ({train_correct}/{train_total}) | "
            f"Avg Labels/Client: {np.mean(per_client_unique_labels):.2f}"
        )
        print(log_round)
        print(log_metric)

        with open(detail_path, "a", encoding="utf-8") as f:
            f.write(log_round + "\n")
            f.write(log_metric + "\n")

        with open(metric_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{personal_acc:.6f},{personal_loss:.6f},"
                f"{selected_num},{np.mean(per_client_unique_labels):.6f},{train_correct_ratio:.6f}\n"
            )

    print("Total time: {:.1f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
