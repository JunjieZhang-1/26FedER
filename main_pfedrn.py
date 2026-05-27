#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PFedRN main: Personalized-model-guided FedRN.

核心思想：
1) net1 = 全局模型，参与 FedRN clean sample selection 和 FedAvg 聚合；
2) net2 = 个性化本地模型，不上传、不聚合，用于辅助判断本地样本是否干净；
3) warmup 后，clean probability = FedRN neighbor probability + personalized probability + global-local agreement 修正。

建议第一阶段先这样跑：
python main_pfedrn.py --method pfedrn --dataset cifar10 --epochs 400 --warmup_epochs 80 --num_shards 500 --group_noise_rate 0 0.8

当前版本在代码里强制 args.num_edges=1、args.neighbor_scope="global"，先暂时停用边缘服务器。
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import copy
import time
import random
# ... 下面继续你原来的 import numpy as np 等等
import os
import copy
import time
import random
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

from utils import load_dataset
from utils.options import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label
from models.nets import get_model
from models.test import test_img
from models.update import LocalUpdatePFedRN


# class DatasetSplitWithIndex(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)
#
#     def __len__(self):
#         return len(self.idxs)
#
#     def __getitem__(self, item):
#         item = int(item)
#         image, label = self.dataset[self.idxs[item]]
#         return image, label, item, self.idxs[item]


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[int(item)]]
        return image, label


def get_logits(output, head="global"):
    """兼容普通单输出模型和当前代码里的双头 CNN4Conv。"""
    if isinstance(output, (tuple, list)):
        if head == "local" and len(output) > 1:
            return output[1]
        return output[0]
    return output


def is_local_head_key(key):
    return "fc_local" in key


def is_head_key(key):
    # CIFAR 双头: fc_global / fc_local; 旧模型: linear; MNIST: fc2 常作为分类头
    return ("linear" in key) or ("fc_global" in key) or ("fc_local" in key) or key.startswith("fc2")


def fedavg(weights, weight_list, skip_local_head=True):
    if len(weights) == 0:
        return None
    total = float(sum(weight_list))
    w_avg = copy.deepcopy(weights[0])

    for k in w_avg.keys():
        if skip_local_head and is_local_head_key(k):
            # local head 不参与云端/边缘聚合
            continue
        w_avg[k] = w_avg[k] * weight_list[0]
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k] * weight_list[i]
        w_avg[k] = torch.div(w_avg[k], total)
    return w_avg


def sync_global_to_client(local_model, global_state, mode="global"):
    """
    mode='global': 同步全局模型，保留 fc_local。
    mode='personal': 个性化模型只同步 backbone，保留个人分类头。
    """
    state = local_model.state_dict()
    for k, v in global_state.items():
        if k not in state:
            continue
        if mode == "global":
            if is_local_head_key(k):
                continue
            state[k] = v.clone()
        elif mode == "personal":
            if is_head_key(k):
                continue
            state[k] = v.clone()
    local_model.load_state_dict(state)


def evaluate_personalized_apa(local_objects, dataset, dict_users_test, args):
    """评估每个客户端保留的 personalized model(net2) 在本地 holdout 上的平均准确率。"""
    acc_sum, loss_sum, valid = 0.0, 0.0, 0

    for client_idx, local in enumerate(local_objects):
        test_idxs = dict_users_test.get(client_idx, [])
        if len(test_idxs) == 0:
            continue

        loader = DataLoader(
            DatasetSplit(dataset, test_idxs),
            batch_size=args.local_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        local.net2.eval()
        correct, total, loss_total = 0.0, 0, 0.0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(args.device)
                labels = labels.to(args.device)

                logits = get_logits(local.net2(images), head="local")
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

    # 需要在 utils/options.py 中把 pfedrn 加到 choices；这里兜底，方便你先直接运行本文件。
    args.method = "pfedrn"
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    args.schedule = [int(x) for x in args.schedule]

    # PFedRN 只上传/聚合 global model，personalized model 留在本地。
    args.send_2_models = False

    # 第一阶段暂时停用边缘服务器：退化成 Client-Cloud。
    # 这样先验证 PFedRN 本身是否比 FedRN 有提升，避免边缘聚合带来额外变量。
    args.num_edges = 1
    args.neighbor_scope = "global"

    # 新参数兜底：正式建议写进 options.py
    if not hasattr(args, "pfl_local_weight"):
        args.pfl_local_weight = 0.3
    if not hasattr(args, "pfl_agree_weight"):
        args.pfl_agree_weight = 0.15
    if not hasattr(args, "pfl_personal_ep"):
        args.pfl_personal_ep = 1
    if not hasattr(args, "neighbor_scope"):
        args.neighbor_scope = "edge"  # edge 或 global

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    classes_per_user = args.num_shards // args.num_users if args.partition == "shard" else "dirichlet"
    save_dir = os.path.join("resultdate", "PFedRN")
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(
        save_dir,
        f"详细模型_PFedRN{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
    )
    metrics_log_filename = os.path.join(
        save_dir,
        f"总体模型_PFedRN{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{timestamp}.txt",
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

    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    labels = np.array(dataset_train.targets if hasattr(dataset_train, "targets") else dataset_train.train_labels)
    args.img_size = int(dataset_train[0][0].shape[1])

    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == "shard":
        dict_users = sample_noniid_shard(labels=labels, num_users=args.num_users, num_shards=args.num_shards)
    else:
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)

    # PFL 评估需要每个客户端留出一小部分本地 holdout；只对 train 部分加噪声。
    dict_users_train, dict_users_test = {}, {}
    for i in range(args.num_users):
        idxs = list(copy.deepcopy(dict_users[i]))
        rng = np.random.RandomState(args.seed + i)
        rng.shuffle(idxs)
        split = int(len(idxs) * 0.8)
        dict_users_train[i] = idxs
        dict_users_test[i] = []
        # dict_users_train[i] = idxs[:split]
        # dict_users_test[i] = idxs[split:]

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

    log_train_loader = DataLoader(dataset_train, batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)
    log_test_loader = DataLoader(dataset_test, batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)

    net_glob = get_model(args).to(args.device)
    initial_state = copy.deepcopy(net_glob.state_dict())
    gaussian_noise = torch.randn(1, args.num_channels, args.img_size, args.img_size).to(args.device)

    local_objects = []
    for i in range(args.num_users):
        local = LocalUpdatePFedRN(
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
    print("PFedRN: global model is aggregated; personalized local model is kept on each client.\n")

    with open(log_filename, "w") as f:
        f.write(f"Experiment Start: {timestamp}\n")
        f.write(f"Args: {args}\n")
        f.write("=" * 80 + "\n")
    with open(metrics_log_filename, "w") as f:
        f.write("epoch,train_loss,global_test_acc,global_test_loss,personal_apa,personal_loss,clean_ratio,agreement_ratio\n")

    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}: {} => {}".format(epoch + 1, args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay
        args.g_epoch = epoch

        edge_weights, edge_samples = [], []
        edge_logs = []
        round_losses = []
        round_clean_ratios = []
        round_agreements = []

        for edge_id, current_edge_clients in edge_clients_map.items():
            m = max(int(args.frac * len(current_edge_clients)), 1)
            selected_clients = np.random.choice(current_edge_clients, m, replace=False)

            client_weights, client_samples = [], []
            client_losses = []
            client_clean, client_agree = [], []

            for client_idx in selected_clients:
                local = local_objects[client_idx]
                local.args = args

                # net1 是 global model 副本，参与上传与聚合；
                # net2 是 personalized local model，只在该客户端本地保留。
                net_global = copy.deepcopy(local.net1).to(args.device)
                net_personal = copy.deepcopy(local.net2).to(args.device)
                sample_size = len(dict_users_train[client_idx])

                if epoch < args.warmup_epochs:
                    w, loss, w_personal, p_loss = local.train_phase1_pfedrn(
                        net_global,
                        net_personal,
                    )
                    clean_ratio = 1.0
                    agree_ratio = 0.0
                else:
                    # neighbor pool: 第一阶段建议 num_edges=1 或 neighbor_scope=global；保留 edge 仅做普通分层聚合。
                    if args.neighbor_scope == "global":
                        neighbor_pool = all_client_ids
                    else:
                        neighbor_pool = current_edge_clients

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
                    w_alpha = getattr(args, "w_alpha", 0.5)
                    prev_score = w_alpha * exp_norm[local_pos] + (1.0 - w_alpha) * 1.0

                    scores = []
                    for pos, (e_score, s_score) in enumerate(zip(exp_norm, sim_norm)):
                        global_client_id = neighbor_pool[pos]
                        if global_client_id == client_idx:
                            continue
                        score = w_alpha * e_score + (1.0 - w_alpha) * s_score
                        scores.append((score, global_client_id))
                    scores.sort(key=lambda x: x[0], reverse=True)

                    neighbor_list, neighbor_score_list = [], []
                    for score, n_id in scores[:args.num_neighbors]:
                        neighbor_list.append(copy.deepcopy(local_objects[n_id].net1).to(args.device))
                        neighbor_score_list.append(score)

                    w, loss, w_personal, p_loss, clean_ratio, agree_ratio = local.train_phase2_pfedrn(
                        net_global,
                        net_personal,
                        prev_score,
                        neighbor_list,
                        neighbor_score_list,
                    )

                w_cpu = {k: v.detach().cpu() for k, v in w.items()}
                client_weights.append(w_cpu)
                client_samples.append(sample_size)
                client_losses.append(loss)
                client_clean.append(clean_ratio)
                client_agree.append(agree_ratio)

            if len(client_weights) > 0:
                w_edge = fedavg(client_weights, client_samples, skip_local_head=True)
                edge_weights.append(w_edge)
                edge_samples.append(sum(client_samples))
                edge_logs.append(
                    "  --> [Edge {:02d}] clients={} loss={:.4f} clean={:.3f} agree={:.3f}".format(
                        edge_id + 1,
                        len(client_weights),
                        float(np.mean(client_losses)),
                        float(np.mean(client_clean)),
                        float(np.mean(client_agree)),
                    )
                )
                round_losses.extend(client_losses)
                round_clean_ratios.extend(client_clean)
                round_agreements.extend(client_agree)

        if len(edge_weights) > 0:
            w_glob = fedavg(edge_weights, edge_samples, skip_local_head=True)
            net_glob.load_state_dict(w_glob, strict=False)

            for local in local_objects:
                # net1 是全局模型副本，同步全局除 local head 之外的参数
                sync_global_to_client(local.net1, w_glob, mode="global")
                # net2 是个性化模型，只吸收全局 backbone，不覆盖本地 head
                sync_global_to_client(local.net2, w_glob, mode="personal")

        global_train_acc, global_train_loss = test_img(net_glob, log_train_loader, args)
        global_test_acc, global_test_loss = test_img(net_glob, log_test_loader, args)
        personal_apa, personal_loss = evaluate_personalized_apa(local_objects, dataset_train, dict_users_test, args)

        train_loss = float(np.mean(round_losses)) if round_losses else 0.0
        clean_ratio = float(np.mean(round_clean_ratios)) if round_clean_ratios else 0.0
        agree_ratio = float(np.mean(round_agreements)) if round_agreements else 0.0

        log_round = "\n==================== Round {:3d} ====================".format(epoch)
        log_metric = (
            "Global Acc: {:.2f}% | Global Loss: {:.4f} | Personal APA: {:.2f}% | "
            "Personal Loss: {:.4f} | Train Loss: {:.4f} | Clean: {:.3f} | Agree: {:.3f}"
        ).format(global_test_acc, global_test_loss, personal_apa, personal_loss, train_loss, clean_ratio, agree_ratio)

        print(log_round)
        for s in edge_logs:
            print(s)
        print(log_metric)

        with open(log_filename, "a") as f:
            f.write(log_round + "\n")
            for s in edge_logs:
                f.write(s + "\n")
            f.write(log_metric + "\n")
        with open(metrics_log_filename, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{global_test_acc:.6f},{global_test_loss:.6f},"
                f"{personal_apa:.6f},{personal_loss:.6f},{clean_ratio:.6f},{agree_ratio:.6f}\n"
            )

    print("Total time: {:.1f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
