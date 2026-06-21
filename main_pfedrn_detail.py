#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PFedRN main with detailed clean/noisy selection logging.

需要配合已经更新过的 models/update.py 中的 LocalUpdatePFedRN 使用：
- LocalUpdatePFedRN.__init__(..., clean_label_mask=None)
- local.last_selection_stats

新增输出：
1) 本轮被选中客户端的总训练样本数；
2) 实际参与 global 训练的样本数；
3) 方法认为干净/噪声的样本数；
4) 真实干净/真实噪声样本数；
5) 干净被误判为噪声 missed_clean；
6) 噪声被误判为干净 false_clean。
"""

import copy
import datetime
import inspect
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Windows/conda 环境兼容；路径不存在时不影响 Linux/其他环境。
_CONDA_DLL_PATH = r"C:\Users\25839\.conda\envs\improve-FedRN-main\Library\bin"
if os.path.exists(_CONDA_DLL_PATH):
    os.environ["PATH"] = _CONDA_DLL_PATH + os.pathsep + os.environ.get("PATH", "")

from utils import load_dataset
from utils.options import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label
from models.nets import get_model
from models.test import test_img
from models.update import LocalUpdatePFedRN


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[int(item)]]
        return image, label


def get_dataset_labels(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "train_labels"):
        return np.asarray(dataset.train_labels)
    raise AttributeError("dataset has no targets/train_labels attribute")


def set_dataset_label(dataset, idx, label):
    if hasattr(dataset, "targets"):
        dataset.targets[idx] = label
    else:
        dataset.train_labels[idx] = label


def get_logits(output, head="global"):
    """兼容普通单输出模型和双头模型。"""
    if isinstance(output, (tuple, list)):
        if head == "local" and len(output) > 1:
            return output[1]
        return output[0]
    return output


def is_local_head_key(key):
    return "fc_local" in key


def is_head_key(key):
    return (
        "linear" in key
        or "fc_global" in key
        or "fc_local" in key
        or key.startswith("fc2")
        or "classifier" in key
    )


def fedavg(weights, weight_list, skip_local_head=True):
    if len(weights) == 0:
        return None

    total = float(sum(weight_list))
    if total <= 0:
        raise ValueError("fedavg received non-positive total aggregation weight")

    w_avg = copy.deepcopy(weights[0])
    for k in w_avg.keys():
        if skip_local_head and is_local_head_key(k):
            # 个性化 local head 不参与云端聚合。
            continue
        w_avg[k] = w_avg[k] * weight_list[0]
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k] * weight_list[i]
        w_avg[k] = torch.div(w_avg[k], total)
    return w_avg


def sync_global_to_client(local_model, global_state, mode="global"):
    """
    mode='global': 同步全局模型，保留 fc_local。
    mode='personal': 个性化模型只同步 backbone，保留本地分类头。
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
        else:
            raise ValueError(f"Unknown sync mode: {mode}")
    local_model.load_state_dict(state)


def evaluate_personalized_apa(local_objects, dataset, dict_users_test, args):
    """评估每个客户端 personalized model(net2) 在本地 holdout 上的平均准确率。"""
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


_SELECTION_COUNT_KEYS = [
    "selected_total_count",
    "global_train_count",
    "pred_clean_count",
    "pred_noisy_count",
    "true_clean_count",
    "true_noise_count",
    "actual_train_clean_count",
    "missed_clean_count",
    "false_clean_count",
    "true_noise_rejected_count",
    "total_wrong_count",
]


def empty_selection_stats():
    out = {k: 0 for k in _SELECTION_COUNT_KEYS}
    out.update(
        {
            "train_ratio": 0.0,
            "actual_train_clean_ratio": 0.0,
            "clean_precision": 0.0,
            "clean_recall": 0.0,
            "missed_clean_ratio": 0.0,
            "false_clean_ratio": 0.0,
        }
    )
    return out


def aggregate_selection_stats(stats_list):
    """按样本数汇总客户端筛样统计。"""
    out = {k: 0 for k in _SELECTION_COUNT_KEYS}
    valid_num = 0
    for st in stats_list:
        if not st:
            continue
        valid_num += 1
        for k in _SELECTION_COUNT_KEYS:
            out[k] += int(st.get(k, 0))

    if valid_num == 0 or out["selected_total_count"] == 0:
        return empty_selection_stats()

    total = float(out["selected_total_count"] + 1e-8)
    train = float(out["global_train_count"] + 1e-8)
    true_clean = float(out["true_clean_count"] + 1e-8)
    true_noise = float(out["true_noise_count"] + 1e-8)

    out["train_ratio"] = float(out["global_train_count"]) / total
    out["actual_train_clean_ratio"] = float(out["actual_train_clean_count"]) / train
    out["clean_precision"] = out["actual_train_clean_ratio"]
    out["clean_recall"] = float(out["actual_train_clean_count"]) / true_clean
    out["missed_clean_ratio"] = float(out["missed_clean_count"]) / true_clean
    out["false_clean_ratio"] = float(out["false_clean_count"]) / true_noise
    return out


def format_selection_stats(prefix, st):
    return (
        f"{prefix} total={st['selected_total_count']} train={st['global_train_count']} "
        f"train_ratio={st['train_ratio']:.3f} pred_clean={st['pred_clean_count']} "
        f"pred_noisy={st['pred_noisy_count']} true_clean={st['true_clean_count']} "
        f"true_noise={st['true_noise_count']} actual_train_clean={st['actual_train_clean_count']} "
        f"actual_clean_ratio={st['actual_train_clean_ratio']:.3f} clean_recall={st['clean_recall']:.3f} "
        f"missed_clean={st['missed_clean_count']} false_clean={st['false_clean_count']} "
        f"true_noise_rejected={st['true_noise_rejected_count']} total_wrong={st['total_wrong_count']} "
        f"missed_clean_ratio={st['missed_clean_ratio']:.3f} false_clean_ratio={st['false_clean_ratio']:.3f}"
    )


def write_metrics_header(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,global_train_acc,global_train_loss,global_test_acc,global_test_loss,"
            "personal_apa,personal_loss,clean_ratio,agreement_ratio,"
            "selected_total_count,global_train_count,train_ratio,"
            "pred_clean_count,pred_noisy_count,true_clean_count,true_noise_count,"
            "actual_train_clean_count,actual_train_clean_ratio,clean_recall,"
            "missed_clean_count,false_clean_count,true_noise_rejected_count,total_wrong_count,"
            "missed_clean_ratio,false_clean_ratio\n"
        )


def append_metrics_row(path, epoch, train_loss, global_train_acc, global_train_loss, global_test_acc,
                       global_test_loss, personal_apa, personal_loss, clean_ratio, agree_ratio, st):
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch},{train_loss:.6f},{global_train_acc:.6f},{global_train_loss:.6f},"
            f"{global_test_acc:.6f},{global_test_loss:.6f},{personal_apa:.6f},{personal_loss:.6f},"
            f"{clean_ratio:.6f},{agree_ratio:.6f},"
            f"{st['selected_total_count']},{st['global_train_count']},{st['train_ratio']:.6f},"
            f"{st['pred_clean_count']},{st['pred_noisy_count']},"
            f"{st['true_clean_count']},{st['true_noise_count']},"
            f"{st['actual_train_clean_count']},{st['actual_train_clean_ratio']:.6f},"
            f"{st['clean_recall']:.6f},{st['missed_clean_count']},{st['false_clean_count']},"
            f"{st['true_noise_rejected_count']},{st['total_wrong_count']},"
            f"{st['missed_clean_ratio']:.6f},{st['false_clean_ratio']:.6f}\n"
        )


def build_local_update(args, dataset_train, client_idx, idxs, gaussian_noise, clean_label_mask):
    """兼容新旧 LocalUpdatePFedRN；新版会传 clean_label_mask。"""
    params = inspect.signature(LocalUpdatePFedRN.__init__).parameters
    kwargs = dict(
        args=args,
        user_idx=client_idx,
        dataset=dataset_train,
        idxs=idxs,
        gaussian_noise=gaussian_noise,
    )
    if "clean_label_mask" in params:
        kwargs["clean_label_mask"] = clean_label_mask
    return LocalUpdatePFedRN(**kwargs)


def parse_noise_rates(args):
    if sum(args.noise_group_num) != args.num_users:
        raise ValueError("sum(args.noise_group_num) must equal args.num_users")

    group_noise_rate = list(args.group_noise_rate)
    if len(group_noise_rate) == 1:
        group_noise_rate = group_noise_rate * 2
    if len(group_noise_rate) % 2 != 0:
        raise ValueError("args.group_noise_rate must contain pairs like 0 0.8")

    args.group_noise_rate = [
        (group_noise_rate[i * 2], group_noise_rate[i * 2 + 1])
        for i in range(len(group_noise_rate) // 2)
    ]

    user_noise_type_rates = []
    for num_users_in_group, noise_type, (min_r, max_r) in zip(
        args.noise_group_num, args.noise_type_lst, args.group_noise_rate
    ):
        step = (max_r - min_r) / max(num_users_in_group, 1)
        rates = np.asarray(range(num_users_in_group)) * step + min_r
        user_noise_type_rates += list(zip([noise_type] * num_users_in_group, rates))
    return user_noise_type_rates


def main():
    start_time = time.time()
    args = args_parser()
    args.method = "pfedrn"

    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    if isinstance(args.schedule, str):
        args.schedule = [int(x) for x in args.schedule.split() if str(x).strip()]
    else:
        args.schedule = [int(x) for x in args.schedule]

    # PFedRN 当前版本：只上传/聚合 net1，全局模型；net2 留在客户端本地。
    args.send_2_models = False
    args.num_edges = 1
    args.neighbor_scope = "global"

    # 新增参数兜底，避免 options.py 未加入时出错。
    if not hasattr(args, "pfl_local_weight"):
        args.pfl_local_weight = 0.3
    if not hasattr(args, "pfl_agree_weight"):
        args.pfl_agree_weight = 0.15
    if not hasattr(args, "pfl_personal_ep"):
        args.pfl_personal_ep = 1

    if not torch.cuda.is_available() and args.gpu != -1:
        raise RuntimeError("CUDA is not available. Use --gpu -1 for CPU debugging.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    noise_name_for_file = str(args.group_noise_rate)
    classes_per_user = args.num_shards // args.num_users if args.partition == "shard" else "dirichlet"
    save_dir = os.path.join("resultdate", "PFedRN")
    os.makedirs(save_dir, exist_ok=True)

    log_filename = os.path.join(
        save_dir,
        f"详细模型_PFedRN{args.epochs}_{args.dataset}_{classes_per_user}_{noise_name_for_file}_{timestamp}.txt",
    )
    metrics_log_filename = os.path.join(
        save_dir,
        f"总体模型_PFedRN{args.epochs}_{args.dataset}_{classes_per_user}_{noise_name_for_file}_{timestamp}.txt",
    )

    print("Results will be saved to:", log_filename)
    print("Metrics ONLY will be saved to:", metrics_log_filename)
    print("Device:", args.device)
    print("PFedRN detailed logging enabled.")
    print("LocalUpdatePFedRN supports clean_label_mask:", "clean_label_mask" in inspect.signature(LocalUpdatePFedRN.__init__).parameters)
    for x in vars(args).items():
        print(x)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    labels = get_dataset_labels(dataset_train).copy()
    clean_labels = labels.copy()
    args.img_size = int(dataset_train[0][0].shape[1])

    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == "shard":
        dict_users = sample_noniid_shard(labels=labels, num_users=args.num_users, num_shards=args.num_shards)
    else:
        dict_users = sample_dirichlet(labels=labels, num_users=args.num_users, alpha=args.dd_alpha)

    # 当前为全量本地训练；保留 dict_users_test 接口，Personal APA 默认可能为 0。
    dict_users_train, dict_users_test = {}, {}
    for i in range(args.num_users):
        idxs = list(copy.deepcopy(dict_users[i]))
        rng = np.random.RandomState(args.seed + i)
        rng.shuffle(idxs)
        dict_users_train[i] = idxs
        dict_users_test[i] = []

    user_noise_type_rates = parse_noise_rates(args)

    # 注入标签噪声。
    for user, (noise_type, noise_rate) in enumerate(user_noise_type_rates):
        if noise_type == "clean" or float(noise_rate) <= 0:
            continue
        data_indices = list(copy.deepcopy(dict_users_train[user]))
        random.seed(args.seed + user)
        random.shuffle(data_indices)
        noise_num = int(len(data_indices) * float(noise_rate))
        for d_idx in data_indices[:noise_num]:
            y = int(get_dataset_labels(dataset_train)[d_idx])
            noisy_y = noisify_label(y, num_classes=args.num_classes, noise_type=noise_type)
            set_dataset_label(dataset_train, d_idx, noisy_y)

    noisy_labels = get_dataset_labels(dataset_train)
    clean_label_mask = noisy_labels == clean_labels
    injected_noise_summary = (
        "Injected noise summary: true_clean={} true_noisy={} clean_ratio={:.4f}".format(
            int(clean_label_mask.sum()),
            int((~clean_label_mask).sum()),
            float(clean_label_mask.mean()),
        )
    )
    print(injected_noise_summary)

    log_train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
    log_test_loader = DataLoader(dataset_test, batch_size=args.bs, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    net_glob = get_model(args).to(args.device)
    initial_state = copy.deepcopy(net_glob.state_dict())
    gaussian_noise = torch.randn(1, args.num_channels, args.img_size, args.img_size).to(args.device)

    local_objects = []
    for i in range(args.num_users):
        local = build_local_update(
            args=args,
            dataset_train=dataset_train,
            client_idx=i,
            idxs=dict_users_train[i],
            gaussian_noise=gaussian_noise,
            clean_label_mask=clean_label_mask,
        )
        local.net1.load_state_dict(initial_state)
        local.net2.load_state_dict(initial_state)
        local_objects.append(local)

    if local_objects and not hasattr(local_objects[0], "last_selection_stats"):
        print("WARNING: 当前 LocalUpdatePFedRN 没有 last_selection_stats，说明 update.py 不是详细日志版本。")

    num_edges = max(int(args.num_edges), 1)
    clients_per_edge = args.num_users // num_edges
    all_client_ids = list(range(args.num_users))
    edge_clients_map = {}
    for e in range(num_edges):
        s = e * clients_per_edge
        edge_clients_map[e] = all_client_ids[s:] if e == num_edges - 1 else all_client_ids[s:s + clients_per_edge]

    print("\nStructure: {} edge server(s), neighbor_scope={}".format(num_edges, args.neighbor_scope))
    print("PFedRN: net1/global is aggregated; net2/personalized is kept on each client.\n")

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Experiment Start: {timestamp}\n")
        f.write(f"Args: {args}\n")
        f.write(injected_noise_summary + "\n")
        f.write("=" * 100 + "\n")
    write_metrics_header(metrics_log_filename)

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
        round_selection_stats = []

        for edge_id, current_edge_clients in edge_clients_map.items():
            m = max(int(args.frac * len(current_edge_clients)), 1)
            selected_clients = np.random.choice(current_edge_clients, m, replace=False)

            client_weights, client_samples = [], []
            client_losses, client_clean, client_agree = [], [], []
            client_selection_stats = []

            for client_idx in selected_clients:
                local = local_objects[client_idx]
                local.args = args
                net_global = copy.deepcopy(local.net1).to(args.device)
                net_personal = copy.deepcopy(local.net2).to(args.device)
                sample_size = len(dict_users_train[client_idx])

                if epoch < args.warmup_epochs:
                    w, loss, w_personal, p_loss = local.train_phase1_pfedrn(net_global, net_personal)
                    clean_ratio = 1.0
                    agree_ratio = 0.0
                else:
                    neighbor_pool = all_client_ids if args.neighbor_scope == "global" else current_edge_clients

                    sim_list, exp_list = [], []
                    cosine_sim = torch.nn.CosineSimilarity(dim=1)
                    for u in neighbor_pool:
                        sim = cosine_sim(
                            local.arbitrary_output.view(1, -1).to(args.device),
                            local_objects[u].arbitrary_output.view(1, -1).to(args.device),
                        ).item()
                        sim_list.append(sim)
                        exp_list.append(float(local_objects[u].expertise))

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
                client_losses.append(float(loss))
                client_clean.append(float(clean_ratio))
                client_agree.append(float(agree_ratio))
                client_selection_stats.append(getattr(local, "last_selection_stats", {}))

            if len(client_weights) > 0:
                w_edge = fedavg(client_weights, client_samples, skip_local_head=True)
                edge_weights.append(w_edge)
                edge_samples.append(sum(client_samples))
                edge_stat = aggregate_selection_stats(client_selection_stats)
                edge_logs.append(
                    "  --> [Edge {:02d}] clients={} loss={:.4f} clean={:.3f} agree={:.3f}".format(
                        edge_id + 1,
                        len(client_weights),
                        float(np.mean(client_losses)),
                        float(np.mean(client_clean)),
                        float(np.mean(client_agree)),
                    )
                )
                edge_logs.append(format_selection_stats("      DataStats:", edge_stat))

                round_losses.extend(client_losses)
                round_clean_ratios.extend(client_clean)
                round_agreements.extend(client_agree)
                round_selection_stats.extend(client_selection_stats)

        if len(edge_weights) > 0:
            w_glob = fedavg(edge_weights, edge_samples, skip_local_head=True)
            net_glob.load_state_dict(w_glob, strict=False)
            for local in local_objects:
                sync_global_to_client(local.net1, w_glob, mode="global")
                sync_global_to_client(local.net2, w_glob, mode="personal")

        global_train_acc, global_train_loss = test_img(net_glob, log_train_loader, args)
        global_test_acc, global_test_loss = test_img(net_glob, log_test_loader, args)
        personal_apa, personal_loss = evaluate_personalized_apa(local_objects, dataset_train, dict_users_test, args)

        train_loss = float(np.mean(round_losses)) if round_losses else 0.0
        clean_ratio = float(np.mean(round_clean_ratios)) if round_clean_ratios else 0.0
        agree_ratio = float(np.mean(round_agreements)) if round_agreements else 0.0
        round_stat = aggregate_selection_stats(round_selection_stats)

        log_round = "\n==================== Round {:3d} ====================".format(epoch)
        log_metric = (
            "Global Train Acc: {:.2f}% | Global Train Loss: {:.4f} | "
            "Global Test Acc: {:.2f}% | Global Test Loss: {:.4f} | Personal APA: {:.2f}% | "
            "Personal Loss: {:.4f} | Train Loss: {:.4f} | Clean: {:.3f} | Agree: {:.3f}"
        ).format(
            global_train_acc,
            global_train_loss,
            global_test_acc,
            global_test_loss,
            personal_apa,
            personal_loss,
            train_loss,
            clean_ratio,
            agree_ratio,
        )
        log_data_metric = format_selection_stats("RoundData:", round_stat)

        print(log_round)
        for s in edge_logs:
            print(s)
        print(log_metric)
        print(log_data_metric)

        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(log_round + "\n")
            for s in edge_logs:
                f.write(s + "\n")
            f.write(log_metric + "\n")
            f.write(log_data_metric + "\n")

        append_metrics_row(
            metrics_log_filename,
            epoch,
            train_loss,
            global_train_acc,
            global_train_loss,
            global_test_acc,
            global_test_loss,
            personal_apa,
            personal_loss,
            clean_ratio,
            agree_ratio,
            round_stat,
        )

    print("Total time: {:.1f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
