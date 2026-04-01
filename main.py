# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import sys

# =============================================================================
# Windows 环境兼容性修复 (DLL Load Failed Fix)
# =============================================================================
conda_path = r"C:\Users\25839\.conda\envs\improve-FedRN-main\Library\bin"
if os.path.exists(conda_path):
    os.environ['PATH'] = conda_path + os.pathsep + os.environ['PATH']
else:
    # 仅作为提示，不阻断运行
    pass

import copy
import numpy as np
import random
import time
import datetime  # 用于生成带时间戳的文件名

import torchvision
import torch
from torch.utils.data import DataLoader

from utils import load_dataset
from utils.options import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label

from models.fed import LocalModelWeights
from models.nets import get_model
from models.test import test_img
from models.update import get_local_update_objects


# =============================================================================
# 修复: FedAvg 加权平均函数 (用于 Edge 聚合 和 Cloud 聚合)
# =============================================================================
def FedAvg(w_list, weight_list):
    """
    加权联邦平均算法
    """
    if not w_list or not weight_list:
        return None

    total_samples = sum(weight_list)
    w_avg = copy.deepcopy(w_list[0])

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight_list[0]
        for i in range(1, len(w_list)):
            w_avg[k] += w_list[i][k] * weight_list[i]
        w_avg[k] = torch.div(w_avg[k], total_samples)

    return w_avg


if __name__ == '__main__':
    start = time.time()
    # parse args
    args = args_parser()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu',
    )
    args.schedule = [int(x) for x in args.schedule]

    # =============================================================================
    # 🔓 解除封印：允许开启双模型
    # =============================================================================
    args.send_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix','feder']

    # =============================================================================
    # 日志文件设置
    # =============================================================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = "resultdate"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = f"详细模型{args.epochs}_{args.dataset}_{args.group_noise_rate}_{args.method}_{timestamp}.txt"
    log_filename = os.path.join(save_dir, file_name)
    metrics_file_name = f"总体模型{args.epochs}_{args.dataset}_{args.group_noise_rate}_{args.method}_{timestamp}.txt"
    metrics_log_filename = os.path.join(save_dir, metrics_file_name)

    print(f"Results will be saved to: {log_filename}")
    print(f"Metrics ONLY will be saved to: {metrics_log_filename}")

    with open(log_filename, 'w') as f:
        f.write(f"Experiment Start: {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Args: {args}\n")
        f.write("=" * 50 + "\n")

    for x in vars(args).items():
        print(x)

    if not torch.cuda.is_available():
        exit('ERROR: Cuda is not available!')
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    gaussian_noise = torch.randn(1, 3, 32, 32)

    ##############################
    # Load dataset and split users
    ##############################
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)

    # 兼容新老版本的标签名称
    if hasattr(dataset_train, 'targets'):
        labels = np.array(dataset_train.targets)
    else:
        labels = np.array(dataset_train.train_labels)

    img_size = dataset_train[0][0].shape  # used to get model
    args.img_size = int(img_size[1])

    # Sample users (iid / non-iid)
    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == 'shard':
        dict_users = sample_noniid_shard(
            labels=labels,
            num_users=args.num_users,
            num_shards=args.num_shards,
        )
    elif args.partition == 'dirichlet':
        dict_users = sample_dirichlet(
            labels=labels,
            num_users=args.num_users,
            alpha=args.dd_alpha,
        )

    ##############################
    # Add label noise to data
    ##############################
    if sum(args.noise_group_num) != args.num_users:
        exit('Error: sum of the number of noise group have to be equal the number of users')

    if len(args.group_noise_rate) == 1:
        args.group_noise_rate = args.group_noise_rate * 2

    if not len(args.noise_group_num) == len(args.group_noise_rate) and \
            len(args.group_noise_rate) * 2 == len(args.noise_type_lst):
        exit('Error: The noise input is invalid.')

    args.group_noise_rate = [(args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
                             for i in range(len(args.group_noise_rate) // 2)]

    user_noise_type_rates = []
    for num_users_in_group, noise_type, (min_group_noise_rate, max_group_noise_rate) in zip(
            args.noise_group_num, args.noise_type_lst, args.group_noise_rate):
        noise_types = [noise_type] * num_users_in_group

        step = (max_group_noise_rate - min_group_noise_rate) / num_users_in_group
        noise_rates = np.array(range(num_users_in_group)) * step + min_group_noise_rate

        user_noise_type_rates += [*zip(noise_types, noise_rates)]

    for user, (user_noise_type, user_noise_rate) in enumerate(user_noise_type_rates):
        if user_noise_type != "clean":
            data_indices = list(copy.deepcopy(dict_users[user]))
            random.seed(args.seed)
            random.shuffle(data_indices)
            noise_index = int(len(data_indices) * user_noise_rate)
            for d_idx in data_indices[:noise_index]:
                if hasattr(dataset_train, 'targets'):
                    true_label = dataset_train.targets[d_idx]
                    noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
                    dataset_train.targets[d_idx] = noisy_label
                else:
                    true_label = dataset_train.train_labels[d_idx]
                    noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
                    dataset_train.train_labels[d_idx] = noisy_label

    # Logging loaders
    logging_args = dict(batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, **logging_args)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, **logging_args)

    ##############################
    # Build model & Init
    ##############################
    net_glob = get_model(args)
    net_glob = net_glob.to(args.device)

    # 🔓 解除封印：初始化第二个全局模型
    net_glob2 = None
    if args.send_2_models:
        print("🟢 双模型(Dual-Model)机制已开启！")
        net_glob2 = get_model(args)
        net_glob2 = net_glob2.to(args.device)

    forget_rate_schedule = []
    if args.method in ['coteaching', 'coteaching+', 'dividemix','feder']:
        num_gradual = args.warmup_epochs
        forget_rate = args.forget_rate
        exponent = 1
        forget_rate_schedule = np.ones(args.epochs) * forget_rate
        forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

    pred_user_noise_rates = [args.forget_rate] * args.num_users

    # Initialize local update objects
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        noise_rates=pred_user_noise_rates,
        gaussian_noise=gaussian_noise,
    )

    # ========================= [初始化所有客户端模型] =========================
    initial_state = copy.deepcopy(net_glob.state_dict())
    if args.send_2_models:
        initial_state2 = copy.deepcopy(net_glob2.state_dict())

    for i in range(args.num_users):
        local_update_objects[i].net1.load_state_dict(initial_state)
        # 🔓 赋予客户端第二套模型权重
        if args.send_2_models:
            local_update_objects[i].net2.load_state_dict(initial_state2)

    # ========================= [配置三层架构 (Edge Settings)] =========================
    NUM_EDGES = args.num_edges
    CLIENTS_PER_EDGE = args.num_users // NUM_EDGES

    edge_clients_map = {}
    all_client_ids = list(range(args.num_users))

    for i in range(NUM_EDGES):
        start_idx = i * CLIENTS_PER_EDGE
        if i == NUM_EDGES - 1:
            edge_clients_map[i] = all_client_ids[start_idx:]
        else:
            edge_clients_map[i] = all_client_ids[start_idx: start_idx + CLIENTS_PER_EDGE]

    args.clients_per_edge = args.num_users // args.num_edges
    NUM_CLIENT = args.clients_per_edge
    print(f"\nStructure: {NUM_EDGES} 边缘服务器. 每个边缘服务器有：{NUM_CLIENT} 客户端")
    print("开始分层训练（客户端 - 边缘 - 云端）\n")

    ##############################
    # Training Loop (Client-Edge-Cloud)
    ##############################
    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        args.g_epoch = epoch

        # 存储每个 Edge 聚合后的模型
        edge_weights_list = []
        edge_weights_list2 = []  # 🔓 模型2的Edge列表
        edge_samples_list = []
        edge_losses_list = []
        edge_log_strings = []

        # --- 第一层循环: 遍历每个 Edge Server ---
        for edge_id in range(NUM_EDGES):
            current_edge_clients = edge_clients_map[edge_id]

            client_weights_list = []
            client_weights_list2 = []  # 🔓 模型2的Client列表
            client_samples_list = []
            client_losses = []

            m = max(int(args.frac * len(current_edge_clients)), 1)
            selected_clients = np.random.choice(current_edge_clients, m, replace=False)

            # --- 第二层循环: Edge 下发模型，Client 本地更新 ---
            for client_idx in selected_clients:
                local = local_update_objects[client_idx]
                local.args = args

                net_local = copy.deepcopy(net_glob).to(args.device)
                # 🔓 如果开启双模型，一并下发第二个网络
                net_local2 = copy.deepcopy(net_glob2).to(args.device) if args.send_2_models else None

                client_sample_size = len(dict_users[client_idx])

                if args.method == "fedrn":
                    if epoch < args.warmup_epochs:
                        w, loss = local.train_phase1(net_local)
                    else:
                        w, loss = local.train_phase_self_clean(net_local)
                # 🔓 恢复双模型本地训练调用
                elif args.send_2_models:
                    w, loss, w2, loss2 = local.train(net_local, net_local2)
                else:
                    w, loss = local.train(net_local)

                # 收集模型1权重
                w_cpu = {k: v.cpu() for k, v in w.items()}
                client_weights_list.append(w_cpu)
                client_samples_list.append(client_sample_size)
                client_losses.append(loss)

                # 🔓 收集模型2权重
                if args.send_2_models:
                    w2_cpu = {k: v.cpu() for k, v in w2.items()}
                    client_weights_list2.append(w2_cpu)

            # --- Edge Aggregation (边缘聚合) ---
            if len(client_weights_list) > 0:
                if args.method == "feder" and epoch >= args.warmup_epochs:
                    # TODO: 在这里植入 FedER 独有的可靠邻居聚合机制！
                    # 1. 计算 client_weights_list 中各个模型的相似度矩阵
                    # 2. 识别出可靠邻居 (Reliable Neighbors)
                    # 3. 动态调整聚合权重 w_alpha
                    w_edge = FedAvg(client_weights_list, client_samples_list)  # 目前先用 FedAvg 顶替
                else:
                    # ✅ 已修复：恢复使用网络 1 的权重列表
                    w_edge = FedAvg(client_weights_list, client_samples_list)
                edge_weights_list.append(w_edge)

                # 🔓 模型2 在边缘层进行聚合
                if args.send_2_models:
                    if args.method == "feder" and epoch >= args.warmup_epochs:
                        # TODO: 模型2的 FedER 聚合
                        w_edge2 = FedAvg(client_weights_list2, client_samples_list)
                    else:
                        w_edge2 = FedAvg(client_weights_list2, client_samples_list)
                    edge_weights_list2.append(w_edge2)

                # 记录样本数和损失
                edge_samples_list.append(sum(client_samples_list))
                avg_edge_loss = sum(client_losses) / len(client_losses)
                edge_losses_list.append(avg_edge_loss)

                # 评估模型1 (保持只看模型1的成绩即可)
                net_edge = copy.deepcopy(net_glob).to(args.device)
                net_edge.load_state_dict(w_edge)
                edge_test_acc, edge_test_loss = test_img(net_edge, log_test_data_loader, args)

                edge_str = f"  --> [Edge Server {edge_id}] Test Acc: {edge_test_acc:.2f}% | Test Loss: {edge_test_loss:.6f}"
                edge_log_strings.append(edge_str)


            # # --- Edge Aggregation (边缘聚合) ---
            # if len(client_weights_list) > 0:
            #     if args.method == "feder" and epoch >= args.warmup_epochs:
            #         # TODO: 在这里植入 FedER 独有的可靠邻居聚合机制！
            #         # 1. 计算 client_weights_list 中各个模型的相似度矩阵
            #         # 2. 识别出可靠邻居 (Reliable Neighbors)
            #         # 3. 动态调整聚合权重 w_alpha
            #         w_edge = FedAvg(client_weights_list, client_samples_list)  # 目前先用 FedAvg 顶替
            #     else:
            #         w_edge = FedAvg(client_weights_list2, client_samples_list)
            #     edge_weights_list.append(w_edge)
            #
            #     # 🔓 模型2 在边缘层进行聚合
            #     if args.send_2_models:
            #         if args.method == "feder" and epoch >= args.warmup_epochs:
            #             # TODO: 模型2的 FedER 聚合
            #             w_edge2 = FedAvg(client_weights_list2, client_samples_list)
            #         else:
            #             w_edge2 = FedAvg(client_weights_list2, client_samples_list)
            #         edge_weights_list2.append(w_edge2)
            #     edge_samples_list.append(sum(client_samples_list))
            #     avg_edge_loss = sum(client_losses) / len(client_losses)
            #     edge_losses_list.append(avg_edge_loss)
            #
            #     # 评估模型1 (保持只看模型1的成绩即可)
            #     net_edge = copy.deepcopy(net_glob).to(args.device)
            #     net_edge.load_state_dict(w_edge)
            #     edge_test_acc, edge_test_loss = test_img(net_edge, log_test_data_loader, args)
            #
            #     edge_str = f"  --> [Edge Server {edge_id}] Test Acc: {edge_test_acc:.2f}% | Test Loss: {edge_test_loss:.6f}"
            #     edge_log_strings.append(edge_str)

        # --- Cloud Aggregation (云端聚合) ---
        if len(edge_weights_list) > 0:
            w_glob = FedAvg(edge_weights_list, edge_samples_list)
            net_glob.load_state_dict(w_glob)

            # 🔓 模型2 在云端进行聚合
            if args.send_2_models:
                w_glob2 = FedAvg(edge_weights_list2, edge_samples_list)
                net_glob2.load_state_dict(w_glob2)

            # 同步最新模型回本地
            for i in range(args.num_users):
                local_update_objects[i].net1.load_state_dict(w_glob)
                if args.send_2_models:
                    local_update_objects[i].net2.load_state_dict(w_glob2)

        # ========================= [测试与日志输出] =========================
        # 日志以模型 1 的性能为准进行打印（因为两个网络通常性能持平）
        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)

        results = dict(train_acc=train_acc, train_loss=train_loss,
                       test_acc=test_acc, test_loss=test_loss)

        log_str_round = 'Round {:3d}'.format(epoch)
        log_str_metrics = ' - '.join([f'{k}: {v:.6f}' for k, v in results.items()])

        print(log_str_round)
        for edge_str in edge_log_strings:
            print(edge_str)
        print(log_str_metrics)

        with open(log_filename, 'a') as f:
            f.write(log_str_round + '\n')
            for edge_str in edge_log_strings:
                f.write(edge_str + '\n')
            f.write(log_str_metrics + '\n')
            f.write('-' * 30 + '\n')

        with open(metrics_log_filename, 'a') as mf:
            mf.write(log_str_round + '\n')
            mf.write(log_str_metrics + '\n')