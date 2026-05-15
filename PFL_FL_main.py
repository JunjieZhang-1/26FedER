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

import copy
import torch


def FedAvg_PFL(w_list, weight_list):
    """
    🌟 PFL 双头专属聚合函数：表征共享，全局头共享，决策私有
    """
    if not w_list or not weight_list:
        return None

    w_avg = copy.deepcopy(w_list[0])
    total_weight = sum(weight_list)

    for k in w_avg.keys():
        # ==========================================
        # 🌟 核心修改：只拦截 'fc_local'！
        # 这样 Backbone 和 'fc_global' 都会顺利通过并进行加权平均
        # ==========================================
        if 'fc_local' in k:
            continue

        w_avg[k] = w_avg[k] * weight_list[0]
        for i in range(1, len(w_list)):
            w_avg[k] += w_list[i][k] * weight_list[i]

        w_avg[k] = torch.div(w_avg[k], total_weight)

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
    args.send_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix','feder','fedco']

    # =============================================================================
    # 日志文件设置
    # =============================================================================
    classes_per_user = args.num_shards // args.num_users if args.partition == 'shard' else "IID"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("resultdate","PFL")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    file_name = f"详细模型_PFL{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{args.method}_{timestamp}.txt"
    log_filename = os.path.join(save_dir, file_name)
    metrics_file_name = f"总体模型_PFL{args.epochs}_{args.dataset}_{classes_per_user}_{args.group_noise_rate}_{args.method}_{timestamp}.txt"
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


    # === 这个切分逻辑保留在这里 ===
    dict_users_train = {}
    dict_users_test = {}
    for i in range(args.num_users):
        idxs = list(copy.deepcopy(dict_users[i]))
        np.random.seed(args.seed)
        np.random.shuffle(idxs)

        split_point = int(len(idxs) * 0.8)
        dict_users_train[i] = idxs[:split_point]
        dict_users_test[i] = idxs[split_point:]

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
            # data_indices = list(copy.deepcopy(dict_users[user]))
            #只对训练级加噪声
            data_indices = list(copy.deepcopy(dict_users_train[user]))
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
    if args.method in ['coteaching', 'coteaching+', 'dividemix','fedco','feder']:
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
        # dict_users=dict_users,#之前的
        dict_users=dict_users_train,
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
        # =========================================================
        # 🌟 [就是加在这里！]：准备在这一轮收集全局的精确 APA
        global_total_acc = 0.0
        global_total_loss = 0.0
        global_valid_clients = 0
        # =========================================================
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
            client_losses2 = []  # 🚀 新增：用于收集模型 2 的 Loss

            m = max(int(args.frac * len(current_edge_clients)), 1)
            selected_clients = np.random.choice(current_edge_clients, m, replace=False)

            # --- 第二层循环: Edge 下发模型，Client 本地更新 ---
            for client_idx in selected_clients:
                local = local_update_objects[client_idx]
                local.args = args
                # 🌟 [修复 1：提取本地专属模型，而不是去拿全局空壳]
                net_local = copy.deepcopy(local.net1).to(args.device)
                net_local2 = copy.deepcopy(local.net2).to(args.device) if args.send_2_models else None

                # net_local = copy.deepcopy(net_glob).to(args.device)
                # # 🔓 如果开启双模型，一并下发第二个网络
                # net_local2 = copy.deepcopy(net_glob2).to(args.device) if args.send_2_models else None

                client_sample_size = len(dict_users[client_idx])

                if args.method == "fedrn":
                    if epoch < args.warmup_epochs:
                        w, loss = local.train_phase1(net_local)
                    else:
                        w, loss = local.train_phase_self_clean(net_local)

                elif args.method == "fedrnn":
                    if epoch < args.warmup_epochs:
                        # Warm-up 阶段: 正常训练，并生成输出指纹和专业度
                        w, loss = local.train_phase1(net_local)
                    else:
                        # 🚀 发力期：原版 FedRN 寻找靠谱邻居的核心逻辑
                        sim_list = []
                        exp_list = []
                        cosine_sim = torch.nn.CosineSimilarity()

                        # 1. 计算与同一边缘服务器下其他客户端的相似度和专业度
                        for user in current_edge_clients:
                            sim = cosine_sim(
                                local.arbitrary_output.to(args.device),
                                local_update_objects[user].arbitrary_output.to(args.device),
                            ).item()
                            exp = local_update_objects[user].expertise
                            sim_list.append(sim)
                            exp_list.append(exp)

                        # 2. 归一化 (Min-Max)
                        sim_list = [(sim - min(sim_list)) / (max(sim_list) - min(sim_list) + 1e-8) for sim in sim_list]
                        exp_list = [(exp - min(exp_list)) / (max(exp_list) - min(exp_list) + 1e-8) for exp in exp_list]

                        # 定位当前客户端在列表中的索引
                        local_idx_in_edge = current_edge_clients.index(client_idx)

                        # 为了防止找不到参数，设置默认值 w_alpha=0.5
                        w_alpha = getattr(args, 'w_alpha', 0.5)
                        prev_score = w_alpha * exp_list[local_idx_in_edge] + (1 - w_alpha)

                        score_list = []
                        for neighbor_local_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
                            if neighbor_local_idx != local_idx_in_edge:
                                score = w_alpha * exp + (1 - w_alpha) * sim
                                global_neighbor_idx = current_edge_clients[neighbor_local_idx]
                                score_list.append([score, global_neighbor_idx])

                        score_list.sort(key=lambda x: x[0], reverse=True)

                        # 3. 获取 Top-K 靠谱邻居 (默认取 2 个)
                        neighbor_list = []
                        neighbor_score_list = []
                        num_neighbors = getattr(args, 'num_neighbors', 2)

                        for k in range(min(num_neighbors, len(score_list))):
                            neighbor_score, global_neighbor_idx = score_list[k]
                            # 深拷贝邻居模型，防止微调时污染别人
                            neighbor_net = copy.deepcopy(local_update_objects[global_neighbor_idx].net1)
                            neighbor_list.append(neighbor_net)
                            neighbor_score_list.append(neighbor_score)

                        # 4. 调用原版的 train_phase2 进行锁头微调和 GMM 投票清洗
                        w, loss = local.train_phase2(net_local, prev_score, neighbor_list, neighbor_score_list)
                # 👇 🚀 增加全新方法 FedCO 的分支
                elif args.method == "fedco":
                    if epoch < args.warmup_epochs:
                        # 预热期：双模型普通训练，不丢数据，同时生成指纹
                        w, loss, w2, loss2 = local.train_phase1_dual(net_local, net_local2)
                    else:
                        # 发力期：寻找邻居共识
                        sim_list, exp_list = [], []
                        cosine_sim = torch.nn.CosineSimilarity()
                        for user in current_edge_clients:
                            sim = cosine_sim(local.arbitrary_output.to(args.device), local_update_objects[user].arbitrary_output.to(args.device)).item()
                            exp = local_update_objects[user].expertise
                            sim_list.append(sim)
                            exp_list.append(exp)

                        sim_list = [(s - min(sim_list)) / (max(sim_list) - min(sim_list) + 1e-8) for s in sim_list]
                        exp_list = [(e - min(exp_list)) / (max(exp_list) - min(exp_list) + 1e-8) for e in exp_list]

                        local_idx_in_edge = current_edge_clients.index(client_idx)
                        w_alpha = getattr(args, 'w_alpha', 0.5)
                        prev_score = w_alpha * exp_list[local_idx_in_edge] + (1 - w_alpha)

                        score_list = []
                        for n_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
                            if n_idx != local_idx_in_edge:
                                score_list.append([w_alpha * exp + (1 - w_alpha) * sim, current_edge_clients[n_idx]])
                        score_list.sort(key=lambda x: x[0], reverse=True)

                        neighbor_list, neighbor_score_list = [], []
                        num_neighbors = getattr(args, 'num_neighbors', 2)
                        for k in range(min(num_neighbors, len(score_list))):
                            n_score, n_global_idx = score_list[k]
                            # 这里只需深拷贝网络1的参数即可，作为邻居的底座
                            neighbor_list.append(copy.deepcopy(local_update_objects[n_global_idx].net1))
                            neighbor_score_list.append(n_score)

                        # 调用专属的双模型第二阶段训练！
                        w, loss, w2, loss2 = local.train_phase2_dual(net_local, net_local2, prev_score, neighbor_list, neighbor_score_list)
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

                # 🌟 [修复 2：立刻保存本地训练成果，死死保住个性化头！]
                local_update_objects[client_idx].net1.load_state_dict(w_cpu)
                # 🔓 收集模型2权重
                if args.send_2_models:
                    w2_cpu = {k: v.cpu() for k, v in w2.items()}
                    client_weights_list2.append(w2_cpu)
                    client_losses2.append(loss2)  # 🚀 新增：保存模型 2 的 Loss




            # --- Edge Aggregation (边缘动态防御聚合) ---
            if len(client_weights_list) > 0:
                # 🌟 只有在发力期且为 fedco 时，才开启极其强悍的边缘防毒机制
                if args.method == "fedco" and epoch >= args.warmup_epochs:
                    import torch.nn.functional as F

                    # 1. 计算模型 1 的专业性 (exp) 与相似度 (sim)
                    # 提取倒数第二层 (通常是展平后的特征层) 来计算客户端之间的相似度
                    last_layer_key = list(client_weights_list[0].keys())[-2]
                    features1 = torch.stack([w[last_layer_key].flatten() for w in client_weights_list])
                    sim_matrix1 = F.cosine_similarity(features1.unsqueeze(1), features1.unsqueeze(0), dim=-1)
                    avg_sim1 = sim_matrix1.mean(dim=0)
                    norm_sim1 = (avg_sim1 - avg_sim1.min()) / (avg_sim1.max() - avg_sim1.min() + 1e-8)

                    losses_tensor1 = torch.tensor(client_losses)
                    # 损失越小，专业性越高 (用 1 减去归一化后的 loss)
                    norm_exp1 = 1.0 - (losses_tensor1 - losses_tensor1.min()) / (losses_tensor1.max() - losses_tensor1.min() + 1e-8)

                    # 2. 计算模型 2 的专业性 (exp) 与相似度 (sim)
                    features2 = torch.stack([w[last_layer_key].flatten() for w in client_weights_list2])
                    sim_matrix2 = F.cosine_similarity(features2.unsqueeze(1), features2.unsqueeze(0), dim=-1)
                    avg_sim2 = sim_matrix2.mean(dim=0)
                    norm_sim2 = (avg_sim2 - avg_sim2.min()) / (avg_sim2.max() - avg_sim2.min() + 1e-8)

                    losses_tensor2 = torch.tensor(client_losses2)
                    norm_exp2 = 1.0 - (losses_tensor2 - losses_tensor2.min()) / (losses_tensor2.max() - losses_tensor2.min() + 1e-8)

                    # 3. 综合质量打分 (默认 exp 占 0.6，sim 占 0.4)
                    exp_w = getattr(args, 'feder_exp_weight', 0.6)
                    sim_w = 1.0 - exp_w

                    score1 = exp_w * norm_exp1 + sim_w * norm_sim1
                    score2 = exp_w * norm_exp2 + sim_w * norm_sim2

                    # 🌟 4. 【核心创新：动态一票否决 + 互相微调交叉聚合】
                    mean_score1 = score1.mean()
                    mean_score2 = score2.mean()

                    # 踢出得分过低的毒化节点 (直接将其权重设为 0，不再参与 Edge 聚合)
                    score1[score1 < mean_score1 * 0.5] = 0.0
                    score2[score2 < mean_score2 * 0.5] = 0.0

                    # 兜底：如果被全票否决了，恢复成按样本量
                    agg_weights_for_net2 = client_samples_list if score1.sum() == 0 else (score1 / score1.sum()).tolist()
                    agg_weights_for_net1 = client_samples_list if score2.sum() == 0 else (score2 / score2.sum()).tolist()

                    # 💥 交叉指导聚合：用网络 2 评出的干净权重去聚合网络 1，反之亦然！
                    w_edge = FedAvg_PFL(client_weights_list, agg_weights_for_net1)
                    w_edge2 = FedAvg_PFL(client_weights_list2, agg_weights_for_net2)

                else:
                    #  预热期或 Baseline：老老实实按样本量做常规 PFL 聚合
                    w_edge = FedAvg_PFL(client_weights_list, client_samples_list)
                    if args.send_2_models:
                        w_edge2 = FedAvg_PFL(client_weights_list2, client_samples_list)


                edge_weights_list.append(w_edge)
                if args.send_2_models:
                    edge_weights_list2.append(w_edge2)

                # 记录样本数和损失
                edge_samples_list.append(sum(client_samples_list))
                avg_edge_loss = sum(client_losses) / len(client_losses)
                edge_losses_list.append(avg_edge_loss)



                # ==========================================================
                # 🌟 专业 PFL 评估：计算当前 Edge 管辖的 20 个客户端的局部 APA
                # ==========================================================
                from utils.dataset import DatasetSplit

                edge_total_acc = 0.0
                edge_total_loss = 0.0  # 新增：记录 local_loss
                edge_valid_clients = 0

                for c_idx in current_edge_clients:
                    if len(dict_users_test[c_idx]) > 0:
                        # 拿出刚训练完的完美配合的个性化模型
                        c_net = local_update_objects[c_idx].net1
                        c_loader = DataLoader(
                            DatasetSplit(dataset_train, dict_users_test[c_idx]),
                            batch_size=args.local_bs,
                            shuffle=False
                        )
                        # 测试出精确分数
                        c_acc, c_loss = test_img(c_net, c_loader, args)

                        edge_total_acc += c_acc
                        edge_total_loss += c_loss
                        edge_valid_clients += 1

                # 🌟 [关键]：累加到全局记分牌
                global_total_acc += edge_total_acc
                global_total_loss += edge_total_loss
                global_valid_clients += edge_valid_clients

                edge_apa = edge_total_acc / edge_valid_clients if edge_valid_clients > 0 else 0
                edge_avg_loss = edge_total_loss / edge_valid_clients if edge_valid_clients > 0 else 0

                edge_str = f"  --> [Edge Server {edge_id + 1}] Local APA: {edge_apa:.2f}% | Local Loss: {edge_avg_loss:.4f}"
                edge_log_strings.append(edge_str)





        # --- Cloud Aggregation (云端聚合) ---
        if len(edge_weights_list) > 0:
            w_glob = FedAvg_PFL(edge_weights_list, edge_samples_list)
            net_glob.load_state_dict(w_glob)

            # 🔓 模型2 在云端进行聚合
            if args.send_2_models:
                w_glob2 = FedAvg_PFL(edge_weights_list2, edge_samples_list)
                net_glob2.load_state_dict(w_glob2)

            # ==========================================================
            # 🌟 [修复 3：精准同步！双头架构专属]
            # 云端下发 Backbone 和 全局头 (fc_global)，坚决保留本地头 (fc_local)
            # ==========================================================
            for i in range(args.num_users):
                local_state1 = local_update_objects[i].net1.state_dict()
                for k in w_glob.keys():
                    # 💡 核心改变：以前是拦截所有 fc 和 linear，现在只拦截 fc_local
                    if 'fc_local' not in k:
                        local_state1[k] = w_glob[k]
                local_update_objects[i].net1.load_state_dict(local_state1)

                if args.send_2_models:
                    local_state2 = local_update_objects[i].net2.state_dict()
                    for k in w_glob2.keys():
                        # 同理，只拦截 fc_local
                        if 'fc_local' not in k:
                            local_state2[k] = w_glob2[k]
                    local_update_objects[i].net2.load_state_dict(local_state2)



        # ========================= [PFL 专属 APA 测试与日志输出] =========================

        # 🌟 直接用刚才在 Edge 循环里累加好的数据，避免测试“缝合怪”模型，节省 50% 时间！
        test_acc = global_total_acc / global_valid_clients if global_valid_clients > 0 else 0
        test_loss = global_total_loss / global_valid_clients if global_valid_clients > 0 else 0
        train_loss = sum(edge_losses_list) / len(edge_losses_list) if edge_losses_list else 0

        results = dict(train_loss=train_loss, test_acc=test_acc, test_loss=test_loss)

        log_str_round = '\n' + '='*20 + f' Round {epoch:3d} ' + '='*20
        log_str_metrics = f"Overall Metrics -> Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Global APA: {test_acc:.2f}%"

        # 1. 干净利落地打印到控制台
        print(log_str_round)
        for edge_str in edge_log_strings:
            print(edge_str)
        print("-" * 52)
        print(log_str_metrics)

        # 2. 写入详细日志 txt 文件
        with open(log_filename, 'a') as f:
            f.write(log_str_round + '\n')
            for edge_str in edge_log_strings:
                f.write(edge_str + '\n')
            f.write("-" * 52 + '\n')
            f.write(log_str_metrics + '\n')

        # 3. 写入精简日志 txt 文件 (方便以后画折线图)
        with open(metrics_log_filename, 'a') as mf:
            mf.write(f"{epoch}, {train_loss:.6f}, {test_loss:.6f}, {test_acc:.6f}\n")


