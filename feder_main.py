# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import sys

# =============================================================================
# Windows 环境兼容性修复 (解决部分机器上的 DLL Load Failed 报错)
# =============================================================================
conda_path = r"C:\Users\25839\.conda\envs\improve-FedRN-main\Library\bin"
if os.path.exists(conda_path):
    os.environ['PATH'] = conda_path + os.pathsep + os.environ['PATH']
else:
    pass

import copy
import numpy as np
import random
import time
import datetime  # 用于生成带时间戳的文件名，防止实验日志相互覆盖

import torchvision
import torch
from torch.utils.data import DataLoader

from utils import load_dataset
from utils.options import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label

from models.nets import get_model
from models.test import test_img
from models.update import get_local_update_objects


# =============================================================================
# 联邦聚合核心函数: FedAvg (支持样本量加权 或 自定义打分加权)
# =============================================================================
def FedAvg(w_list, weight_list):
    """
    加权联邦平均算法
    w_list: 包含多个模型参数字典的列表
    weight_list: 对应模型的权重列表（可以是样本数量，也可以是 FedER 中的交叉打分）
    """
    if not w_list or not weight_list:
        return None

    total_samples = sum(weight_list)
    w_avg = copy.deepcopy(w_list[0])

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight_list[0]
        for i in range(1, len(w_list)):
            w_avg[k] += w_list[i][k] * weight_list[i]
        # 根据总权重进行归一化
        w_avg[k] = torch.div(w_avg[k], total_samples)

    return w_avg


if __name__ == '__main__':
    start = time.time()

    # 1. 解析命令行参数
    args = args_parser()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu',
    )
    args.schedule = [int(x) for x in args.schedule]

    # =============================================================================
    # 🔓 算法路由机制：识别哪些算法需要开启“双模型 (Dual-Model)” 机制
    # feder 属于双核交叉互学流派，因此将其加入白名单
    # =============================================================================
    args.send_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix', 'feder']

    # =============================================================================
    # 自动日志系统：分离“详细过程日志”与“纯指标日志（用于画图）”
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

    # 固定随机数种子，确保多次实验的可复现性 (Reproducibility)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    # FedRN/FedRNN 使用的全局模型指纹噪声输入
    gaussian_noise = torch.randn(1, 3, 32, 32)

    ##############################
    # 数据集加载与 Non-IID 切分
    ##############################
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)

    # 兼容新老版本 PyTorch Dataset 的标签名称属性
    if hasattr(dataset_train, 'targets'):
        labels = np.array(dataset_train.targets)
    else:
        labels = np.array(dataset_train.train_labels)

    img_size = dataset_train[0][0].shape
    args.img_size = int(img_size[1])

    # 根据参数模拟不同的数据异构性 (Data Heterogeneity)
    if args.iid:
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == 'shard':  # 极端偏科划分
        dict_users = sample_noniid_shard(
            labels=labels, num_users=args.num_users, num_shards=args.num_shards)
    elif args.partition == 'dirichlet':  # 狄利克雷分布划分
        dict_users = sample_dirichlet(
            labels=labels, num_users=args.num_users, alpha=args.dd_alpha)

    ##############################
    # 模拟现实环境：注入标签噪声 (Label Noise Injection)
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

    # 将噪声实际应用到数据集的 label 上
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

    logging_args = dict(batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, **logging_args)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, **logging_args)

    ##############################
    # 初始化全局模型 (Global Models)
    ##############################
    net_glob = get_model(args).to(args.device)

    # 🔓 解除封印：如果是双核算法，初始化并挂载第二个全局模型
    net_glob2 = None
    if args.send_2_models:
        print("🟢 双模型(Dual-Model)机制已开启！")
        net_glob2 = get_model(args).to(args.device)

    # 配置传统 Co-teaching 的动态丢弃率 (Forget Rate)
    forget_rate_schedule = []
    if args.method in ['coteaching', 'coteaching+', 'dividemix', 'feder']:
        num_gradual = args.warmup_epochs
        forget_rate = args.forget_rate
        exponent = 1
        forget_rate_schedule = np.ones(args.epochs) * forget_rate
        forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

    pred_user_noise_rates = [args.forget_rate] * args.num_users

    # 调用路由器：根据 args.method 实例化对应的底层本地更新对象 (LocalUpdate Object)
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        noise_rates=pred_user_noise_rates,
        gaussian_noise=gaussian_noise,
    )

    # ========================= [初始化所有客户端模型] =========================
    # 模拟系统刚启动时，云端将初始全局权重同步下发给所有终端设备
    initial_state = copy.deepcopy(net_glob.state_dict())
    if args.send_2_models:
        initial_state2 = copy.deepcopy(net_glob2.state_dict())

    for i in range(args.num_users):
        local_update_objects[i].net1.load_state_dict(initial_state)
        if args.send_2_models:
            local_update_objects[i].net2.load_state_dict(initial_state2)

    # ========================= [配置三层网络拓扑 (Edge Topology Settings)] =========================
    # 将客户端（Client）均匀划分给不同的边缘基站（Edge Server）进行管辖
    NUM_EDGES = args.num_edges
    CLIENTS_PER_EDGE = args.num_users // NUM_EDGES

    edge_clients_map = {}
    all_client_ids = list(range(args.num_users))

    for i in range(NUM_EDGES):
        start_idx = i * CLIENTS_PER_EDGE
        # 处理可能无法整除的余数情况
        if i == NUM_EDGES - 1:
            edge_clients_map[i] = all_client_ids[start_idx:]
        else:
            edge_clients_map[i] = all_client_ids[start_idx: start_idx + CLIENTS_PER_EDGE]

    args.clients_per_edge = args.num_users // args.num_edges
    NUM_CLIENT = args.clients_per_edge
    print(f"\nStructure: {NUM_EDGES} 边缘服务器. 每个边缘服务器有：{NUM_CLIENT} 客户端")
    print("开始分层训练（客户端 - 边缘 - 云端）\n")

    ##############################
    # 三层联邦训练主循环 (Training Loop)
    ##############################
    for epoch in range(args.epochs):
        # 学习率衰减
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        # 更新本轮的基础丢弃率 (即使是 FedER 也需记录，仅作兼容备用)
        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        args.g_epoch = epoch

        # 用于存储各个 Edge 聚合后的模型权重，准备发往 Cloud
        edge_weights_list = []
        edge_weights_list2 = []
        edge_samples_list = []
        edge_losses_list = []
        edge_log_strings = []

        # --- [第一层循环: 遍历每个边缘服务器 (Edge Server)] ---
        for edge_id in range(NUM_EDGES):
            current_edge_clients = edge_clients_map[edge_id]

            # 存储该 Edge 下所有参与训练的 Client 上传的模型权重
            client_weights_list = []
            client_weights_list2 = []
            client_samples_list = []
            client_losses = []
            client_losses2 = []

            # 按照比例随机采样参与本轮更新的设备
            m = max(int(args.frac * len(current_edge_clients)), 1)
            selected_clients = np.random.choice(current_edge_clients, m, replace=False)

            # --- [第二层循环: 客户端本地训练 (Client Local Update)] ---
            for client_idx in selected_clients:
                local = local_update_objects[client_idx]
                local.args = args

                # 模拟 Edge 将模型发给 Client
                net_local = copy.deepcopy(net_glob).to(args.device)
                net_local2 = copy.deepcopy(net_glob2).to(args.device) if args.send_2_models else None
                client_sample_size = len(dict_users[client_idx])

                # ---------------- [路由 1：消融版单机 FedRN] ----------------
                if args.method == "fedrn":
                    if epoch < args.warmup_epochs:
                        w, loss = local.train_phase1(net_local)
                    else:
                        # 只靠自己跑 GMM 清洗，不去边缘基站内找邻居交叉感染
                        w, loss = local.train_phase_self_clean(net_local)

                # ---------------- [路由 2：原汁原味的原作者 FedRN Baseline] ----------------
                elif args.method == "fedrnn":
                    if epoch < args.warmup_epochs:
                        # Warm-up 阶段: 正常训练，并生成用于指纹比对的高斯输出
                        w, loss = local.train_phase1(net_local)
                    else:
                        # 🚀 发力期：核心的“寻找靠谱邻居”逻辑
                        sim_list = []
                        exp_list = []
                        cosine_sim = torch.nn.CosineSimilarity()

                        # 1. 物理隔离：只被允许和同一个边缘服务器（Edge）下的设备计算相似度和专业度
                        for user in current_edge_clients:
                            sim = cosine_sim(
                                local.arbitrary_output.to(args.device),
                                local_update_objects[user].arbitrary_output.to(args.device),
                            ).item()
                            exp = local_update_objects[user].expertise
                            sim_list.append(sim)
                            exp_list.append(exp)

                        # 2. 归一化得分 (Min-Max Normalization)
                        sim_list = [(sim - min(sim_list)) / (max(sim_list) - min(sim_list) + 1e-8) for sim in sim_list]
                        exp_list = [(exp - min(exp_list)) / (max(exp_list) - min(exp_list) + 1e-8) for exp in exp_list]

                        local_idx_in_edge = current_edge_clients.index(client_idx)
                        w_alpha = getattr(args, 'w_alpha', 0.5)
                        prev_score = w_alpha * exp_list[local_idx_in_edge] + (1 - w_alpha)

                        # 计算出本基站下所有候选人的综合得分
                        score_list = []
                        for neighbor_local_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
                            if neighbor_local_idx != local_idx_in_edge:
                                score = w_alpha * exp + (1 - w_alpha) * sim
                                global_neighbor_idx = current_edge_clients[neighbor_local_idx]
                                score_list.append([score, global_neighbor_idx])

                        score_list.sort(key=lambda x: x[0], reverse=True)

                        # 3. 获取 Top-K 靠谱邻居 (如果在偏科严重的环境下，挑出来的邻居也是坏的)
                        neighbor_list = []
                        neighbor_score_list = []
                        num_neighbors = getattr(args, 'num_neighbors', 2)

                        for k in range(min(num_neighbors, len(score_list))):
                            neighbor_score, global_neighbor_idx = score_list[k]
                            # 深拷贝邻居模型，准备进入底层去进行锁头微调
                            neighbor_net = copy.deepcopy(local_update_objects[global_neighbor_idx].net1)
                            neighbor_list.append(neighbor_net)
                            neighbor_score_list.append(neighbor_score)

                        # 4. 调用原版的 train_phase2 进行锁头微调和 GMM 投票清洗
                        w, loss = local.train_phase2(net_local, prev_score, neighbor_list, neighbor_score_list)

                # ---------------- [路由 3：你的 FedER 双核动态互学机制] ----------------
                elif args.send_2_models:
                    # 进入 update.py 中的 LocalUpdateFedER 触发强大的 GMM掩码交叉互喂
                    w, loss, w2, loss2 = local.train(net_local, net_local2)

                # ---------------- [路由 4：默认基础算法 (如 FedAvg)] ----------------
                else:
                    w, loss = local.train(net_local)

                # 将在 CPU 上卸载完毕的权重保存，准备聚合
                w_cpu = {k: v.cpu() for k, v in w.items()}
                client_weights_list.append(w_cpu)
                client_samples_list.append(client_sample_size)
                client_losses.append(loss)

                # 如果有模型 2 的结果，同样保存
                if args.send_2_models:
                    w2_cpu = {k: v.cpu() for k, v in w2.items()}
                    client_weights_list2.append(w2_cpu)
                    client_losses2.append(loss2)

                    # --- [第三层循环: 边缘服务器局部聚合 (Edge Aggregation)] ---
            if len(client_weights_list) > 0:

                # 🚀 论文核心卖点：FedER 专属的服务器端交叉加权聚合！
                # 避免了在客户端本地硬微调带来的交叉感染，将算力上移至 Edge
                if args.method == "feder" and epoch >= args.warmup_epochs:
                    import torch
                    import torch.nn.functional as F

                    # 🌟 1. 评估网络 1：用特征计算相似度，用 Loss 计算专业度
                    last_layer_key = list(client_weights_list[0].keys())[-2]
                    features1 = torch.stack([w[last_layer_key].flatten() for w in client_weights_list])
                    sim_matrix1 = F.cosine_similarity(features1.unsqueeze(1), features1.unsqueeze(0), dim=-1)
                    avg_sim1 = sim_matrix1.mean(dim=0)
                    norm_sim1 = (avg_sim1 - avg_sim1.min()) / (avg_sim1.max() - avg_sim1.min() + 1e-8)

                    losses_tensor1 = torch.tensor(client_losses)
                    norm_exp1 = 1.0 - (losses_tensor1 - losses_tensor1.min()) / (
                                losses_tensor1.max() - losses_tensor1.min() + 1e-8)

                    # 🌟 2. 评估网络 2
                    features2 = torch.stack([w[last_layer_key].flatten() for w in client_weights_list2])
                    sim_matrix2 = F.cosine_similarity(features2.unsqueeze(1), features2.unsqueeze(0), dim=-1)
                    avg_sim2 = sim_matrix2.mean(dim=0)
                    norm_sim2 = (avg_sim2 - avg_sim2.min()) / (avg_sim2.max() - avg_sim2.min() + 1e-8)

                    losses_tensor2 = torch.tensor(client_losses2)
                    norm_exp2 = 1.0 - (losses_tensor2 - losses_tensor2.min()) / (
                                losses_tensor2.max() - losses_tensor2.min() + 1e-8)

                    # 🌟 3. 动态合并得分 (Extreme Non-IID 下可调高 exp_w 降低正交特征带来的误判)
                    exp_w = args.feder_exp_weight
                    sim_w = 1.0 - exp_w

                    score1 = exp_w * norm_exp1 + sim_w * norm_sim1
                    score2 = exp_w * norm_exp2 + sim_w * norm_sim2

                    # 🌟 4. 【核心创新：交叉聚合 Cross-Aggregation】
                    # 利用网络 2 对系统的评估打分，去作为 网络 1 权重的聚合配比，打破单一系统的确认偏差
                    agg_weights_for_net1 = (score2 / score2.sum()).tolist()
                    agg_weights_for_net2 = (score1 / score1.sum()).tolist()

                    w_edge = FedAvg(client_weights_list, agg_weights_for_net1)
                    w_edge2 = FedAvg(client_weights_list2, agg_weights_for_net2)

                else:
                    # ✅ 预热期及其他 Baseline：使用最经典的基于样本量的 FedAvg
                    w_edge = FedAvg(client_weights_list, client_samples_list)
                    if args.send_2_models:
                        w_edge2 = FedAvg(client_weights_list2, client_samples_list)

                # 保存当前 Edge 的成果
                edge_weights_list.append(w_edge)
                if args.send_2_models:
                    edge_weights_list2.append(w_edge2)

                edge_samples_list.append(sum(client_samples_list))
                avg_edge_loss = sum(client_losses) / len(client_losses)
                edge_losses_list.append(avg_edge_loss)

                # 验证当前 Edge 聚合后的中间态性能 (用于分析各孤岛的坍缩情况)
                net_edge = copy.deepcopy(net_glob).to(args.device)
                net_edge.load_state_dict(w_edge)
                edge_test_acc, edge_test_loss = test_img(net_edge, log_test_data_loader, args)

                edge_str = f"  --> [Edge Server {edge_id + 1}] Test Acc: {edge_test_acc:.2f}% | Test Loss: {edge_test_loss:.6f}"
                edge_log_strings.append(edge_str)

        # --- [第四层循环: 云端服务器全局聚合 (Cloud Aggregation)] ---
        if len(edge_weights_list) > 0:
            w_glob = FedAvg(edge_weights_list, edge_samples_list)
            net_glob.load_state_dict(w_glob)

            if args.send_2_models:
                w_glob2 = FedAvg(edge_weights_list2, edge_samples_list)
                net_glob2.load_state_dict(w_glob2)

            # [同步下发] 将云端融合了所有 Edge 知识的最终模型，下发更新每个客户端本地对象中的缓存
            # 必须执行此步骤，否则下一轮客户端跑 fit_gmm(self.net1) 时用的就是旧知识了
            for i in range(args.num_users):
                local_update_objects[i].net1.load_state_dict(w_glob)
                if args.send_2_models:
                    local_update_objects[i].net2.load_state_dict(w_glob2)

        # ========================= [云端测试与多级日志输出] =========================
        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)

        results = dict(train_acc=train_acc, train_loss=train_loss,
                       test_acc=test_acc, test_loss=test_loss)

        log_str_round = 'Round {:3d}'.format(epoch)
        log_str_metrics = ' - '.join([f'{k}: {v:.6f}' for k, v in results.items()])

        # 1. 终端打印
        print(log_str_round)
        for edge_str in edge_log_strings:
            print(edge_str)
        print(log_str_metrics)

        # 2. 写入详细日志 (包含各 Edge 的微观表现)
        with open(log_filename, 'a') as f:
            f.write(log_str_round + '\n')
            for edge_str in edge_log_strings:
                f.write(edge_str + '\n')
            f.write(log_str_metrics + '\n')
            f.write('-' * 30 + '\n')

        # 3. 写入提纯指标日志 (用于 Origin/Matplotlib 画折线图)
        with open(metrics_log_filename, 'a') as mf:
            mf.write(log_str_round + '\n')
            mf.write(log_str_metrics + '\n')