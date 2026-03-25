# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Python version: 3.6
# import os
# import sys
#
# # =============================================================================
# # Windows 环境兼容性修复 (DLL Load Failed Fix)
# # =============================================================================
# conda_path = r"C:\Users\25839\.conda\envs\improve-FedRN-main\Library\bin"
# if os.path.exists(conda_path):
#     os.environ['PATH'] = conda_path + os.pathsep + os.environ['PATH']
# else:
#     # 仅作为提示，不阻断运行
#     pass
#
# import copy
# import numpy as np
# import random
# import time
# import datetime  # 用于生成带时间戳的文件名
#
# import torchvision
# import torch
# from torch.utils.data import DataLoader
#
# from utils import load_dataset
# from utils.options import args_parser
# from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
# from utils.utils import noisify_label
#
# from models.fed import LocalModelWeights
# from models.nets import get_model
# from models.test import test_img
# from models.update import get_local_update_objects
#
#
# # =============================================================================
# # 修复: FedAvg 加权平均函数 (用于 Edge 聚合 和 Cloud 聚合)
# # 根据客户端的本地样本数量(weight_list)进行加权，保证 Non-IID 下的精度
# # =============================================================================
# def FedAvg(w_list, weight_list):
#     """
#     加权联邦平均算法
#     :param w_list: list of state_dict (模型权重列表)
#     :param weight_list: list of int (每个模型对应的样本数量)
#     :return: averaged state_dict
#     """
#     if not w_list or not weight_list:
#         return None
#
#     total_samples = sum(weight_list)
#     w_avg = copy.deepcopy(w_list[0])
#
#     for k in w_avg.keys():
#         w_avg[k] = w_avg[k] * weight_list[0]
#         for i in range(1, len(w_list)):
#             w_avg[k] += w_list[i][k] * weight_list[i]
#         w_avg[k] = torch.div(w_avg[k], total_samples)
#
#     return w_avg
#
#
# if __name__ == '__main__':
#     start = time.time()
#     # parse args
#     args = args_parser()
#     args.device = torch.device(
#         'cuda:{}'.format(args.gpu)
#         if torch.cuda.is_available() and args.gpu != -1
#         else 'cpu',
#     )
#     args.schedule = [int(x) for x in args.schedule]
#
#     # 【已注释双模型逻辑】判断是否使用双模型的标志位
#     # args.send_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix', ]
#     args.send_2_models = False  # 强制关闭双模型
#
#     # =============================================================================
#     # 日志文件设置
#     # =============================================================================
#     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     # 文件名标记为 Hierarchical (分层)
#
#
#     # 定义目标文件夹名称
#     save_dir = "resultdate"
#
#     # 如果该文件夹不存在，自动创建它（防止报错）
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     #将文件夹路径和文件名拼接到一起
#     file_name = f"详细模型{args.epochs}_{args.dataset}_{args.group_noise_rate}_{args.method}_{timestamp}.txt"
#     log_filename = os.path.join(save_dir, file_name)
#     #  新增：定义一个专门只存训练指标的“纯净版”文件
#     metrics_file_name = f"总体模型{args.epochs}_{args.dataset}_{args.group_noise_rate}_{args.method}_{timestamp}.txt"
#     metrics_log_filename = os.path.join(save_dir, metrics_file_name)
#
#     print(f"Results will be saved to: {log_filename}")
#     print(f"Metrics ONLY will be saved to: {metrics_log_filename}")
#
#     # print(f"Results will be saved to: {log_filename}")
#
#     with open(log_filename, 'w') as f:
#         f.write(f"Experiment Start: {timestamp}\n")
#         f.write("=" * 50 + "\n")
#         f.write(f"Args: {args}\n")
#         f.write("=" * 50 + "\n")
#
#     for x in vars(args).items():
#         print(x)
#
#     if not torch.cuda.is_available():
#         exit('ERROR: Cuda is not available!')
#     print('torch version: ', torch.__version__)
#     print('torchvision version: ', torchvision.__version__)
#
#     # Seed
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     torch.backends.cudnn.deterministic = True
#     np.random.seed(args.seed)
#
#     # Arbitrary gaussian noise
#     gaussian_noise = torch.randn(1, 3, 32, 32)
#
#     ##############################
#     # Load dataset and split users
#     ##############################
#     dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
#     labels = np.array(dataset_train.train_labels)
#     img_size = dataset_train[0][0].shape  # used to get model
#     args.img_size = int(img_size[1])
#
#     # Sample users (iid / non-iid)
#     if args.iid:
#         dict_users = sample_iid(labels, args.num_users)
#     elif args.partition == 'shard':
#         dict_users = sample_noniid_shard(
#             labels=labels,
#             num_users=args.num_users,
#             num_shards=args.num_shards,
#         )
#     elif args.partition == 'dirichlet':
#         dict_users = sample_dirichlet(
#             labels=labels,
#             num_users=args.num_users,
#             alpha=args.dd_alpha,
#         )
#
#     ##############################
#     # Add label noise to data
#     ##############################
#     if sum(args.noise_group_num) != args.num_users:
#         exit('Error: sum of the number of noise group have to be equal the number of users')
#
#     if len(args.group_noise_rate) == 1:
#         args.group_noise_rate = args.group_noise_rate * 2
#
#     if not len(args.noise_group_num) == len(args.group_noise_rate) and \
#             len(args.group_noise_rate) * 2 == len(args.noise_type_lst):
#         exit('Error: The noise input is invalid.')
#
#     args.group_noise_rate = [(args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
#                              for i in range(len(args.group_noise_rate) // 2)]
#
#     user_noise_type_rates = []
#     for num_users_in_group, noise_type, (min_group_noise_rate, max_group_noise_rate) in zip(
#             args.noise_group_num, args.noise_type_lst, args.group_noise_rate):
#         noise_types = [noise_type] * num_users_in_group
#
#         step = (max_group_noise_rate - min_group_noise_rate) / num_users_in_group
#         noise_rates = np.array(range(num_users_in_group)) * step + min_group_noise_rate
#
#         user_noise_type_rates += [*zip(noise_types, noise_rates)]
#
#     for user, (user_noise_type, user_noise_rate) in enumerate(user_noise_type_rates):
#         if user_noise_type != "clean":
#             data_indices = list(copy.deepcopy(dict_users[user]))
#             random.seed(args.seed)
#             random.shuffle(data_indices)
#             noise_index = int(len(data_indices) * user_noise_rate)
#             for d_idx in data_indices[:noise_index]:
#                 true_label = dataset_train.train_labels[d_idx]
#                 noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
#                 dataset_train.train_labels[d_idx] = noisy_label
#
#     # Logging loaders
#     logging_args = dict(batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)
#     log_train_data_loader = torch.utils.data.DataLoader(dataset_train, **logging_args)
#     log_test_data_loader = torch.utils.data.DataLoader(dataset_test, **logging_args)
#
#     ##############################
#     # Build model & Init
#     ##############################
#     net_glob = get_model(args)
#     net_glob = net_glob.to(args.device)
#
#     # 【已注释双模型逻辑】不再初始化第二个全局模型 net_glob2
#     # net_glob2 = None
#     # if args.send_2_models:
#     #     net_glob2 = get_model(args)
#     #     net_glob2 = net_glob2.to(args.device)
#
#     # 学习率调度相关
#     forget_rate_schedule = []
#     if args.method in ['coteaching', 'coteaching+']:
#         num_gradual = args.warmup_epochs
#         forget_rate = args.forget_rate
#         exponent = 1
#         forget_rate_schedule = np.ones(args.epochs) * forget_rate
#         forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)
#
#     pred_user_noise_rates = [args.forget_rate] * args.num_users
#
#     # Initialize local update objects
#     local_update_objects = get_local_update_objects(
#         args=args,
#         dataset_train=dataset_train,
#         dict_users=dict_users,
#         noise_rates=pred_user_noise_rates,
#         gaussian_noise=gaussian_noise,
#     )
#
#     # ========================= [初始化所有客户端模型] =========================
#     # 确保所有客户端从相同的全局权重开始 (模拟系统初始化)
#     initial_state = copy.deepcopy(net_glob.state_dict())
#     for i in range(args.num_users):
#         local_update_objects[i].net1.load_state_dict(initial_state)
#
#     # ========================= [配置三层架构 (Edge Settings)] =========================
#     NUM_EDGES = args.num_edges # 定义边缘服务器数量
#     CLIENTS_PER_EDGE = args.num_users // NUM_EDGES
#
#     # 建立 Edge -> Client 的映射
#     edge_clients_map = {}
#     all_client_ids = list(range(args.num_users))
#
#     # 修复：处理无法整除的情况，将最后剩下的所有客户端都分给最后一个 Edge
#     for i in range(NUM_EDGES):
#         start_idx = i * CLIENTS_PER_EDGE
#         if i == NUM_EDGES - 1:
#             edge_clients_map[i] = all_client_ids[start_idx:]
#         else:
#             edge_clients_map[i] = all_client_ids[start_idx: start_idx + CLIENTS_PER_EDGE]
#     args.clients_per_edge = args.num_users // args.num_edges
#     NUM_CLIENT=args.clients_per_edge
#     print(f"\nStructure: {NUM_EDGES} 边缘服务器.每个边缘服务器有：{NUM_CLIENT}客户端")
#     print("开始分层训练（客户端 - 边缘 - 云端）\n")
#
#     ##############################
#     # Training Loop (Client-Edge-Cloud)
#     ##############################
#     for epoch in range(args.epochs):
#         # 1. 学习率调整
#         if (epoch + 1) in args.schedule:
#             print("Learning Rate Decay Epoch {}".format(epoch + 1))
#             print("{} => {}".format(args.lr, args.lr * args.lr_decay))
#             args.lr *= args.lr_decay
#
#         if len(forget_rate_schedule) > 0:
#             args.forget_rate = forget_rate_schedule[epoch]
#
#         args.g_epoch = epoch
#
#         # 存储每个 Edge 聚合后的模型 (用于最后 Cloud 聚合)
#         edge_weights_list = []
#         edge_samples_list = []  # 用于云端加权平均
#         edge_losses_list = []
#         #  新增：用于临时存储本轮各个边缘服务器的测试结果字符串
#         edge_log_strings = []
#
#         # --- 第一层循环: 遍历每个 Edge Server ---
#         for edge_id in range(NUM_EDGES):
#             current_edge_clients = edge_clients_map[edge_id]
#
#             # 存储该 Edge 下所有 Client 更新后的模型
#             client_weights_list = []
#             client_samples_list = []  # 用于边缘端加权平均
#             client_losses = []
#
#
#
#             # 修复：Edge 端下发模型时进行按比例(frac)随机采样，符合真实联邦学习设定
#             m = max(int(args.frac * len(current_edge_clients)), 1)
#             selected_clients = np.random.choice(current_edge_clients, m, replace=False)
#
#             # --- 第二层循环: Edge 下发模型，Client 本地更新 ---
#             for client_idx in selected_clients:
#                 local = local_update_objects[client_idx]
#                 local.args = args
#
#                 # 模拟下发：深拷贝一份全局模型给客户端
#                 net_local = copy.deepcopy(net_glob).to(args.device)
#
#                 # 获取该客户端的数据量，用于加权平均
#                 client_sample_size = len(dict_users[client_idx])
#
#                 # 根据方法选择训练逻辑
#                 if args.method == "fedrn":
#                     if epoch < args.warmup_epochs:
#                         w, loss = local.train_phase1(net_local)
#                     else:
#                         w, loss = local.train_phase_self_clean(net_local)
#
#                 # 【已注释双模型逻辑】屏蔽了双模型的训练分支
#                 # elif args.send_2_models:
#                 #     net_local2 = copy.deepcopy(net_glob2).to(args.device) if net_glob2 else None
#                 #     w, loss, w2, loss2 = local.train(net_local, net_local2)
#
#                 else:
#                     # 默认基础方法
#                     w, loss = local.train(net_local)
#
#                 # 修复：将权重转移到 CPU，防止 GPU 显存溢出 (OOM)
#                 w_cpu = {k: v.cpu() for k, v in w.items()}
#
#                 client_weights_list.append(w_cpu)
#                 client_samples_list.append(client_sample_size)
#                 client_losses.append(loss)
#
#             # --- Edge Aggregation (边缘聚合) ---
#
#             if len(client_weights_list) > 0:
#                 # 修复：使用加权平均 FedAvg
#                 w_edge = FedAvg(client_weights_list, client_samples_list)
#                 edge_weights_list.append(w_edge)
#
#                 # Edge 服务器包含的总样本数为该 Edge 下采样客户端的样本数总和
#                 edge_samples_list.append(sum(client_samples_list))
#
#                 # 计算平均 Loss 仅用于日志展示
#                 avg_edge_loss = sum(client_losses) / len(client_losses)
#                 edge_losses_list.append(avg_edge_loss)
#
#                 # 👉 新增：立即评估该边缘服务器聚合出的模型性能（在测试集上）
#                 net_edge = copy.deepcopy(net_glob).to(args.device)
#                 net_edge.load_state_dict(w_edge)
#                 edge_test_acc, edge_test_loss = test_img(net_edge, log_test_data_loader, args)
#
#                 # 打印到控制台，带有缩进，方便和全局模型区分
#                 edge_str = f"  --> [Edge Server {edge_id}] Test Acc: {edge_test_acc:.2f}% | Test Loss: {edge_test_loss:.6f}"
#                 edge_log_strings.append(edge_str)
#
#             # --- Edge Aggregation (边缘聚合) ---
#             # if len(client_weights_list) > 0:
#             #     # 修复：使用加权平均 FedAvg
#             #     w_edge = FedAvg(client_weights_list, client_samples_list)
#             #     edge_weights_list.append(w_edge)
#             #
#             #     # Edge 服务器包含的总样本数为该 Edge 下采样客户端的样本数总和
#             #     edge_samples_list.append(sum(client_samples_list))
#             #
#             #     # 计算平均 Loss 仅用于日志展示
#             #     avg_edge_loss = sum(client_losses) / len(client_losses)
#             #     edge_losses_list.append(avg_edge_loss)
#
#         # --- Cloud Aggregation (云端聚合) ---
#         # 聚合所有 Edge 的模型，更新全局模型
#         if len(edge_weights_list) > 0:
#             # 修复：使用加权平均 FedAvg
#             w_glob = FedAvg(edge_weights_list, edge_samples_list)
#             net_glob.load_state_dict(w_glob)
#
#             # 必须遍历所有客户端对象，将最新的全局权重 w_glob 同步到它们的 net1 中
#             # 这样在下一轮 epoch 开始时，fit_gmm(self.net1) 才能基于最新的模型进行筛选
#             for i in range(args.num_users):
#                 local_update_objects[i].net1.load_state_dict(w_glob)
#         # ========================= [循环结束] =========================
#
#         # ========================= [测试与日志输出] =========================
#         train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
#         test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
#
#         results = dict(train_acc=train_acc, train_loss=train_loss,
#                        test_acc=test_acc, test_loss=test_loss)
#
#         # 构造日志字符串
#         log_str_round = 'Round {:3d}'.format(epoch)
#         log_str_metrics = ' - '.join([f'{k}: {v:.6f}' for k, v in results.items()])
#
#
#         # 打印到控制台
#         print(log_str_round)
#         for edge_str in edge_log_strings:
#             print(edge_str)
#         print(log_str_metrics)
#
#
#         # 写入文件
#         with open(log_filename, 'a') as f:
#             f.write(log_str_round + '\n')
#             # 👉 新增：把边缘服务器的日志写入 txt
#             for edge_str in edge_log_strings:
#                 f.write(edge_str + '\n')
#             f.write(log_str_metrics + '\n')
#             f.write('-' * 30 + '\n')
#         #  新增：只把 Round 和 train_acc/loss 等核心指标写入“纯净版”文件
#         with open(metrics_log_filename, 'a') as mf:
#             mf.write(log_str_round + '\n')
#             mf.write(log_str_metrics + '\n')
#
#!/usr/bin/env python
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
# 根据客户端的本地样本数量(weight_list)进行加权，保证 Non-IID 下的精度
# =============================================================================
def FedAvg(w_list, weight_list):
    """
    加权联邦平均算法
    :param w_list: list of state_dict (模型权重列表)
    :param weight_list: list of int (每个模型对应的样本数量)
    :return: averaged state_dict
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

    # 【已注释双模型逻辑】判断是否使用双模型的标志位
    # args.send_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix', ]
    args.send_2_models = False  # 强制关闭双模型

    # =============================================================================
    # 日志文件设置
    # =============================================================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 文件名标记为 Hierarchical (分层)


    # 定义目标文件夹名称
    save_dir = "resultdate"

    # 如果该文件夹不存在，自动创建它（防止报错）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #将文件夹路径和文件名拼接到一起
    file_name = f"详细模型{args.epochs}_{args.dataset}_{args.group_noise_rate}_{args.method}_{timestamp}.txt"
    log_filename = os.path.join(save_dir, file_name)
    #  新增：定义一个专门只存训练指标的“纯净版”文件
    metrics_file_name = f"总体模型{args.epochs}_{args.dataset}_{args.group_noise_rate}_{args.method}_{timestamp}.txt"
    metrics_log_filename = os.path.join(save_dir, metrics_file_name)

    print(f"Results will be saved to: {log_filename}")
    print(f"Metrics ONLY will be saved to: {metrics_log_filename}")

    # print(f"Results will be saved to: {log_filename}")

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

    # Arbitrary gaussian noise
    gaussian_noise = torch.randn(1, 3, 32, 32)
#######
    ##############################
    # Load dataset and split users
    ##############################
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
#
#     # =============================================================================
#     # 👉 终极提速大招：暴力阉割数据集 (保留 20%) - 升级版
#     # =============================================================================
#     # CIFAR-10 原始有 50000 训练集，10000 测试集。这里我们各取前 20%
#     NUM_KEEP_TRAIN = 25000
#     NUM_KEEP_TEST = 5000
#
#     # 无死角截断训练集
#     if hasattr(dataset_train, 'data'): dataset_train.data = dataset_train.data[:NUM_KEEP_TRAIN]
#     if hasattr(dataset_train, 'train_data'): dataset_train.train_data = dataset_train.train_data[:NUM_KEEP_TRAIN]
#     if hasattr(dataset_train, 'targets'): dataset_train.targets = dataset_train.targets[:NUM_KEEP_TRAIN]
#     if hasattr(dataset_train, 'train_labels'): dataset_train.train_labels = dataset_train.train_labels[:NUM_KEEP_TRAIN]
#
#     # 无死角截断测试集
#     if hasattr(dataset_test, 'data'): dataset_test.data = dataset_test.data[:NUM_KEEP_TEST]
#     if hasattr(dataset_test, 'test_data'): dataset_test.test_data = dataset_test.test_data[:NUM_KEEP_TEST]
#     if hasattr(dataset_test, 'targets'): dataset_test.targets = dataset_test.targets[:NUM_KEEP_TEST]
#     if hasattr(dataset_test, 'test_labels'): dataset_test.test_labels = dataset_test.test_labels[:NUM_KEEP_TEST]
#
#     print(f"🚀 数据集已真正阉割！当前训练集大小: {len(dataset_train)}, 测试集大小: {len(dataset_test)}")
#     # =============================================================================

    # 顺手帮你修复那个烦人的警告："train_labels has been renamed targets"
    if hasattr(dataset_train, 'targets'):
        labels = np.array(dataset_train.targets)
    else:
        labels = np.array(dataset_train.train_labels)

    img_size = dataset_train[0][0].shape  # used to get model
    args.img_size = int(img_size[1])
    #######


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
                # 兼容新老版本的标签名称
                if hasattr(dataset_train, 'targets'):
                    true_label = dataset_train.targets[d_idx]
                    noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
                    dataset_train.targets[d_idx] = noisy_label
                else:
                    true_label = dataset_train.train_labels[d_idx]
                    noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
                    dataset_train.train_labels[d_idx] = noisy_label
            # for d_idx in data_indices[:noise_index]:
            #     true_label = dataset_train.train_labels[d_idx]
            #     noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
            #     dataset_train.train_labels[d_idx] = noisy_label

    # Logging loaders
    logging_args = dict(batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, **logging_args)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, **logging_args)

    ##############################
    # Build model & Init
    ##############################
    net_glob = get_model(args)
    net_glob = net_glob.to(args.device)

    # 【已注释双模型逻辑】不再初始化第二个全局模型 net_glob2
    # net_glob2 = None
    # if args.send_2_models:
    #     net_glob2 = get_model(args)
    #     net_glob2 = net_glob2.to(args.device)

    # 学习率调度相关
    forget_rate_schedule = []
    if args.method in ['coteaching', 'coteaching+']:
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
    # 确保所有客户端从相同的全局权重开始 (模拟系统初始化)
    initial_state = copy.deepcopy(net_glob.state_dict())
    for i in range(args.num_users):
        local_update_objects[i].net1.load_state_dict(initial_state)

    # ========================= [配置三层架构 (Edge Settings)] =========================
    NUM_EDGES = args.num_edges # 定义边缘服务器数量
    CLIENTS_PER_EDGE = args.num_users // NUM_EDGES

    # 建立 Edge -> Client 的映射
    edge_clients_map = {}
    all_client_ids = list(range(args.num_users))

    # 修复：处理无法整除的情况，将最后剩下的所有客户端都分给最后一个 Edge
    for i in range(NUM_EDGES):
        start_idx = i * CLIENTS_PER_EDGE
        if i == NUM_EDGES - 1:
            edge_clients_map[i] = all_client_ids[start_idx:]
        else:
            edge_clients_map[i] = all_client_ids[start_idx: start_idx + CLIENTS_PER_EDGE]
    args.clients_per_edge = args.num_users // args.num_edges
    NUM_CLIENT=args.clients_per_edge
    print(f"\nStructure: {NUM_EDGES} 边缘服务器.每个边缘服务器有：{NUM_CLIENT}客户端")
    print("开始分层训练（客户端 - 边缘 - 云端）\n")

    ##############################
    # Training Loop (Client-Edge-Cloud)
    ##############################
    for epoch in range(args.epochs):
        # 1. 学习率调整
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        args.g_epoch = epoch

        # 存储每个 Edge 聚合后的模型 (用于最后 Cloud 聚合)
        edge_weights_list = []
        edge_samples_list = []  # 用于云端加权平均
        edge_losses_list = []
        #  新增：用于临时存储本轮各个边缘服务器的测试结果字符串
        edge_log_strings = []

        # --- 第一层循环: 遍历每个 Edge Server ---
        for edge_id in range(NUM_EDGES):
            current_edge_clients = edge_clients_map[edge_id]

            # 存储该 Edge 下所有 Client 更新后的模型
            client_weights_list = []
            client_samples_list = []  # 用于边缘端加权平均
            client_losses = []



            # 修复：Edge 端下发模型时进行按比例(frac)随机采样，符合真实联邦学习设定
            m = max(int(args.frac * len(current_edge_clients)), 1)
            selected_clients = np.random.choice(current_edge_clients, m, replace=False)

            # --- 第二层循环: Edge 下发模型，Client 本地更新 ---
            for client_idx in selected_clients:
                local = local_update_objects[client_idx]
                local.args = args

                # 模拟下发：深拷贝一份全局模型给客户端
                net_local = copy.deepcopy(net_glob).to(args.device)

                # 获取该客户端的数据量，用于加权平均
                client_sample_size = len(dict_users[client_idx])

                # 根据方法选择训练逻辑
                if args.method == "fedrn":
                    if epoch < args.warmup_epochs:
                        w, loss = local.train_phase1(net_local)
                    else:
                        w, loss = local.train_phase_self_clean(net_local)

                # 【已注释双模型逻辑】屏蔽了双模型的训练分支
                # elif args.send_2_models:
                #     net_local2 = copy.deepcopy(net_glob2).to(args.device) if net_glob2 else None
                #     w, loss, w2, loss2 = local.train(net_local, net_local2)

                else:
                    # 默认基础方法
                    w, loss = local.train(net_local)

                # 修复：将权重转移到 CPU，防止 GPU 显存溢出 (OOM)
                w_cpu = {k: v.cpu() for k, v in w.items()}

                client_weights_list.append(w_cpu)
                client_samples_list.append(client_sample_size)
                client_losses.append(loss)

            # --- Edge Aggregation (边缘聚合) ---

            if len(client_weights_list) > 0:
                # 修复：使用加权平均 FedAvg
                w_edge = FedAvg(client_weights_list, client_samples_list)
                edge_weights_list.append(w_edge)

                # Edge 服务器包含的总样本数为该 Edge 下采样客户端的样本数总和
                edge_samples_list.append(sum(client_samples_list))

                # 计算平均 Loss 仅用于日志展示
                avg_edge_loss = sum(client_losses) / len(client_losses)
                edge_losses_list.append(avg_edge_loss)

                # 👉 新增：立即评估该边缘服务器聚合出的模型性能（在测试集上）
                net_edge = copy.deepcopy(net_glob).to(args.device)
                net_edge.load_state_dict(w_edge)
                edge_test_acc, edge_test_loss = test_img(net_edge, log_test_data_loader, args)

                # 打印到控制台，带有缩进，方便和全局模型区分
                edge_str = f"  --> [Edge Server {edge_id}] Test Acc: {edge_test_acc:.2f}% | Test Loss: {edge_test_loss:.6f}"
                edge_log_strings.append(edge_str)

            # --- Edge Aggregation (边缘聚合) ---
            # if len(client_weights_list) > 0:
            #     # 修复：使用加权平均 FedAvg
            #     w_edge = FedAvg(client_weights_list, client_samples_list)
            #     edge_weights_list.append(w_edge)
            #
            #     # Edge 服务器包含的总样本数为该 Edge 下采样客户端的样本数总和
            #     edge_samples_list.append(sum(client_samples_list))
            #
            #     # 计算平均 Loss 仅用于日志展示
            #     avg_edge_loss = sum(client_losses) / len(client_losses)
            #     edge_losses_list.append(avg_edge_loss)

        # --- Cloud Aggregation (云端聚合) ---
        # 聚合所有 Edge 的模型，更新全局模型
        if len(edge_weights_list) > 0:
            # 修复：使用加权平均 FedAvg
            w_glob = FedAvg(edge_weights_list, edge_samples_list)
            net_glob.load_state_dict(w_glob)

            # 必须遍历所有客户端对象，将最新的全局权重 w_glob 同步到它们的 net1 中
            # 这样在下一轮 epoch 开始时，fit_gmm(self.net1) 才能基于最新的模型进行筛选
            for i in range(args.num_users):
                local_update_objects[i].net1.load_state_dict(w_glob)
        # ========================= [循环结束] =========================

        # ========================= [测试与日志输出] =========================
        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)

        results = dict(train_acc=train_acc, train_loss=train_loss,
                       test_acc=test_acc, test_loss=test_loss)

        # 构造日志字符串
        log_str_round = 'Round {:3d}'.format(epoch)
        log_str_metrics = ' - '.join([f'{k}: {v:.6f}' for k, v in results.items()])


        # 打印到控制台
        print(log_str_round)
        for edge_str in edge_log_strings:
            print(edge_str)
        print(log_str_metrics)


        # 写入文件
        with open(log_filename, 'a') as f:
            f.write(log_str_round + '\n')
            # 👉 新增：把边缘服务器的日志写入 txt
            for edge_str in edge_log_strings:
                f.write(edge_str + '\n')
            f.write(log_str_metrics + '\n')
            f.write('-' * 30 + '\n')
        #  新增：只把 Round 和 train_acc/loss 等核心指标写入“纯净版”文件
        with open(metrics_log_filename, 'a') as mf:
            mf.write(log_str_round + '\n')
            mf.write(log_str_metrics + '\n')

