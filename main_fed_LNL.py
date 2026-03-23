# # # !/usr/bin/env python
# # # -*- coding: utf-8 -*-
# # # Python version: 3.6
# # import os
# # import sys
# #
# # # =============================================================================
# # # Windows 环境兼容性修复 (DLL Load Failed Fix)
# # # =============================================================================
# # conda_path = r"C:\Users\25839\.conda\envs\improve-FedRN-main\Library\bin"
# # if os.path.exists(conda_path):
# #     os.environ['PATH'] = conda_path + os.pathsep + os.environ['PATH']
# # else:
# #     print(f"警告: 找不到路径 {conda_path}，请检查你的环境安装位置")
# #
# # import copy
# # import numpy as np
# # import random
# # import time
# # import datetime  # 新增：用于生成带时间戳的文件名
# #
# # import torchvision
# # import torch
# # from torch.utils.data import DataLoader
# #
# # from utils import load_dataset
# # from utils.options import args_parser
# # from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
# # from utils.utils import noisify_label
# #
# # from models.fed import LocalModelWeights
# # from models.nets import get_model
# # from models.test import test_img
# # from models.update import get_local_update_objects
# #
# # #通用联邦平均聚合函数 (用于 Edge 聚合 和 Cloud 聚合)
# # def FedAvg(w_list):
# #     """
# #     计算模型参数的平均值
# #     :param w_list: list of state_dict
# #     :return: averaged state_dict
# #     """
# #     w_avg = copy.deepcopy(w_list[0])
# #     for k in w_avg.keys():
# #         for i in range(1, len(w_list)):
# #             w_avg[k] += w_list[i][k]
# #         w_avg[k] = torch.div(w_avg[k], len(w_list))
# #     return w_avg
# #
# # if __name__ == '__main__':
# #     start = time.time()
# #     # parse args
# #     args = args_parser()
# #     args.device = torch.device(
# #         'cuda:{}'.format(args.gpu)
# #         if torch.cuda.is_available() and args.gpu != -1
# #         else 'cpu',
# #     )
# #     args.schedule = [int(x) for x in args.schedule]
# #     args.send_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix', ]
# #
# #     # =============================================================================
# #     # 新增: 定义日志文件路径并写入参数头信息
# #     # =============================================================================
# #     # 生成一个带时间戳的文件名，防止覆盖之前的实验结果
# #     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# #     log_filename = f"results_{args.dataset}_{args.method}_{timestamp}.txt"
# #
# #     print(f"Results will be saved to: {log_filename}")
# #
# #     with open(log_filename, 'w') as f:
# #         f.write(f"Experiment Start: {timestamp}\n")
# #         f.write("=" * 50 + "\n")
# #         f.write(f"Args: {args}\n")
# #         f.write("=" * 50 + "\n")
# #
# #     for x in vars(args).items():
# #         print(x)
# #
# #     if not torch.cuda.is_available():
# #         exit('ERROR: Cuda is not available!')
# #     print('torch version: ', torch.__version__)
# #     print('torchvision version: ', torchvision.__version__)
# #
# #     # Seed
# #     torch.manual_seed(args.seed)
# #     torch.cuda.manual_seed(args.seed)
# #     torch.backends.cudnn.deterministic = True
# #     np.random.seed(args.seed)
# #
# #     # Arbitrary gaussian noise
# #     gaussian_noise = torch.randn(1, 3, 32, 32)
# #
# #     ##############################
# #     # Load dataset and split users
# #     ##############################
# #     dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
# #     labels = np.array(dataset_train.train_labels)
# #     img_size = dataset_train[0][0].shape  # used to get model
# #     args.img_size = int(img_size[1])
# #
# #     # Sample users (iid / non-iid)
# #     if args.iid:
# #         dict_users = sample_iid(labels, args.num_users)
# #
# #     elif args.partition == 'shard':
# #         dict_users = sample_noniid_shard(
# #             labels=labels,
# #             num_users=args.num_users,
# #             num_shards=args.num_shards,
# #         )
# #
# #     elif args.partition == 'dirichlet':
# #         dict_users = sample_dirichlet(
# #             labels=labels,
# #             num_users=args.num_users,
# #             alpha=args.dd_alpha,
# #         )
# #
# #     ##############################
# #     # Add label noise to data
# #     ##############################
# #     if sum(args.noise_group_num) != args.num_users:
# #         exit('Error: sum of the number of noise group have to be equal the number of users')
# #
# #     if len(args.group_noise_rate) == 1:
# #         args.group_noise_rate = args.group_noise_rate * 2
# #
# #     if not len(args.noise_group_num) == len(args.group_noise_rate) and \
# #             len(args.group_noise_rate) * 2 == len(args.noise_type_lst):
# #         exit('Error: The noise input is invalid.')
# #
# #     args.group_noise_rate = [(args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
# #                              for i in range(len(args.group_noise_rate) // 2)]
# #
# #     user_noise_type_rates = []
# #     for num_users_in_group, noise_type, (min_group_noise_rate, max_group_noise_rate) in zip(
# #             args.noise_group_num, args.noise_type_lst, args.group_noise_rate):
# #         noise_types = [noise_type] * num_users_in_group
# #
# #         step = (max_group_noise_rate - min_group_noise_rate) / num_users_in_group
# #         noise_rates = np.array(range(num_users_in_group)) * step + min_group_noise_rate
# #
# #         user_noise_type_rates += [*zip(noise_types, noise_rates)]
# #
# #     for user, (user_noise_type, user_noise_rate) in enumerate(user_noise_type_rates):
# #         if user_noise_type != "clean":
# #             data_indices = list(copy.deepcopy(dict_users[user]))
# #
# #             # for reproduction
# #             random.seed(args.seed)
# #             random.shuffle(data_indices)
# #
# #             noise_index = int(len(data_indices) * user_noise_rate)
# #
# #             for d_idx in data_indices[:noise_index]:
# #                 true_label = dataset_train.train_labels[d_idx]
# #                 noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
# #                 dataset_train.train_labels[d_idx] = noisy_label
# #
# #     # for logging purposes
# #     logging_args = dict(
# #         batch_size=args.bs,
# #         num_workers=args.num_workers,
# #         pin_memory=True,
# #     )
# #     log_train_data_loader = torch.utils.data.DataLoader(dataset_train, **logging_args)
# #     log_test_data_loader = torch.utils.data.DataLoader(dataset_test, **logging_args)
# #
# #     ##############################
# #     # Build model
# #     ##############################
# #     net_glob = get_model(args)
# #     net_glob = net_glob.to(args.device)
# #
# #     if args.send_2_models:
# #         net_glob2 = get_model(args)
# #         net_glob2 = net_glob2.to(args.device)
# #
# #     ##############################
# #     # Training
# #     ##############################
# #     CosineSimilarity = torch.nn.CosineSimilarity()
# #
# #     forget_rate_schedule = []
# #     if args.method in ['coteaching', 'coteaching+']:
# #         num_gradual = args.warmup_epochs
# #         forget_rate = args.forget_rate
# #         exponent = 1
# #         forget_rate_schedule = np.ones(args.epochs) * forget_rate
# #         forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)
# #
# #     pred_user_noise_rates = [args.forget_rate] * args.num_users
# #
# #     # Initialize local model weights
# #     fed_args = dict(
# #         all_clients=args.all_clients,
# #         num_users=args.num_users,
# #         method=args.fed_method,
# #         dict_users=dict_users,
# #     )
# #
# #     local_weights = LocalModelWeights(net_glob=net_glob, **fed_args)
# #     if args.send_2_models:
# #         local_weights2 = LocalModelWeights(net_glob=net_glob2, **fed_args)
# #
# #     # Initialize local update objects
# #     local_update_objects = get_local_update_objects(
# #         args=args,
# #         dataset_train=dataset_train,
# #         dict_users=dict_users,
# #         noise_rates=pred_user_noise_rates,
# #         gaussian_noise=gaussian_noise,
# #     )
# #     #  [初始化所有客户端模型]
# #     # 确保所有客户端从相同的全局权重开始
# #     initial_state = copy.deepcopy(net_glob.state_dict())
# #     for i in range(args.num_users):
# #         local_update_objects[i].net1.load_state_dict(initial_state)
# #
# #     NUM_EDGES = 5  # 定义边缘服务器数量
# #     CLIENTS_PER_EDGE = args.num_users // NUM_EDGES
# #
# #     # 建立 Edge -> Client 的映射
# #     # edge_clients_map[0] = [0, 1, ..., 9] (假设总共50个用户)
# #     edge_clients_map = {}
# #     all_client_ids = list(range(args.num_users))
# #     for i in range(NUM_EDGES):
# #         # 简单切分，每个 Edge 管理一部分用户
# #         edge_clients_map[i] = all_client_ids[i * CLIENTS_PER_EDGE: (i + 1) * CLIENTS_PER_EDGE]
# #
# #     print(f"\nStructure: {NUM_EDGES} Edge Servers, {CLIENTS_PER_EDGE} Clients per Edge.")
# #     print("Start Hierarchical Training (Client-Edge-Cloud)...\n")
# #
# #     for epoch in range(args.epochs):
# #         if (epoch + 1) in args.schedule:
# #             print("Learning Rate Decay Epoch {}".format(epoch + 1))
# #             print("{} => {}".format(args.lr, args.lr * args.lr_decay))
# #             args.lr *= args.lr_decay
# #
# #         if len(forget_rate_schedule) > 0:
# #             args.forget_rate = forget_rate_schedule[epoch]
# #
# #         local_losses = []
# #         local_losses2 = []
# #         args.g_epoch = epoch
# #
# #         m = max(int(args.frac * args.num_users), 1)
# #         idxs_users = np.random.choice(range(args.num_users), m, replace=False)
# #
# #         # Local Update
# #         for client_num, idx in enumerate(idxs_users):
# #             local = local_update_objects[idx]
# #             local.args = args
# #
# #             if args.method == "fedrn":
# #                 if epoch < args.warmup_epochs:
# #                     # Phase 1: 预热 (已修改为不计算指标)
# #                     w, loss = local.train_phase1(copy.deepcopy(net_glob).to(args.device))
# #                 else:
# #                     # Get similarity, expertise values
# #                     # 直接调用修改后的 train_phase2，仅传入全局模型副本
# #                     w, loss = local.train_phase2(copy.deepcopy(net_glob).to(args.device))
# #
# #
# #             elif args.send_2_models:
# #                 w, loss, w2, loss2 = local.train(
# #                     copy.deepcopy(net_glob).to(args.device),
# #                     copy.deepcopy(net_glob2).to(args.device),
# #                 )
# #                 local_weights2.update(idx, w2)
# #                 local_losses2.append(copy.deepcopy(loss2))
# #
# #             else:
# #                 w, loss = local.train(copy.deepcopy(net_glob).to(args.device))
# #
# #             local_weights.update(idx, w)
# #             local_losses.append(copy.deepcopy(loss))
# #
# #         w_glob = local_weights.average()  # update global weights
# #         net_glob.load_state_dict(w_glob, strict=False)  # copy weight to net_glob
# #         local_weights.init()
# #
# #         train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
# #         test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
# #         # for logging purposes
# #         results = dict(train_acc=train_acc, train_loss=train_loss,
# #                        test_acc=test_acc, test_loss=test_loss, )
# #
# #         if args.send_2_models:
# #             w_glob2 = local_weights2.average()
# #             net_glob2.load_state_dict(w_glob2)
# #             local_weights2.init()
# #             # for logging purposes
# #             train_acc2, train_loss2 = test_img(net_glob2, log_train_data_loader, args)
# #             test_acc2, test_loss2 = test_img(net_glob2, log_test_data_loader, args)
# #             results2 = dict(train_acc2=train_acc2, train_loss2=train_loss2,
# #                             test_acc2=test_acc2, test_loss2=test_loss2, )
# #
# #             results = {**results, **results2}
# #
# #         # =============================================================================
# #         # 修改: 格式化输出字符串，并同时打印到控制台和写入文件
# #         # =============================================================================
# #
# #         # 构造日志字符串
# #         log_str_round = 'Round {:3d}'.format(epoch)
# #         log_str_metrics = ' - '.join([f'{k}: {v:.6f}' for k, v in results.items()])
# #
# #         # 1. 打印到控制台 (保持原有行为)
# #         print(log_str_round)
# #         print(log_str_metrics)
# #
# #         # 2. 写入文件
# #         with open(log_filename, 'a') as f:
# #             f.write(log_str_round + '\n')
# #             f.write(log_str_metrics + '\n')
# #             f.write('-' * 30 + '\n')  # 加个分隔符让文件更易读
#
#
# # !/usr/bin/env python
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
#     # print(f"警告: 找不到路径 {conda_path}，请检查你的环境安装位置")
#     pass
#
# import copy
# import numpy as np
# import random
# import time
# import datetime  # 新增：用于生成带时间戳的文件名
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
# # ========================= [ 新增 FedAvg 聚合函数] =========================
# def FedAvg(w_list):
#     """
#     手动实现的联邦平均算法，用于 Edge 聚合和 Cloud 聚合。
#     :param w_list: list of state_dict
#     :return: averaged state_dict
#     """
#     if not w_list:
#         return None
#
#     w_avg = copy.deepcopy(w_list[0])
#     # 简单的平均聚合
#     for k in w_avg.keys():
#         for i in range(1, len(w_list)):
#             w_avg[k] += w_list[i][k]
#         w_avg[k] = torch.div(w_avg[k], len(w_list))
#     return w_avg
#
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
#     args.send_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix', ]
#
#     # =============================================================================
#     # 日志文件设置
#     # =============================================================================
#     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     # 可以在文件名中标记这是分层架构
#     log_filename = f"results_Hierarchical_{args.dataset}_{args.method}_{timestamp}.txt"
#
#     print(f"Results will be saved to: {log_filename}")
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
#         # 注意: BaseLocalUpdate 初始化了 self.net1 和 self.net2
#         local_update_objects[i].net1.load_state_dict(initial_state)
#
#
#     # ========================= [配置三层架构 (Edge Settings)] =========================
#     NUM_EDGES = 5  # 定义边缘服务器数量
#     # 确保能整除，或者处理余数。这里简单起见假设能整除。
#     CLIENTS_PER_EDGE = args.num_users // NUM_EDGES
#
#     # 建立 Edge -> Client 的映射
#     # edge_clients_map[0] = [0, 1, ..., 9] (假设总共50个用户)
#     edge_clients_map = {}
#     all_client_ids = list(range(args.num_users))
#
#     # 简单的按顺序切分，每个 Edge 管理一部分用户
#     for i in range(NUM_EDGES):
#         edge_clients_map[i] = all_client_ids[i * CLIENTS_PER_EDGE: (i + 1) * CLIENTS_PER_EDGE]
#
#     print(f"\n开始: {NUM_EDGES} 边缘服务器, 每个服务器有{CLIENTS_PER_EDGE}个客户端.")
#     print("开始分层训练（客户端 - 边缘 - 云)\n")
#
#     ##############################
#     # Training Loop (Client-Edge-Cloud客户端-边缘服务器-云服务器)
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
#         # ========================= [修改 START: 重构为三层循环结构] =========================
#         # 存储每个 Edge 聚合后的模型 (用于最后 Cloud 聚合)
#         edge_weights_list = []
#         edge_losses_list = []
#
#         # --- 第一层循环: 遍历每个 Edge Server ---
#         for edge_id in range(NUM_EDGES):
#             current_edge_clients = edge_clients_map[edge_id]
#
#             # 存储该 Edge 下所有 Client 更新后的模型
#             client_weights_list = []
#             client_losses = []
#
#             # --- 第二层循环: Edge 下发模型，Client 本地更新 ---
#             # 这里不进行随机采样 (idxs_users)，而是处理该 Edge 下的所有 Client
#             for client_idx in current_edge_clients:
#                 local = local_update_objects[client_idx]
#                 local.args = args
#
#                 # 模拟下发：深拷贝一份全局模型给客户端
#                 # 在真实场景中，应该是 Cloud -> Edge (缓存) -> Client
#                 net_local = copy.deepcopy(net_glob).to(args.device)
#
#                 # 根据方法选择训练逻辑
#                 if args.method == "fedrn":
#                     if epoch < args.warmup_epochs:
#                         # Warm-up 阶段: 使用 train_phase1 (通常是正常训练)
#                         # 注意：这里需要 update.py 中保留 train_phase1
#                         w, loss = local.train_phase1(net_local)
#                     else:
#                         # 正式阶段: 使用新增的 train_phase_self_clean (只用本地 GMM 筛选)
#                         # 核心修改：不再调用 train_phase2，也不传入 neighbor 列表
#                         # 确保您的 update.py 中 LocalUpdateFedRN 类已经添加了 train_phase_self_clean
#                         w, loss = local.train_phase_self_clean(net_local)
#
#                 elif args.send_2_models:
#                     # 兼容 Coteaching 等双模型方法
#                     # 注意：这类方法通常比较重，可能需要额外的适配
#                     w, loss, w2, loss2 = local.train(
#                         copy.deepcopy(net_glob).to(args.device),
#                         #copy.deepcopy(net_glob2).to(args.device) if 'net_glob2' in locals() else None,
#                     )
#                     # 暂时只取模型1的权重用于演示
#                 else:
#                     # 兼容 Default 方法
#                     w, loss = local.train(net_local)
#
#                 client_weights_list.append(w)
#                 client_losses.append(loss)
#
#             # --- Edge Aggregation (边缘聚合) ---
#             # 聚合该 Edge 下所有 Client 的模型
#             if len(client_weights_list) > 0:
#                 w_edge = FedAvg(client_weights_list)
#                 edge_weights_list.append(w_edge)
#
#                 avg_edge_loss = sum(client_losses) / len(client_losses)
#                 edge_losses_list.append(avg_edge_loss)
#
#         # --- Cloud Aggregation (云端聚合) ---
#         # 聚合所有 Edge 的模型，更新全局模型
#         if len(edge_weights_list) > 0:
#             w_glob = FedAvg(edge_weights_list)
#             net_glob.load_state_dict(w_glob)
#         # ========================= [修改 END] =========================
#
#         # ========================= [修改 START: 测试与日志输出] =========================
#         train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
#         test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
#
#         results = dict(train_acc=train_acc, train_loss=train_loss,
#                        test_acc=test_acc, test_loss=test_loss)
#
#         # 兼容双模型方法的日志逻辑 (如果 args.send_2_models 为 True，需要 net_glob2 逻辑，此处略作简化)
#         # 如果您只用 fedrn，上面这些就够了
#
#         # 构造日志字符串
#         log_str_round = 'Round {:3d}'.format(epoch)
#         log_str_metrics = ' - '.join([f'{k}: {v:.6f}' for k, v in results.items()])
#
#         # 打印到控制台
#         print(log_str_round)
#         print(log_str_metrics)
#
#         # 写入文件
#         with open(log_filename, 'a') as f:
#             f.write(log_str_round + '\n')
#             f.write(log_str_metrics + '\n')
#             f.write('-' * 30 + '\n')
#         # ========================= [修改 END] =========================
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
# 新增: FedAvg 聚合函数 (用于 Edge 聚合 和 Cloud 聚合)
# =============================================================================
def FedAvg(w_list):
    """
    手动实现的联邦平均算法
    :param w_list: list of state_dict
    :return: averaged state_dict
    """
    if not w_list:
        return None

    w_avg = copy.deepcopy(w_list[0])
    # 简单的平均聚合
    for k in w_avg.keys():
        for i in range(1, len(w_list)):
            w_avg[k] += w_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_list))
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
    args.send_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix', ]

    # =============================================================================
    # 日志文件设置
    # =============================================================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 文件名标记为 Hierarchical (分层)
    log_filename = f"results_Hierarchical_{args.dataset}_{args.method}_{timestamp}.txt"

    print(f"Results will be saved to: {log_filename}")

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

    ##############################
    # Load dataset and split users
    ##############################
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
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

    # 如果使用了需要双模型的方法
    net_glob2 = None
    if args.send_2_models:
        net_glob2 = get_model(args)
        net_glob2 = net_glob2.to(args.device)

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
        # 如果有 net2 也可以初始化，但这里暂时只关注 net1

    # ========================= [配置三层架构 (Edge Settings)] =========================
    NUM_EDGES = 5  # 定义边缘服务器数量
    # 确保能整除，或者处理余数。这里简单起见假设能整除。
    CLIENTS_PER_EDGE = args.num_users // NUM_EDGES

    # 建立 Edge -> Client 的映射
    edge_clients_map = {}
    all_client_ids = list(range(args.num_users))

    # 简单的按顺序切分，每个 Edge 管理一部分用户
    for i in range(NUM_EDGES):
        edge_clients_map[i] = all_client_ids[i * CLIENTS_PER_EDGE: (i + 1) * CLIENTS_PER_EDGE]

    print(f"\nStructure: {NUM_EDGES} 边缘服务器 ，每个服务器{CLIENTS_PER_EDGE}个客户端 .")
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

        # ========================= [重构为三层循环结构] =========================
        # 存储每个 Edge 聚合后的模型 (用于最后 Cloud 聚合)
        edge_weights_list = []
        edge_losses_list = [

        ]

        # --- 第一层循环: 遍历每个 Edge Server ---
        for edge_id in range(NUM_EDGES):
            current_edge_clients = edge_clients_map[edge_id]

            # 存储该 Edge 下所有 Client 更新后的模型
            client_weights_list = []
            client_losses = []

            # --- 第二层循环: Edge 下发模型，Client 本地更新 ---
            # 这里不进行随机采样 (idxs_users)，而是处理该 Edge 下的所有 Client
            for client_idx in current_edge_clients:
                local = local_update_objects[client_idx]
                local.args = args

                # 模拟下发：深拷贝一份全局模型给客户端
                # 在真实场景中，应该是 Cloud -> Edge (缓存) -> Client
                net_local = copy.deepcopy(net_glob).to(args.device)

                # 根据方法选择训练逻辑
                if args.method == "fedrn":
                    if epoch < args.warmup_epochs:
                        # Warm-up 阶段: 使用 train_phase1 (通常是正常训练)
                        w, loss = local.train_phase1(net_local)
                    else:
                        # 正式阶段: 使用新增的 train_phase_self_clean (只用本地 GMM 筛选)
                        # 确保您的 update.py 中 LocalUpdateFedRN 类已经添加了 train_phase_self_clean
                        w, loss = local.train_phase_self_clean(net_local)

                elif args.send_2_models:
                    # 兼容 Coteaching 等双模型方法
                    # 这里需要传入两个模型，修复了您代码中被注释的参数
                    net_local2 = copy.deepcopy(net_glob2).to(args.device) if net_glob2 else None
                    w, loss, w2, loss2 = local.train(
                        net_local,
                        net_local2,
                    )
                    # 在此简单实现中，Edge聚合暂只处理模型1 (w)，如需处理 w2 需扩展代码
                else:
                    # 兼容 Default 方法net_glob.load_state_dict(w_glob)
                    w, loss = local.train(net_local)

                client_weights_list.append(w)
                client_losses.append(loss)

            # --- Edge Aggregation (边缘聚合) ---
            # 聚合该 Edge 下所有 Client 的模型
            if len(client_weights_list) > 0:
                w_edge = FedAvg(client_weights_list)
                edge_weights_list.append(w_edge)

                # 计算平均 Loss 仅用于日志展示
                avg_edge_loss = sum(client_losses) / len(client_losses)
                edge_losses_list.append(avg_edge_loss)

        # --- Cloud Aggregation (云端聚合) ---
        # 聚合所有 Edge 的模型，更新全局模型
        if len(edge_weights_list) > 0:
            w_glob = FedAvg(edge_weights_list)
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
        print(log_str_metrics)

        # 写入文件
        with open(log_filename, 'a') as f:
            f.write(log_str_round + '\n')
            f.write(log_str_metrics + '\n')
            f.write('-' * 30 + '\n')