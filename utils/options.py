# # #!/usr/bin/env python
# # # -*- coding: utf-8 -*-
# # # Python version: 3.6
# #
# # import argparse
# #
# #
# # def args_parser():
# #     parser = argparse.ArgumentParser()
# #     # label noise method
# #     parser.add_argument('--method', type=str, default='default',
# #                         choices=['default', 'selfie', 'jointoptim', 'coteaching', 'coteaching+', 'dividemix', 'fedrn'],
# #                         help='method name')
# #
# #     # federated arguments
# #     parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
# #     parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
# #     parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
# #     parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
# #     parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
# #     parser.add_argument('--bs', type=int, default=128, help="test batch size")
# #     parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
# #     parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
# #     parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
# #     parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')
# #     parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay")
# #     parser.add_argument('--weight_decay', type=float, default=0, help="sgd weight decay")
# #     parser.add_argument('--partition', type=str, choices=['shard', 'dirichlet'], default='shard')
# #     parser.add_argument('--dd_alpha', type=float, default=0.5, help="dirichlet distribution alpha")
# #     parser.add_argument('--num_shards', type=int, default=200, help="number of shards")
# #     parser.add_argument('--fed_method', type=str, default='fedavg', choices=['fedavg'],
# #                         help="federated learning method")
# #
# #     # model arguments
# #     parser.add_argument('--model', type=str, default='cnn4conv', choices=['cnn4conv'], help='model name')
# #
# #     # other arguments
# #     parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset",
# #                         choices=['cifar10', 'cifar100'])
# #     parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
# #     parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
# #     parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
# #     parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
# #     parser.add_argument('--verbose', action='store_true', help='verbose print')
# #     parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
# #     parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
# #     parser.add_argument('--num_workers', type=int, default=4, help='num_workers to load data')
# #
# #     # noise label arguments
# #     parser.add_argument('--noise_type_lst', nargs='+', default=['symmetric'], help='[pairflip, symmetric]')
# #     parser.add_argument('--noise_group_num', nargs='+', default=[100], type=int)
# #     parser.add_argument('--group_noise_rate', nargs='+', default=[0.2], type=float,
# #                         help='Should be 2 noise rates for each group: min_group_noise_rate max_group_noise_rate but '
# #                              'if there is only 1 group and 1 noise rate, same noise rate will be applied to all users')
# #     parser.add_argument('--warmup_epochs', type=int, default=100, help='number of warmup epochs')
# #
# #     # SELFIE / Joint optimization arguments
# #     parser.add_argument('--queue_size', type=int, default=15, help='size of history queue')
# #     # SELFIE / Co-teaching arguments
# #     parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate for co-teaching")
# #     # SELFIE arguments
# #     parser.add_argument('--uncertainty_threshold', type=float, default=0.05, help='uncertainty threshold')
# #     # Joint optimization arguments
# #     parser.add_argument('--alpha', type=float, default=1.2, help="alpha for joint optimization")
# #     parser.add_argument('--beta', type=float, default=0.8, help="beta for joint optimization")
# #     parser.add_argument('--labeling', type=str, default='soft', help='[soft, hard]')
# #     # MixMatch arguments
# #     parser.add_argument('--mm_alpha', default=4, type=float)
# #     parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
# #     parser.add_argument('--T', default=0.5, type=float)
# #     parser.add_argument('--p_threshold', default=0.5, type=float)
# #
# #     # FedRN
# #     parser.add_argument('--num_neighbors', type=int, default=2, help="number of neighbors")
# #     parser.add_argument('--w_alpha', type=float, help='weight alpha for our method', default=0.5)
# #
# #     args = parser.parse_args()
#
#
#
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    # 创建参数解析器，用于解析命令行输入的参数
    parser = argparse.ArgumentParser()

    # 1. 算法选择参数（指定使用的抗噪声或联邦学习方法）
    parser.add_argument('--method', type=str, default='fedco',
                        choices=['default', 'selfie', 'jointoptim', 'coteaching', 'coteaching+', 'dividemix', 'fedrn','feder','fedrnn','fedco'],
                        help='选择训练方法：default(默认，对应FedAvg)、fedrn(原论文方法)等抗噪声算法')

    # 2. 联邦学习核心参数（控制联邦训练流程）
    parser.add_argument('--epochs', type=int, default=500,
                        help="联邦学习总通信轮次（默认500轮）")
    parser.add_argument('--num_users', type=int, default=100,
                        help="用户总数K（默认100个用户）")
    parser.add_argument('--frac', type=float, default=0.1,
                        help="每轮参与训练的用户比例C（默认10%）")
    parser.add_argument('--local_ep', type=int, default=4,
                        help="每个用户的本地训练轮次E（默认5轮）")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="用户本地训练的批次大小B")
    parser.add_argument('--bs', type=int, default=128,
                        help="测试集的批次大小")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="初始学习率（默认0.01）")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="SGD优化器的动量参数（默认0.5）")
    parser.add_argument('--split', type=str, default='user',
                        help="训练集与测试集的划分方式：user(按用户)或sample(按样本)")
    parser.add_argument('--schedule', nargs='+', default=[],
                        help="学习率衰减的轮次列表（如[200, 300]表示在200和300轮衰减）26328本来是空的，调试selfie修改")
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help="学习率衰减系数（默认衰减为原来的0.1倍）")
    parser.add_argument('--weight_decay', type=float, default=0,
                        help="SGD的权重衰减（默认0，即不使用）")
    parser.add_argument('--partition', type=str, choices=['shard', 'dirichlet'], default='shard',
                        help="非IID数据划分方式：shard(分片划分)或dirichlet(狄利克雷分布划分)")
    parser.add_argument('--dd_alpha', type=float, default=0.5,
                        help="狄利克雷划分的浓度参数alpha（值越小，数据异质性越强）")
    parser.add_argument('--num_shards', type=int, default=200,
                        help="分片划分的总分片数（默认200，100用户各分2片）")
    parser.add_argument('--fed_method', type=str, default='fedavg', choices=['fedavg'],
                        help="联邦学习聚合方法（当前仅支持fedavg，即联邦平均）")

    # 3. 模型相关参数
    # parser.add_argument('--model', type=str, default='cnn4conv', choices=['cnn4conv'],
    #                     help="模型结构（默认cnn4conv，适用于CIFAR数据集的卷积神经网络）")

    parser.add_argument('--model', type=str, default='cnn4conv', choices=['cnn4conv', 'cnn_mnist'],
                        help="模型结构（cnn_mnist适用手写体，cnn4conv适用CIFAR）26320##")
    # 4. 通用基础参数
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100','mnist'],
                        help="使用的数据集（默认cifar10，可选cifar100,或mnist）")
    parser.add_argument('--iid', action='store_true',
                        help="是否使用IID数据划分（默认非IID，添加该参数则为IID）")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="数据集的类别数（cifar10和mnist默认10，cifar100需设为100）")

    # parser.add_argument('--num_channels', type=int, default=3,
    #                     help="图像的通道数（默认3，对应RGB彩色图像）")
    parser.add_argument('--num_channels', type=int, default=3,
                        help="图像的通道数 （ mnist设为1，cifar设为3）26320##")

    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU设备编号（默认0，-1表示使用CPU）")
    parser.add_argument('--verbose', action='store_true',
                        help="是否打印详细日志（默认不打印）")
    parser.add_argument('--seed', type=int, default=1,
                        help="随机种子（默认1，保证实验可复现）")
    parser.add_argument('--all_clients', action='store_true',
                        help="是否聚合所有用户的模型（默认仅聚合本轮参与的用户）")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="数据加载的线程数（默认4，加速数据读取）26320##默认4")

    # 5. 标签噪声相关参数
    parser.add_argument('--noise_type_lst', nargs='+', default=['symmetric'],
                        help="噪声类型列表（支持symmetric(对称噪声)和pairflip(成对噪声)）")
    parser.add_argument('--noise_group_num', nargs='+', default=[100], type=int,
                        help="每组噪声对应的用户数量默认100（总和需等于num_users，如[50,50]表示两组各50用户）")
    parser.add_argument('--group_noise_rate', nargs='+', default=[0,0.4], type=float,
                        help="每组噪声率的范围，格式为[min1,max1,min2,max2...]")
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help="热身轮次（FedRN算法中前100轮不进行邻居协作，与原论文一致）")

    # 6. 其他抗噪声算法参数（SELFIE、Co-teaching等）
    parser.add_argument('--queue_size', type=int, default=15,
                        help="SELFIE算法的历史队列大小")
    parser.add_argument('--forget_rate', type=float, default=0.2,
                        help="coteaching', 'coteaching+', 'dividemix'算法的遗忘率")
    parser.add_argument('--uncertainty_threshold', type=float, default=0.05,
                        help="SELFIE算法的不确定性阈值")
    parser.add_argument('--alpha', type=float, default=1.2,
                        help="Joint optimization算法的alpha系数")
    parser.add_argument('--beta', type=float, default=0.8,
                        help="Joint optimization算法的beta系数")
    parser.add_argument('--labeling', type=str, default='soft',
                        help="Joint optimization算法的标签类型（soft/hard）")
    parser.add_argument('--mm_alpha', default=4, type=float,
                        help="MixMatch算法的Beta分布参数")
    parser.add_argument('--lambda_u', default=25, type=float,
                        help="MixMatch算法的无监督损失权重")
    parser.add_argument('--T', default=0.5, type=float,
                        help="MixMatch算法的温度参数")
    parser.add_argument('--p_threshold', default=0.5, type=float,
                        help="MixMatch算法的置信度阈值")

    # 7. FedRN算法专属参数（原论文核心参数
    parser.add_argument('--num_neighbors', type=int, default=2,
                        help="FedRN选择的可靠邻居数量（默认2，与原论文最优设置一致）")
    parser.add_argument('--w_alpha', type=float, default=0.5,
                        help="FedRN中专业性与相似度的权重系数（0.5表示两者同等重要）")

    # 8：边缘服务器数量配置
    parser.add_argument('--num_edges', type=int, default=5,
                        help="边缘服务器的数量（默认5个）")
    # 🚀 新增：FedER 专属超参数，用于控制专业性(exp)和相似度(sim)的权重
    parser.add_argument('--feder_exp_weight', type=float, default=0.6,
                        help='FedER聚合时专业性(exp)的权重占比。相似度(sim)的权重将自动设为 1 - feder_exp_weight')
    args = parser.parse_args()
    return args




# # !/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Python version: 3.6
#
# import argparse
#
#
# def args_parser():
#     # 创建参数解析器，用于解析命令行输入的参数
#     parser = argparse.ArgumentParser()
#
#     # 1. 算法选择参数（指定使用的抗噪声或联邦学习方法）
#     parser.add_argument('--method', type=str, default='fedrn',
#                         choices=['default', 'selfie', 'jointoptim', 'coteaching', 'coteaching+', 'dividemix', 'fedrn'],
#                         help='选择训练方法：default(默认，对应FedAvg)、fedrn(原论文方法)等抗噪声算法')
#
#     # 2. 联邦学习核心参数（控制联邦训练流程）
#     parser.add_argument('--epochs', type=int, default=200,
#                         help="联邦学习总通信轮次（默认500轮）")
#     parser.add_argument('--num_users', type=int, default=100,
#                         help="用户总数K（默认100个用户）")
#     parser.add_argument('--frac', type=float, default=0.1,
#                         help="每轮参与训练的用户比例C（默认10%）")
#     parser.add_argument('--local_ep', type=int, default=4,
#                         help="每个用户的本地训练轮次E（默认5轮）")
#     parser.add_argument('--local_bs', type=int, default=50,
#                         help="用户本地训练的批次大小B")
#     parser.add_argument('--bs', type=int, default=128,
#                         help="测试集的批次大小")
#     parser.add_argument('--lr', type=float, default=0.01,
#                         help="初始学习率（默认0.01）")
#     parser.add_argument('--momentum', type=float, default=0.5,
#                         help="SGD优化器的动量参数（默认0.5）")
#     parser.add_argument('--split', type=str, default='user',
#                         help="训练集与测试集的划分方式：user(按用户)或sample(按样本)")
#     parser.add_argument('--schedule', nargs='+', default=[],
#                         help="学习率衰减的轮次列表（如[200, 300]表示在200和300轮衰减）")
#     parser.add_argument('--lr_decay', type=float, default=0.1,
#                         help="学习率衰减系数（默认衰减为原来的0.1倍）")
#     parser.add_argument('--weight_decay', type=float, default=0,
#                         help="SGD的权重衰减（默认0，即不使用）")
#     parser.add_argument('--partition', type=str, choices=['shard', 'dirichlet'], default='shard',
#                         help="非IID数据划分方式：shard(分片划分)或dirichlet(狄利克雷分布划分)")
#     parser.add_argument('--dd_alpha', type=float, default=0.5,
#                         help="狄利克雷划分的浓度参数alpha（值越小，数据异质性越强）")
#     parser.add_argument('--num_shards', type=int, default=200,
#                         help="分片划分的总分片数（默认200，100用户各分2片）")
#     parser.add_argument('--fed_method', type=str, default='fedavg', choices=['fedavg'],
#                         help="联邦学习聚合方法（当前仅支持fedavg，即联邦平均）")
#
#     # 3. 模型相关参数
#     # parser.add_argument('--model', type=str, default='cnn4conv', choices=['cnn4conv'],
#     #                     help="模型结构（默认cnn4conv，适用于CIFAR数据集的卷积神经网络）")
#
#     parser.add_argument('--model', type=str, default='cnn_mnist', choices=['cnn4conv', 'cnn_mnist'],
#                         help="模型结构（cnn_mnist适用手写体，cnn4conv适用CIFAR）26320##")
#     # 4. 通用基础参数
#     parser.add_argument('--dataset', type=str, default='mnist',
#                         choices=['cifar10', 'cifar100','mnist'],
#                         help="使用的数据集（默认cifar10，可选cifar100,或mnist）")
#     parser.add_argument('--iid', action='store_true',
#                         help="是否使用IID数据划分（默认非IID，添加该参数则为IID）")
#     parser.add_argument('--num_classes', type=int, default=10,
#                         help="数据集的类别数（cifar10和mnist默认10，cifar100需设为100）")
#
#     # parser.add_argument('--num_channels', type=int, default=3,
#     #                     help="图像的通道数（默认3，对应RGB彩色图像）")
#     parser.add_argument('--num_channels', type=int, default=1,
#                         help="图像的通道数 （ mnist设为1，cifar设为3）26320##")
#
#     parser.add_argument('--gpu', type=int, default=0,
#                         help="GPU设备编号（默认0，-1表示使用CPU）")
#     parser.add_argument('--verbose', action='store_true',
#                         help="是否打印详细日志（默认不打印）")
#     parser.add_argument('--seed', type=int, default=1,
#                         help="随机种子（默认1，保证实验可复现）")
#     parser.add_argument('--all_clients', action='store_true',
#                         help="是否聚合所有用户的模型（默认仅聚合本轮参与的用户）")
#     parser.add_argument('--num_workers', type=int, default=0,
#                         help="数据加载的线程数（默认4，加速数据读取）26320##默认4")
#
#     # 5. 标签噪声相关参数
#     parser.add_argument('--noise_type_lst', nargs='+', default=['symmetric'],
#                         help="噪声类型列表（支持symmetric(对称噪声)和pairflip(成对噪声)）")
#     parser.add_argument('--noise_group_num', nargs='+', default=[100], type=int,
#                         help="每组噪声对应的用户数量默认100（总和需等于num_users，如[50,50]表示两组各50用户）")
#     parser.add_argument('--group_noise_rate', nargs='+', default=[0,0.8], type=float,
#                         help="每组噪声率的范围，格式为[min1,max1,min2,max2...]")
#     parser.add_argument('--warmup_epochs', type=int, default=60,
#                         help="热身轮次（FedRN算法中前100轮不进行邻居协作，与原论文一致）")
#
#     # 6. 其他抗噪声算法参数（SELFIE、Co-teaching等）
#     parser.add_argument('--queue_size', type=int, default=15,
#                         help="SELFIE算法的历史队列大小")
#     parser.add_argument('--forget_rate', type=float, default=0.2,
#                         help="Co-teaching算法的遗忘率")
#     parser.add_argument('--uncertainty_threshold', type=float, default=0.05,
#                         help="SELFIE算法的不确定性阈值")
#     parser.add_argument('--alpha', type=float, default=1.2,
#                         help="Joint optimization算法的alpha系数")
#     parser.add_argument('--beta', type=float, default=0.8,
#                         help="Joint optimization算法的beta系数")
#     parser.add_argument('--labeling', type=str, default='soft',
#                         help="Joint optimization算法的标签类型（soft/hard）")
#     parser.add_argument('--mm_alpha', default=4, type=float,
#                         help="MixMatch算法的Beta分布参数")
#     parser.add_argument('--lambda_u', default=25, type=float,
#                         help="MixMatch算法的无监督损失权重")
#     parser.add_argument('--T', default=0.5, type=float,
#                         help="MixMatch算法的温度参数")
#     parser.add_argument('--p_threshold', default=0.5, type=float,
#                         help="MixMatch算法的置信度阈值")
#
#     # # 7. FedRN算法专属参数（原论文核心参数
#     # parser.add_argument('--num_neighbors', type=int, default=2,
#     #                     help="FedRN选择的可靠邻居数量（默认2，与原论文最优设置一致）")
#     # parser.add_argument('--w_alpha', type=float, default=0.5,
#     #                     help="FedRN中专业性与相似度的权重系数（0.5表示两者同等重要）")
#
#     # 8：边缘服务器数量配置
#     parser.add_argument('--num_edges', type=int, default=5,
#                         help="边缘服务器的数量（默认5个）")
#     args = parser.parse_args()
#     return args
#
