#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PFedRN PlanA 配置文件
====================

核心思路：
  - net1 = 全局模型（参与 FedAvg 聚合）
  - net2 = 本地个性化模型（不上传、不聚合）
  - 前 warmup_epochs 轮：只训练net1，net2不训练
  - warmup后：目标客户端 A 先用 netA1 得到 scoreA，A自己的net2只用scoreA训练
  - net2_start_epoch 之前：net2只训练，不参与概率判断，判断仍然等价于原FedRN
  - net2_start_epoch 之后：每个可靠邻居 B 的 netB1 和已训练的 netB2 都在 scoreA 上临时微调
  - 微调后的 netB1/netB2 一起辅助判断 A 的 clean/noisy

用法示例：
  python main_pfedrn_planA_neighbor_net2_ft.py \
    --dataset cifar10 --epochs 500 --num_users 100 --frac 0.1 \
    --local_ep 5 --local_bs 50 --num_shards 200 \
    --group_noise_rate 0 0.8 --noise_group_num 100 \
    --warmup_epochs 100 --num_neighbors 2 --w_alpha 0.6 \
    --neighbor_net1_weight 0.7 --neighbor_net2_weight 0.3 \
    --net2_start_epoch 200

作者注释：中文参数说明，方便实验调参
"""

import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="PFedRN PlanA Neighbor Net1/Net2 Fine-tuning"
    )

    # ================================================================
    # 1. 算法名称
    # ================================================================
    parser.add_argument('--method', type=str, default='pfedrn',
                        choices=['pfedrn'],
                        help='方法名称（固定为pfedrn）')

    # ================================================================
    # 2. 联邦学习核心参数
    # ================================================================
    parser.add_argument('--epochs', type=int, default=500,
                        help='联邦学习总通信轮次（默认500）')
    parser.add_argument('--num_users', type=int, default=100,
                        help='用户/客户端总数 K（默认100）')
    parser.add_argument('--frac', type=float, default=0.1,
                        help='每轮参与训练的用户比例 C（默认0.1，即10%%)')
    parser.add_argument('--local_ep', type=int, default=5,
                        help='每个用户的本地训练轮次 E（默认5）')
    parser.add_argument('--local_bs', type=int, default=50,
                        help='本地训练的 batch size（默认50）')
    parser.add_argument('--bs', type=int, default=128,
                        help='测试集的 batch size（默认128）')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='初始学习率（默认0.01）')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD 动量（默认0.5）')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='SGD 权重衰减（默认0）')
    parser.add_argument('--schedule', nargs='+', type=int, default=[],
                        help='学习率衰减轮次列表，如 300 400')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='学习率衰减系数（默认0.1）')
    parser.add_argument('--fed_method', type=str, default='fedavg',
                        choices=['fedavg'],
                        help='联邦聚合方法（仅支持fedavg）')

    # ================================================================
    # 3. 数据划分参数
    # ================================================================
    parser.add_argument('--partition', type=str, default='shard',
                        choices=['shard', 'dirichlet'],
                        help='非IID数据划分方式：shard(分片) 或 dirichlet(狄利克雷)')
    parser.add_argument('--num_shards', type=int, default=200,
                        help='分片划分的总分片数（默认200，即100用户各2片）')
    parser.add_argument('--dd_alpha', type=float, default=0.5,
                        help='狄利克雷划分的alpha参数（越小异质性越强）')
    parser.add_argument('--iid', action='store_true',
                        help='是否使用IID划分（默认非IID）')
    parser.add_argument('--split', type=str, default='user',
                        help='训练测试集划分方式（默认user）')

    # ================================================================
    # 4. 模型参数
    # ================================================================
    parser.add_argument('--model', type=str, default='cnn4conv',
                        choices=['cnn4conv', 'cnn_mnist'],
                        help='模型结构：cnn4conv(CIFAR) 或 cnn_mnist')

    # ================================================================
    # 5. 数据集参数
    # ================================================================
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist'],
                        help='数据集名称')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='类别数（cifar10/mnist=10, cifar100=100）')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='图像通道数（cifar=3, mnist=1）')
    parser.add_argument('--img_size', type=int, default=32,
                        help='图像尺寸（cifar=32, mnist=28）')

    # ================================================================
    # 6. 硬件 / 运行参数
    # ================================================================
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU编号（-1表示CPU）')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子')
    parser.add_argument('--verbose', action='store_true',
                        help='是否打印详细日志')
    parser.add_argument('--all_clients', action='store_true',
                        help='是否聚合所有客户端（默认False只聚合选中客户端）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 加载线程数')

    # ================================================================
    # 7. 标签噪声参数
    # ================================================================
    parser.add_argument('--noise_type_lst', nargs='+', default=['symmetric'],
                        help='噪声类型：symmetric(对称) 或 pairflip(成对)')
    parser.add_argument('--noise_group_num', nargs='+', default=[100], type=int,
                        help='每组噪声的用户数，总和需等于num_users')
    parser.add_argument('--group_noise_rate', nargs='+', default=[0.4], type=float,
                        help='每组噪声率范围，格式 [min1, max1, min2, max2, ...]')
    parser.add_argument('--noise_seed_mode', type=str, default='fedrn',
                        help='噪声种子模式（默认fedrn）')

    # ================================================================
    # 8. FedRN 算法参数
    # ================================================================
    parser.add_argument('--warmup_epochs', type=int, default=100,
                        help='热身轮数（前N轮使用全部数据训练，不筛选）')
    parser.add_argument('--p_threshold', type=float, default=0.5,
                        help='GMM clean概率阈值（>=此值认为干净）')
    parser.add_argument('--num_neighbors', type=int, default=2,
                        help='FedRN选择的可靠邻居数量')
    parser.add_argument('--w_alpha', type=float, default=0.6,
                        help='FedRN中专业性(exp)与相似度(sim)的权重系数')
    parser.add_argument('--neighbor_ft_ep', type=int, default=1,
                        help='原FedRN邻居net1在目标scoreA上临时微调分类头的遍数；默认1，等价原版FedRN')
    parser.add_argument('--queue_size', type=int, default=15,
                        help='历史队列大小（保留兼容）')
    parser.add_argument('--forget_rate', type=float, default=0.2,
                        help='遗忘率（保留兼容）')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.05,
                        help='不确定性阈值（保留兼容）')
    parser.add_argument('--alpha', type=float, default=1.2,
                        help='JointOptim alpha（保留兼容）')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='JointOptim beta（保留兼容）')
    parser.add_argument('--labeling', type=str, default='soft',
                        help='标签类型（保留兼容）')
    parser.add_argument('--mm_alpha', type=float, default=4,
                        help='MixMatch alpha（保留兼容）')
    parser.add_argument('--lambda_u', type=float, default=25,
                        help='MixMatch无监督损失权重（保留兼容）')
    parser.add_argument('--T', type=float, default=0.5,
                        help='MixMatch温度（保留兼容）')

    # ================================================================
    # 9. PFedRN PlanA 专属参数
    # ================================================================
    parser.add_argument('--neighbor_net1_weight', type=float, default=0.7,
                        help='邻居netB1在邻居判断中的权重（默认0.7）')
    parser.add_argument('--neighbor_net2_weight', type=float, default=0.3,
                        help='邻居netB2在邻居判断中的权重（默认0.3）')
    parser.add_argument('--net2_local_ep', type=int, default=5,
                        help='net2在warmup后用net1初筛score训练的epoch数（默认5）')
    parser.add_argument('--net2_ft_ep', type=int, default=1,
                        help='邻居net2在目标scoreA上临时微调分类头的遍数；默认1，建议消融1/2/3')
    parser.add_argument('--net2_start_epoch', type=int, default=200,
                        help='net2开始参与概率判断的通信轮次；默认200，表示100轮warmup后先只训练net2 100轮，到第200轮再参与融合')

    # ================================================================
    # 10. 日志与保存参数
    # ================================================================
    parser.add_argument('--exp_method', type=str, default='PFedRN-PlanA-NeighborNet2FT',
                        help='实验名称前缀，用于日志目录命名')
    parser.add_argument('--save_dir', type=str, default='resultdate',
                        help='结果保存根目录')
    parser.add_argument('--log_client_stats', type=int, default=1,
                        help='是否在日志中输出每个客户端的详细统计')

    args = parser.parse_args()
    return args
