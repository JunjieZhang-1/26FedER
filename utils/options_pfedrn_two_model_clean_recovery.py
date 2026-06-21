#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Options for main_pfedrn_two_model_clean_recovery.py.

PFedRN-2M-CCR / Two-model clean recovery:
- net1: FedRN main/global model, uploaded and aggregated.
- net3: personalized local model, kept locally and used only for conservative recovery.

原则：
1. main 文件不再写死实验超参数；method、exp_method、num_neighbors、num_edges、neighbor_scope 等都从这里或命令行读取。
2. 这里只保留当前 main 文件需要的参数，以及少量 shared-code 兼容参数。
3. recover_max_ratio=0 可退化为“只跑 FedRN 主筛选分支，不执行恢复”。
"""

import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="Option parser for PFedRN-2M-CCR / two-model clean recovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # 1. Experiment identity
    # ------------------------------------------------------------------
    parser.add_argument(
        '--method',
        type=str,
        default='pfedrn_two_model_clean_recovery',
        help='训练方法名；当前 main 推荐使用 pfedrn_two_model_clean_recovery',
    )
    parser.add_argument(
        '--exp_method',
        type=str,
        default='PFedRN-2M-CCR',
        help='实验显示名/日志目录名，例如 PFedRN-2M-CCR',
    )

    # ------------------------------------------------------------------
    # 2. Federated training settings
    # ------------------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=500, help='联邦训练总轮次')
    parser.add_argument('--num_users', type=int, default=100, help='客户端总数')
    parser.add_argument('--frac', type=float, default=0.5, help='每轮参与客户端比例')
    parser.add_argument('--local_ep', type=int, default=5, help='net1 每轮本地训练 epoch 数')
    parser.add_argument('--local_bs', type=int, default=50, help='本地训练 batch size')
    parser.add_argument('--bs', type=int, default=128, help='测试/评估 batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='SGD 学习率')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='SGD weight decay')
    parser.add_argument('--schedule', nargs='*', default=[], help='学习率衰减轮次，例如 --schedule 250 400')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='学习率衰减系数')
    parser.add_argument('--fed_method', type=str, default='fedavg', choices=['fedavg'], help='聚合方法')
    parser.add_argument('--all_clients', action='store_true', help='保留兼容参数；当前 main 默认按 frac 采样')

    # ------------------------------------------------------------------
    # 3. Data partition settings
    # ------------------------------------------------------------------
    parser.add_argument('--partition', type=str, choices=['shard', 'dirichlet'], default='shard',
                        help='非 IID 划分方式')
    parser.add_argument('--num_shards', type=int, default=200, help='shard 总数；100客户端时200表示每客户端约2个 shard')
    parser.add_argument('--dd_alpha', type=float, default=0.5, help='Dirichlet alpha')
    parser.add_argument('--iid', action='store_true', help='是否使用 IID 划分')
    parser.add_argument('--split', type=str, default='user', help='保留兼容参数；当前 main 使用用户划分')

    # ------------------------------------------------------------------
    # 4. Model / dataset / device
    # ------------------------------------------------------------------
    parser.add_argument(
        '--model',
        type=str,
        default='cnn4conv',
        choices=['cnn4conv', 'cnn_mnist', 'CNN4Conv_DualHead'],
        help='模型结构',
    )
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'],
                        help='数据集')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数；main 会根据数据集再次更新')
    parser.add_argument('--num_channels', type=int, default=3, help='输入通道数；MNIST=1，CIFAR=3')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID；-1 表示 CPU')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader num_workers')
    parser.add_argument('--verbose', action='store_true', help='是否打印更详细信息')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')

    # ------------------------------------------------------------------
    # 5. Label-noise settings
    # ------------------------------------------------------------------
    parser.add_argument('--noise_type_lst', nargs='+', default=['symmetric'], help='噪声类型列表')
    parser.add_argument('--noise_group_num', nargs='+', default=[100], type=int,
                        help='每组噪声对应客户端数量；总和需等于 num_users')
    parser.add_argument('--group_noise_rate', nargs='+', default=[0, 0.8], type=float,
                        help='噪声率范围，例如 --group_noise_rate 0 0.8')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='FedRN warmup 轮次')

    # ------------------------------------------------------------------
    # 6. FedRN main branch
    # ------------------------------------------------------------------
    parser.add_argument('--num_neighbors', type=int, default=1, help='FedRN 邻居数量；main 中按该值选择 Top-K 邻居')
    parser.add_argument('--w_alpha', type=float, default=0.6, help='FedRN 专业性/相似度融合权重')
    parser.add_argument('--p_threshold', type=float, default=0.5, help='FedRN clean probability 阈值')
    parser.add_argument('--neighbor_scope', type=str, default='global', choices=['edge', 'global'], help='邻居搜索范围')
    parser.add_argument('--num_edges', type=int, default=1, help='边缘服务器数量；当前对齐实验建议为 1')
    parser.add_argument('--send_2_models', action='store_true', default=False,
                        help='兼容旧代码；当前两模型版本只上传 net1，默认 False')

    # ------------------------------------------------------------------
    # 7. Two-model clean recovery / PFedRN-2M-CCR
    # ------------------------------------------------------------------
    parser.add_argument('--recover_interval', type=int, default=1,
                        help='每隔多少轮执行一次 clean recovery；1表示每轮执行')
    parser.add_argument('--recover_max_ratio', type=float, default=0.05,
                        help='每客户端每轮最多从 FedRN noisy pool 放回的比例；设为0可关闭恢复')
    parser.add_argument('--recover_conf_main', type=float, default=0.7,
                        help='恢复条件：net1 对当前训练标签的最低置信度')
    parser.add_argument('--recover_conf_personal', type=float, default=0.7,
                        help='恢复条件：net3 对当前训练标签的最低置信度')
    parser.add_argument('--recover_min_p_rn', type=float, default=0.2,
                        help='恢复条件：FedRN p_rn 最低要求，避免恢复被主分支强烈否定的样本')
    parser.add_argument('--recover_memory_threshold', type=int, default=1,
                        help='样本连续/累计满足恢复条件达到该次数才放回')
    parser.add_argument('--recover_memory_decay', type=int, default=1, help='未满足条件时 recovery_memory 每轮衰减量')
    parser.add_argument('--personal_train_mode', type=str, default='core', choices=['core', 'core_recover', 'all'],
                        help='net3 训练数据来源；推荐 core')
    parser.add_argument('--pfl_personal_ep', type=int, default=1,
                        help='net3 每轮本地训练 epoch 数；设为0可关闭 net3 训练')
    parser.add_argument('--personal_distill_weight', type=float, default=0.0,
                        help='net3 从 net1 蒸馏的 KL 权重；默认关闭')
    parser.add_argument('--warmup_train_personal', type=int, default=0, choices=[0, 1],
                        help='warmup 阶段是否训练 net3；推荐0')
    parser.add_argument('--reset_net3_after_warmup', type=int, default=1, choices=[0, 1],
                        help='warmup结束后是否用当前 net1 重置 net3；推荐1')

    # ------------------------------------------------------------------
    # 8. Shared-code compatibility parameters
    # ------------------------------------------------------------------
    # LocalUpdatePFedRN / shared baselines may read these names. 当前 main 不主动使用这些算法分支，保留是为了避免 AttributeError。
    parser.add_argument('--queue_size', type=int, default=15)
    parser.add_argument('--forget_rate', type=float, default=0.2)
    parser.add_argument('--uncertainty_threshold', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=1.2)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--labeling', type=str, default='soft')
    parser.add_argument('--mm_alpha', type=float, default=4)
    parser.add_argument('--lambda_u', type=float, default=25)
    parser.add_argument('--T', type=float, default=0.5)

    return parser.parse_args()
