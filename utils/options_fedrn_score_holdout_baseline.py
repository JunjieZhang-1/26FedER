#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="FedRN score集训练net2并用本地干净holdout验证的诊断实验。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # =========================
    # 实验标识与结果保存路径
    # =========================
    parser.add_argument("--method", type=str, default="fedrn", choices=["fedrn"], help="固定使用原版FedRN流程。")
    parser.add_argument("--exp_method", type=str, default="FedRN-ScoreHoldoutBaseline", help="实验名称。")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="resultdate/FedRN_ScoreHoldoutBaseline2612",
        help="所有日志、score快照、net2评估结果的保存目录。",
    )

    # =========================
    # 联邦训练参数：net1/FedRN主模型
    # =========================
    parser.add_argument("--epochs", type=int, default=500, help="联邦训练总轮数。")
    parser.add_argument("--num_users", type=int, default=100, help="客户端数量。")
    parser.add_argument("--frac", type=float, default=0.5, help="每轮参与训练的客户端比例。")
    parser.add_argument("--local_ep", type=int, default=5, help="net1每个客户端本地训练epoch数。")
    parser.add_argument("--local_bs", type=int, default=50, help="本地训练batch size。")
    parser.add_argument("--bs", type=int, default=128, help="测试/评估batch size。")
    parser.add_argument("--lr", type=float, default=0.01, help="net1/FedRN训练学习率。")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD动量。")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="net1/FedRN权重衰减。")
    parser.add_argument("--schedule", nargs="*", default=[], help="在哪些轮次降低学习率，例如 --schedule 250 375。")
    parser.add_argument("--lr_decay", type=float, default=0.1, help="学习率衰减倍率。")
    parser.add_argument("--fed_method", type=str, default="fedavg", choices=["fedavg"], help="聚合方法。")
    parser.add_argument("--all_clients", action="store_true", help="是否每轮聚合所有客户端。一般不启用。")

    # =========================
    # 数据划分参数
    # =========================
    parser.add_argument("--partition", type=str, choices=["shard", "dirichlet"], default="shard", help="客户端数据划分方式。")
    parser.add_argument("--num_shards", type=int, default=200, help="shard划分时的总shard数量。")
    parser.add_argument("--dd_alpha", type=float, default=0.5, help="dirichlet划分的alpha参数。")
    parser.add_argument("--iid", action="store_true", help="是否使用IID划分。")
    parser.add_argument("--split", type=str, default="user", help="兼容原工程参数，一般保持默认。")

    # =========================
    # 本地80/20干净划分
    # 关键点：先在无噪声数据上切分，再只对80%训练集加噪声。
    # =========================
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="每个客户端本地数据中用于net1/FedRN训练的比例；剩下部分作为干净holdout测试net2。",
    )
    parser.add_argument(
        "--holdout_eval_every",
        type=int,
        default=1,
        help="每隔多少轮评估一次20%干净holdout；设为0可关闭。",
    )
    parser.add_argument(
        "--client_holdout_detail",
        type=str,
        default="selected",
        choices=["selected", "none"],
        help="详细日志是否记录本轮选中客户端的holdout准确率。",
    )
    parser.add_argument(
        "--log_client_stats",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否在详细日志中记录每个被选中客户端的score纯度、召回、漏选等信息。",
    )

    # =========================
    # FedRN score快照导出
    # 用于观察FedRN给出的score clean set本身质量。
    # =========================
    parser.add_argument(
        "--save_score_snapshots",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否在指定轮次导出FedRN score判断结果CSV。",
    )
    parser.add_argument(
        "--snapshot_rounds",
        nargs="*",
        type=int,
        default=[100, 150],
        help="在哪些轮次导出score快照。注意轮次从0开始，100表示Round 100。",
    )
    parser.add_argument(
        "--snapshot_with_neighbors",
        type=int,
        default=1,
        choices=[0, 1],
        help="导出score时是否使用FedRN邻居融合后的最终clean判断；建议保持1。",
    )

    # =========================
    # net2持续训练实验
    # 目的：warmup结束后，net2每轮都用当前FedRN score集训练，并在本地20%干净holdout上评估。
    # net2只做本地模型，不上传、不聚合、不影响net1。
    # =========================
    parser.add_argument(
        "--train_net2_after_warmup",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否在warmup结束后每轮训练本地net2。1表示开启，这是当前主实验逻辑。",
    )
    parser.add_argument(
        "--net2_local_ep",
        type=int,
        default=1,
        help="warmup后每一轮net2在FedRN score集上的本地训练epoch数。",
    )
    parser.add_argument("--net2_lr", type=float, default=0.01, help="持续训练net2的学习率。")
    parser.add_argument("--net2_weight_decay", type=float, default=0.0, help="持续训练net2的权重衰减。")
    parser.add_argument(
        "--net2_init",
        type=str,
        default="global",
        choices=["global", "local_net1", "random"],
        help="net2首次初始化方式：global=当前全局模型；local_net1=客户端net1；random=随机初始化。",
    )
    parser.add_argument(
        "--net2_eval_every",
        type=int,
        default=1,
        help="持续net2每隔多少轮记录一次holdout评估；1表示每轮记录。",
    )

    # 额外快照诊断：默认关闭，避免和持续net2混淆。
    parser.add_argument(
        "--run_net2_score_eval",
        type=int,
        default=0,
        choices=[0, 1],
        help="是否额外启用快照式net2诊断。默认关闭。",
    )
    parser.add_argument(
        "--net2_eval_rounds",
        nargs="*",
        type=int,
        default=[100, 150],
        help="在哪些轮次运行net2诊断实验。",
    )
    parser.add_argument(
        "--net2_eval_ep",
        type=int,
        default=20,
        help="net2在FedRN score集上的本地训练epoch数。",
    )
    parser.add_argument(
        "--net2_eval_lr",
        type=float,
        default=0.01,
        help="net2诊断训练学习率。",
    )
    parser.add_argument(
        "--net2_eval_weight_decay",
        type=float,
        default=0.0,
        help="net2诊断训练权重衰减。",
    )
    parser.add_argument(
        "--net2_eval_init",
        type=str,
        default="global",
        choices=["global", "local_net1", "random"],
        help="net2初始化方式：global=当前全局模型；local_net1=客户端本地net1；random=随机初始化。",
    )
    parser.add_argument(
        "--net2_eval_max_clients",
        type=int,
        default=0,
        help="每次net2诊断最多评估多少客户端；0表示全部客户端。调试时可设为5或10。",
    )

    # =========================
    # 模型、数据集与设备
    # =========================
    parser.add_argument("--model", type=str, default="cnn4conv", choices=["cnn4conv"], help="模型结构。")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="数据集。")
    parser.add_argument("--num_classes", type=int, default=10, help="类别数，加载数据集后会自动覆盖。")
    parser.add_argument("--num_channels", type=int, default=3, help="输入通道数。")
    parser.add_argument("--gpu", type=int, default=0, help="GPU编号；设为-1使用CPU。")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader线程数。")
    parser.add_argument("--verbose", action="store_true", help="是否输出更详细训练信息。")
    parser.add_argument("--seed", type=int, default=1, help="随机种子。")

    # =========================
    # 标签噪声设置
    # 当前默认是100个客户端噪声率从0线性增加到0.8。
    # =========================
    parser.add_argument("--noise_type_lst", nargs="+", default=["symmetric"], help="噪声类型，例如 symmetric 或 pairflip。")
    parser.add_argument("--noise_group_num", nargs="+", default=[100], type=int, help="每个噪声组包含的客户端数量。")
    parser.add_argument(
        "--group_noise_rate",
        nargs="+",
        default=[0, 0.8],
        type=float,
        help="每个噪声组的最小/最大噪声率，例如 0 0.8。",
    )
    parser.add_argument(
        "--noise_seed_mode",
        type=str,
        default="fedrn",
        choices=["fedrn", "client"],
        help="加噪声时的随机种子模式：fedrn复现原FedRN写法；client每个客户端不同种子。",
    )
    parser.add_argument("--warmup_epochs", type=int, default=100, help="FedRN预热轮数；预热阶段不筛score集。")

    # =========================
    # FedRN核心参数
    # =========================
    parser.add_argument("--p_threshold", type=float, default=0.5, help="GMM clean概率阈值，大于该值认为干净。")
    parser.add_argument("--num_neighbors", type=int, default=2, help="FedRN邻居辅助模型数量。")
    parser.add_argument("--w_alpha", type=float, default=0.6, help="FedRN中expertise和similarity融合权重。")

    # =========================
    # 兼容原工程其他方法的参数
    # 本脚本不主动使用，但部分共享代码可能会读取这些字段。
    # =========================
    parser.add_argument("--queue_size", type=int, default=15, help="兼容参数。")
    parser.add_argument("--forget_rate", type=float, default=0.2, help="兼容参数。")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.05, help="兼容参数。")
    parser.add_argument("--alpha", type=float, default=1.2, help="兼容参数。")
    parser.add_argument("--beta", type=float, default=0.8, help="兼容参数。")
    parser.add_argument("--labeling", type=str, default="soft", help="兼容参数。")
    parser.add_argument("--mm_alpha", type=float, default=4, help="兼容参数。")
    parser.add_argument("--lambda_u", type=float, default=25, help="兼容参数。")
    parser.add_argument("--T", type=float, default=0.5, help="兼容参数。")

    return parser.parse_args()
