#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Options for PFedRN-Net2BootstrapOracle.

This is an oracle diagnostic experiment:
  - split every client before label noise: 80% train, 20% clean holdout.
  - train local net2 early and choose its best checkpoint by clean holdout.
  - use best net2 to select clean samples for net1 bootstrap.
  - switch back to original FedRN after fedrn_start_epoch.

Because clean holdout labels are used to pick best net2, this is an upper-bound
experiment first. If it works, replace holdout selection with an unsupervised
criterion later.
"""

import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="PFedRN Net2 Bootstrap Oracle"
    )

    # Federated learning.
    parser.add_argument('--method', type=str, default='pfedrn', choices=['pfedrn'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--num_users', type=int, default=100)
    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--local_bs', type=int, default=50)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--schedule', nargs='+', type=int, default=[])
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--fed_method', type=str, default='fedavg', choices=['fedavg'])

    # Data partition.
    parser.add_argument('--partition', type=str, default='shard',
                        choices=['shard', 'dirichlet'])
    parser.add_argument('--num_shards', type=int, default=200)
    parser.add_argument('--dd_alpha', type=float, default=0.5)
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--split', type=str, default='user')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Per-client train split ratio before adding noise.')

    # Model and dataset.
    parser.add_argument('--model', type=str, default='cnn4conv',
                        choices=['cnn4conv', 'cnn_mnist'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=32)

    # Runtime.
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--all_clients', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    # Label noise.
    parser.add_argument('--noise_type_lst', nargs='+', default=['symmetric'])
    parser.add_argument('--noise_group_num', nargs='+', default=[100], type=int)
    parser.add_argument('--group_noise_rate', nargs='+', default=[0, 0.8], type=float)
    parser.add_argument('--noise_seed_mode', type=str, default='fedrn',
                        choices=['fedrn', 'per_user', 'user', 'client'])

    # Original FedRN.
    parser.add_argument('--p_threshold', type=float, default=0.5)
    parser.add_argument('--num_neighbors', type=int, default=2)
    parser.add_argument('--w_alpha', type=float, default=0.6)
    parser.add_argument('--neighbor_ft_ep', type=int, default=1)

    # Net2 bootstrap oracle.
    parser.add_argument('--net2_search_epochs', type=int, default=100,
                        help='Rounds used to train net2 and find best checkpoint.')
    parser.add_argument('--fedrn_start_epoch', type=int, default=100,
                        help='Round to switch back to original FedRN.')
    parser.add_argument('--net2_local_ep', type=int, default=5,
                        help='Local epochs for net2 each selected round.')
    parser.add_argument('--continue_train_net2_after_search', type=int, default=0,
                        help='0 keeps best net2 fixed after search; 1 keeps training net2.')

    # Compatibility parameters used by old scripts/models.
    parser.add_argument('--queue_size', type=int, default=15)
    parser.add_argument('--forget_rate', type=float, default=0.2)
    parser.add_argument('--uncertainty_threshold', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=1.2)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--labeling', type=str, default='soft')
    parser.add_argument('--mm_alpha', type=float, default=4)
    parser.add_argument('--lambda_u', type=float, default=25)
    parser.add_argument('--T', type=float, default=0.5)

    # Output.
    parser.add_argument('--exp_method', type=str, default='PFedRN-Net2BootstrapOracle')
    parser.add_argument('--save_dir', type=str, default='resultdate')
    parser.add_argument('--log_client_stats', type=int, default=1)

    return parser.parse_args()
