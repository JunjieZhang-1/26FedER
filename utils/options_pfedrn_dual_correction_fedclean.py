#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Options for PFedRN Dual-Correction FedClean-style method.

Recommended with:
    main_pfedrn_dual_correction_fedclean_style_persistent.py

Method idea:
- net1: FedRN/global upload branch. It produces S_core with reliable-neighbor selection.
- net2: local CNLL-style correction branch. It confirms S_recover and corrects S_corr.
- Aggregation is unchanged: only net1 is uploaded, sample-size FedAvg.
"""

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------------
    # Method / experiment name
    # ------------------------------------------------------------------
    parser.add_argument(
        '--method', type=str, default='pfedrn_dual_correction_fedclean',
        choices=[
            'default', 'selfie', 'jointoptim', 'coteaching', 'coteaching+', 'dividemix',
            'fedrn', 'feder', 'fedrnn', 'fedco', 'fedcoPFL', 'pfedrn',
            'fedrn_t1pr', 'pfedrn_t1pr_three', 'pfedrn_dual_correction_fedclean'
        ],
        help='method name; main script will force pfedrn_dual_correction_fedclean'
    )

    # ------------------------------------------------------------------
    # Federated learning core arguments
    # ------------------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=500, help='communication rounds')
    parser.add_argument('--num_users', type=int, default=100, help='number of clients')
    parser.add_argument('--frac', type=float, default=0.5, help='fraction of clients per round')
    parser.add_argument('--local_ep', type=int, default=5, help='local epochs for net1/main branch')
    parser.add_argument('--local_bs', type=int, default=50, help='local batch size')
    parser.add_argument('--bs', type=int, default=128, help='test/logging batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='SGD weight decay')
    parser.add_argument('--split', type=str, default='user', help='train-test split type')
    parser.add_argument('--schedule', nargs='+', default=[], help='LR decay rounds')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='LR decay factor')
    parser.add_argument('--fed_method', type=str, default='fedavg', choices=['fedavg'], help='aggregation method')

    # ------------------------------------------------------------------
    # Data partition
    # ------------------------------------------------------------------
    parser.add_argument('--partition', type=str, choices=['shard', 'dirichlet'], default='shard')
    parser.add_argument('--dd_alpha', type=float, default=0.5, help='Dirichlet alpha')
    parser.add_argument('--num_shards', type=int, default=200, help='shards for non-IID shard split')
    parser.add_argument('--iid', action='store_true', help='use IID split')

    # ------------------------------------------------------------------
    # Model / dataset
    # ------------------------------------------------------------------
    parser.add_argument('--model', type=str, default='cnn4conv',
                        choices=['cnn4conv', 'cnn_mnist', 'CNN4Conv_DualHead'], help='model name')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist'], help='dataset')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--num_channels', type=int, default=3, help='input channels')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id; -1 for CPU')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--all_clients', action='store_true', help='aggregate all clients')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')

    # ------------------------------------------------------------------
    # Label noise
    # ------------------------------------------------------------------
    parser.add_argument('--noise_type_lst', nargs='+', default=['symmetric'], help='noise types')
    parser.add_argument('--noise_group_num', nargs='+', default=[100], type=int,
                        help='number of clients in each noise group')
    parser.add_argument('--group_noise_rate', nargs='+', default=[0, 0.8], type=float,
                        help='noise rate range, e.g. 0 0.8')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='warmup rounds')

    # ------------------------------------------------------------------
    # Legacy robust-learning arguments kept for compatibility
    # ------------------------------------------------------------------
    parser.add_argument('--queue_size', type=int, default=15, help='SELFIE history queue size')
    parser.add_argument('--forget_rate', type=float, default=0.2, help='forget rate')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.05, help='SELFIE uncertainty threshold')
    parser.add_argument('--alpha', type=float, default=1.2, help='joint optimization alpha')
    parser.add_argument('--beta', type=float, default=0.8, help='joint optimization beta')
    parser.add_argument('--labeling', type=str, default='soft', choices=['soft', 'hard'], help='labeling type')
    parser.add_argument('--mm_alpha', type=float, default=4.0, help='MixMatch alpha')
    parser.add_argument('--lambda_u', type=float, default=25.0, help='unsupervised loss weight')
    parser.add_argument('--T', type=float, default=0.5, help='temperature')
    parser.add_argument('--p_threshold', type=float, default=0.5, help='FedRN clean probability threshold')

    # ------------------------------------------------------------------
    # FedRN reliable-neighbor selection
    # ------------------------------------------------------------------
    parser.add_argument('--num_neighbors', type=int, default=2, help='number of reliable neighbors')
    parser.add_argument('--w_alpha', type=float, default=0.6,
                        help='neighbor score weight: w_alpha*expertise + (1-w_alpha)*similarity')

    # ------------------------------------------------------------------
    # Edge / neighbor scope. Current dual-correction script forces single-edge/global scope.
    # These arguments are kept for command compatibility.
    # ------------------------------------------------------------------
    parser.add_argument('--num_edges', type=int, default=1, help='number of edge servers')
    parser.add_argument('--neighbor_scope', type=str, default='global', choices=['edge', 'global'],
                        help='neighbor search scope')
    parser.add_argument('--feder_exp_weight', type=float, default=0.6, help='FedER compatibility parameter')

    # ------------------------------------------------------------------
    # PFL / old T1PR compatibility parameters
    # ------------------------------------------------------------------
    parser.add_argument('--pfl_local_weight', type=float, default=0.3, help='legacy PFL local weight')
    parser.add_argument('--pfl_agree_weight', type=float, default=0.0, help='legacy agreement weight; not used here')
    parser.add_argument('--pfl_personal_ep', type=int, default=1, help='legacy PFL local epochs')

    # ------------------------------------------------------------------
    # Dual-model correction training
    # ------------------------------------------------------------------
    parser.add_argument('--recover_ep', type=int, default=4,
                        help='local epochs for net2/local correction branch')
    parser.add_argument('--t1pr_mutual_weight', type=float, default=0.05,
                        help='KL mutual regularization weight between net1 and net2')
    parser.add_argument('--t1pr_mutual_temp', type=float, default=1.0,
                        help='temperature for KL mutual regularization')
    parser.add_argument('--correction_use_mutual', type=int, default=1, choices=[0, 1],
                        help='whether to use net1-net2 mutual KL regularization')

    # ------------------------------------------------------------------
    # FedClean-style label confirmation/correction
    # ------------------------------------------------------------------
    parser.add_argument('--correction_interval', type=int, default=5,
                        help='run correction every N rounds after warmup')
    parser.add_argument('--correction_conf_global', type=float, default=0.70,
                        help='minimum softmax confidence of net1/global prediction')
    parser.add_argument('--correction_conf_local', type=float, default=0.70,
                        help='minimum softmax confidence of net2/local correction prediction')
    parser.add_argument('--correction_min_rn_prob', type=float, default=0.20,
                        help='minimum p_rn for a noisy-pool sample to be considered for correction')
    parser.add_argument('--correction_max_ratio', type=float, default=0.10,
                        help='max ratio of noisy-pool samples corrected/recovered per selected client')
    parser.add_argument('--net1_use_corrected', type=int, default=0, choices=[0, 1],
                        help='0: net1 trains on S_core+S_recover; 1: also use S_corr in same round')
    parser.add_argument('--persistent_correction', type=int, default=1, choices=[0, 1],
                        help='1: write S_corr labels back to dataset_train permanently')

    args = parser.parse_args()
    return args
