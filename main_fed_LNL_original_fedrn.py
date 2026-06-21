#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import random
import time
import os

import torchvision
import torch
from torch.utils.data import DataLoader

from utils import load_dataset
from utils.options_original_fedrn import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label

from models.fed import LocalModelWeights
from models.nets_original_fedrn import get_model
from models.test import test_img
from models.update_original_fedrn import get_local_update_objects


def append_line(path, text):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(str(text) + '\n')


def log_print(text, log_path=None):
    print(text)
    if log_path is not None:
        append_line(log_path, text)


def make_result_paths(args):
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join('resultdate', 'FedRN')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    noise_tag = str(args.group_noise_rate).replace(' ', '')
    exp_name = 'FedRN{}_{}_{}_{}'.format(
        args.epochs,
        args.dataset,
        args.local_ep,
        noise_tag,
    )

    detail_path = os.path.join(save_dir, '详细模型_{}_{}.txt'.format(exp_name, timestamp))
    metric_path = os.path.join(save_dir, '总体模型_{}_{}.txt'.format(exp_name, timestamp))
    return detail_path, metric_path, timestamp



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

    detail_log_path, metric_log_path, run_timestamp = make_result_paths(args)

    print('Results will be saved to: {}'.format(detail_log_path))
    print('Metrics ONLY will be saved to: {}'.format(metric_log_path))

    append_line(detail_log_path, 'Experiment Start: {}'.format(run_timestamp))
    append_line(detail_log_path, 'Args: {}'.format(args))
    append_line(detail_log_path, '=' * 80)
    append_line(detail_log_path, '')

    append_line(metric_log_path, 'Experiment Start: {}'.format(run_timestamp))
    append_line(metric_log_path, 'Args: {}'.format(args))
    append_line(metric_log_path, '=' * 80)
    append_line(metric_log_path, '')
    append_line(metric_log_path, 'round,train_acc,train_loss,test_acc,test_loss,local_update_loss,time_sec')

    for x in vars(args).items():
        log_print(x, detail_log_path)

    if not torch.cuda.is_available():
        exit('ERROR: Cuda is not available!')
    log_print('torch version: {}'.format(torch.__version__), detail_log_path)
    log_print('torchvision version: {}'.format(torchvision.__version__), detail_log_path)

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

            # for reproduction
            random.seed(args.seed)
            random.shuffle(data_indices)

            noise_index = int(len(data_indices) * user_noise_rate)

            for d_idx in data_indices[:noise_index]:
                true_label = dataset_train.train_labels[d_idx]
                noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
                dataset_train.train_labels[d_idx] = noisy_label

    # for logging purposes
    logging_args = dict(
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, **logging_args)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, **logging_args)

    ##############################
    # Build model
    ##############################
    net_glob = get_model(args)
    net_glob = net_glob.to(args.device)

    if args.send_2_models:
        net_glob2 = get_model(args)
        net_glob2 = net_glob2.to(args.device)

    ##############################
    # Training
    ##############################
    CosineSimilarity = torch.nn.CosineSimilarity()

    forget_rate_schedule = []
    if args.method in ['coteaching', 'coteaching+']:
        num_gradual = args.warmup_epochs
        forget_rate = args.forget_rate
        exponent = 1
        forget_rate_schedule = np.ones(args.epochs) * forget_rate
        forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

    pred_user_noise_rates = [args.forget_rate] * args.num_users

    # Initialize local model weights
    fed_args = dict(
        all_clients=args.all_clients,
        num_users=args.num_users,
        method=args.fed_method,
        dict_users=dict_users,
    )

    local_weights = LocalModelWeights(net_glob=net_glob, **fed_args)
    if args.send_2_models:
        local_weights2 = LocalModelWeights(net_glob=net_glob2, **fed_args)

    # Initialize local update objects
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        noise_rates=pred_user_noise_rates,
        gaussian_noise=gaussian_noise,
    )
    for i in range(args.num_users):
        local = local_update_objects[i]
        local.weight = copy.deepcopy(net_glob.state_dict())

    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            log_print("Learning Rate Decay Epoch {}".format(epoch + 1), detail_log_path)
            log_print("{} => {}".format(args.lr, args.lr * args.lr_decay), detail_log_path)
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        local_losses = []
        local_losses2 = []
        args.g_epoch = epoch

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Update
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args

            if args.method == "fedrn":
                if epoch < args.warmup_epochs:
                    w, loss = local.train_phase1(copy.deepcopy(net_glob).to(args.device))
                else:
                    # Get similarity, expertise values
                    sim_list = []
                    exp_list = []
                    for user in range(args.num_users):
                        sim = CosineSimilarity(
                            local.arbitrary_output.to(args.device),
                            local_update_objects[user].arbitrary_output.to(args.device),
                        ).item()
                        exp = local_update_objects[user].expertise
                        sim_list.append(sim)
                        exp_list.append(exp)

                    # Normalize similarity & expertise values
                    sim_list = [(sim - min(sim_list)) / (max(sim_list) - min(sim_list)) for sim in sim_list]
                    exp_list = [(exp - min(exp_list)) / (max(exp_list) - min(exp_list)) for exp in exp_list]

                    # Compute & sort scores
                    prev_score = args.w_alpha * exp_list[idx] + (1 - args.w_alpha)

                    score_list = []
                    for neighbor_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
                        if neighbor_idx != idx:
                            score = args.w_alpha * exp + (1 - args.w_alpha) * sim
                            score_list.append([score, neighbor_idx])
                    score_list.sort(key=lambda x: x[0], reverse=True)

                    # Get top-k neighbors
                    neighbor_list = []
                    neighbor_score_list = []
                    for k in range(args.num_neighbors):
                        neighbor_score, neighbor_idx = score_list[k]
                        neighbor_net = copy.deepcopy(local_update_objects[neighbor_idx].net1)
                        neighbor_list.append(neighbor_net)
                        neighbor_score_list.append(neighbor_score)

                    w, loss = local.train_phase2(copy.deepcopy(net_glob).to(args.device),
                                                 prev_score,
                                                 neighbor_list,
                                                 neighbor_score_list)

            elif args.send_2_models:
                w, loss, w2, loss2 = local.train(
                    copy.deepcopy(net_glob).to(args.device),
                    copy.deepcopy(net_glob2).to(args.device),
                )
                local_weights2.update(idx, w2)
                local_losses2.append(copy.deepcopy(loss2))

            else:
                w, loss = local.train(copy.deepcopy(net_glob).to(args.device))

            local_weights.update(idx, w)
            local_losses.append(copy.deepcopy(loss))

        w_glob = local_weights.average()  # update global weights
        net_glob.load_state_dict(w_glob, strict=False)  # copy weight to net_glob
        local_weights.init()

        avg_local_loss = sum(local_losses) / len(local_losses) if len(local_losses) > 0 else 0.0

        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
        # for logging purposes
        results = dict(train_acc=train_acc, train_loss=train_loss,
                       test_acc=test_acc, test_loss=test_loss,
                       local_update_loss=avg_local_loss, )

        if args.send_2_models:
            w_glob2 = local_weights2.average()
            net_glob2.load_state_dict(w_glob2)
            local_weights2.init()
            # for logging purposes
            train_acc2, train_loss2 = test_img(net_glob2, log_train_data_loader, args)
            test_acc2, test_loss2 = test_img(net_glob2, log_test_data_loader, args)
            results2 = dict(train_acc2=train_acc2, train_loss2=train_loss2,
                            test_acc2=test_acc2, test_loss2=test_loss2, )

            results = {**results, **results2}

        round_line = 'Round {:3d}'.format(epoch)
        metric_line = ' - '.join([f'{k}: {v:.6f}' for k, v in results.items()])

        log_print('', detail_log_path)
        log_print(round_line, detail_log_path)
        log_print(metric_line, detail_log_path)

        elapsed = time.time() - start
        append_line(
            metric_log_path,
            '{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.2f}'.format(
                epoch,
                train_acc,
                train_loss,
                test_acc,
                test_loss,
                avg_local_loss,
                elapsed,
            )
        )

    total_time = time.time() - start
    log_print('', detail_log_path)
    log_print('Total training time: {:.2f} seconds'.format(total_time), detail_log_path)
    append_line(metric_log_path, 'Total training time: {:.2f} seconds'.format(total_time))
