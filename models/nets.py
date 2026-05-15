#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F  # 👈 新增引入 functional

# def get_model(args):
#     return CNN4Conv(num_classes=args.num_classes)
#根据数据集选择cnn
def get_model(args):
    if args.dataset == 'mnist':
        return CNNMnist(args=args)
    else:
        return CNN4Conv(num_classes=args.num_classes)
        print("🧠 [Model Router] 已加载: CNN4Conv_DualHead (双头架构)")




def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


# class CNN4Conv(nn.Module):
#     def __init__(self, num_classes):
#         super(CNN4Conv, self).__init__()
#         in_channels = 3
#         num_classes = num_classes
#         hidden_size = 64
#
#         self.features = nn.Sequential(
#             conv3x3(in_channels, hidden_size),
#             conv3x3(hidden_size, hidden_size),
#             conv3x3(hidden_size, hidden_size),
#             conv3x3(hidden_size, hidden_size)
#         )
#
#         self.linear = nn.Linear(hidden_size * 2 * 2, num_classes)
#
#     def forward(self, x):
#         features = self.features(x)
#         features = features.view((features.size(0), -1))
#         logits = self.linear(features)
#
#         return logits


class CNN4Conv(nn.Module):
    def __init__(self, num_classes):
        super(CNN4Conv, self).__init__()
        in_channels = 3
        num_classes = num_classes
        hidden_size = 64

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        # ==========================================
        # 🌟 修改点 1：拆分线性层，定义双头
        # 原代码: self.linear = nn.Linear(hidden_size * 2 * 2, num_classes)
        # ==========================================
        self.fc_global = nn.Linear(hidden_size * 2 * 2, num_classes)  # 准备上传云端的全局头
        self.fc_local = nn.Linear(hidden_size * 2 * 2, num_classes)  # 死守本地的个性化头

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))

        # ==========================================
        # 🌟 修改点 2：双头分别进行预测，并同时返回
        # 原代码: logits = self.linear(features); return logits
        # ==========================================
        logits_global = self.fc_global(features)
        logits_local = self.fc_local(features)

        return logits_global, logits_local

# 新增这个专门跑 MNIST 的轻量级网络
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x