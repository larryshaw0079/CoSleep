"""
@Time    : 2021/1/12 16:49
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : rp.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import R1DNet


class RelativePosition(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, device='cuda'):
        super(RelativePosition, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)

        self.linear_head = nn.Linear(feature_dim, 1, bias=True)

    def forward(self, x1, x2):
        # print('---------X1', x1.shape)
        # print('---------X2', x2.shape)

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        # print('---------Z1', z1.shape)
        # print('---------Z2', z2.shape)

        out = torch.abs(z1 - z2)
        out = self.linear_head(out)

        return out


class SimpleClassifier(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, num_classes,
                 use_l2_norm, use_dropout, use_batch_norm, device='cuda'):
        super(SimpleClassifier, self).__init__()

        self.num_classes = num_classes
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(feature_dim))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))
        final_fc.append(nn.Linear(feature_dim, num_classes))
        self.final_fc = nn.Sequential(*final_fc)

    def forward(self, x):
        feature = self.encoder(x)
        # feature = feature.view(batch_size, num_epoch, self.feature_dim)

        if self.use_l2_norm:
            feature = F.normalize(feature, p=2, dim=1)

        out = self.final_fc(feature)

        return out
