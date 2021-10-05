"""
@Time    : 2021/4/20 16:39
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : mmp.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import R1DNet, R2DNet
from ..utils import cmd


class MMP(nn.Module):
    def __init__(self, input_channels_v1, input_channels_v2, hidden_channels, feature_dim, device):
        super(MMP, self).__init__()

        self.input_channels_v1 = input_channels_v1
        self.input_channels_v2 = input_channels_v2
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = R1DNet(input_channels_v1, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)
        self.sampler = R2DNet(input_channels_v2, hidden_channels, feature_dim, final_fc=True)

    def forward(self, x1, x2):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, channel, time_len = x1.shape

        x1 = x1.view(batch_size * num_epoch, *x1.shape[2:])
        feature_q = self.encoder(x1)
        feature_q = F.normalize(feature_q, p=2, dim=1)
        feature_q = feature_q.view(batch_size, num_epoch, self.feature_dim)

        x2 = x2.view(batch_size * num_epoch, *x2.shape[2:])
        feature_k = self.sampler(x2)
        feature_k = F.normalize(feature_k, p=2, dim=1)
        feature_k = feature_k.view(batch_size, num_epoch, self.feature_dim)

        #################################################################
        #                       Multi-InfoNCE Loss                      #
        #################################################################
        mask = torch.zeros(batch_size, num_epoch, num_epoch, batch_size, dtype=bool)
        for i in range(batch_size):
            for j in range(num_epoch):
                mask[i, j, :, i] = 1
        mask = mask.cuda(self.device)

        logits_v1 = torch.einsum('ijk,mnk->ijnm', [feature_q, feature_q])
        pos_v1 = torch.exp(logits_v1.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg_v1 = torch.exp(logits_v1.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                                 batch_size * num_epoch - num_epoch)).sum(
            -1)

        loss_v1 = (-torch.log(pos_v1 / (pos_v1 + neg_v1))).mean()

        logits_v2 = torch.einsum('ijk,mnk->ijnm', [feature_k, feature_k])
        pos_v2 = torch.exp(logits_v2.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg_v2 = torch.exp(logits_v2.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                                 batch_size * num_epoch - num_epoch)).sum(
            -1)

        loss_v2 = (-torch.log(pos_v2 / (pos_v2 + neg_v2))).mean()

        return loss_v1, loss_v2, 0


class MMPDiff(nn.Module):
    def __init__(self, input_channels_v1, input_channels_v2, hidden_channels, feature_dim, device):
        super(MMPDiff, self).__init__()

        self.input_channels_v1 = input_channels_v1
        self.input_channels_v2 = input_channels_v2
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = R1DNet(input_channels_v1, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)
        self.sampler = R2DNet(input_channels_v2, hidden_channels, feature_dim, final_fc=True)

        self.I = None

    def forward(self, x1, x2):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, channel, time_len = x1.shape

        x1 = x1.view(batch_size * num_epoch, *x1.shape[2:])
        feature_q = self.encoder(x1)
        feature_q = F.normalize(feature_q, p=2, dim=1)
        feature_q = feature_q.view(batch_size, num_epoch, self.feature_dim)

        x2 = x2.view(batch_size * num_epoch, *x2.shape[2:])
        feature_k = self.sampler(x2)
        feature_k = F.normalize(feature_k, p=2, dim=1)
        feature_k = feature_k.view(batch_size, num_epoch, self.feature_dim)

        #################################################################
        #                       Multi-InfoNCE Loss                      #
        #################################################################
        mask = torch.zeros(batch_size, num_epoch, num_epoch, batch_size, dtype=bool)
        for i in range(batch_size):
            for j in range(num_epoch):
                mask[i, j, :, i] = 1
        mask = mask.cuda(self.device)

        logits_v1 = torch.einsum('ijk,mnk->ijnm', [feature_q, feature_q])
        pos_v1 = torch.exp(logits_v1.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg_v1 = torch.exp(logits_v1.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                                 batch_size * num_epoch - num_epoch)).sum(
            -1)

        loss_v1 = (-torch.log(pos_v1 / (pos_v1 + neg_v1))).mean()

        logits_v2 = torch.einsum('ijk,mnk->ijnm', [feature_k, feature_k])
        pos_v2 = torch.exp(logits_v2.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg_v2 = torch.exp(logits_v2.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                                 batch_size * num_epoch - num_epoch)).sum(
            -1)

        loss_v2 = (-torch.log(pos_v2 / (pos_v2 + neg_v2))).mean()

        if self.I is None:
            self.I = torch.eye(feature_q.view(-1, self.feature_dim).size(0)).cuda(feature_q.device)

        loss_diff = torch.matmul(feature_q.view(-1, self.feature_dim),
                                 feature_k.view(-1, self.feature_dim).t()) - self.I
        loss_diff = loss_diff ** 2
        loss_diff = loss_diff.sum().sqrt()

        return loss_v1, loss_v2, loss_diff


class MMPSim(nn.Module):
    def __init__(self, input_channels_v1, input_channels_v2, hidden_channels, feature_dim, device):
        super(MMPSim, self).__init__()

        self.input_channels_v1 = input_channels_v1
        self.input_channels_v2 = input_channels_v2
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = R1DNet(input_channels_v1, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)
        self.sampler = R2DNet(input_channels_v2, hidden_channels, feature_dim, final_fc=True)

    def forward(self, x1, x2):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, channel, time_len = x1.shape

        x1 = x1.view(batch_size * num_epoch, *x1.shape[2:])
        feature_q = self.encoder(x1)
        feature_q = F.normalize(feature_q, p=2, dim=1)
        feature_q = feature_q.view(batch_size, num_epoch, self.feature_dim)

        x2 = x2.view(batch_size * num_epoch, *x2.shape[2:])
        feature_k = self.sampler(x2)
        feature_k = F.normalize(feature_k, p=2, dim=1)
        feature_k = feature_k.view(batch_size, num_epoch, self.feature_dim)

        #################################################################
        #                       Multi-InfoNCE Loss                      #
        #################################################################
        mask = torch.zeros(batch_size, num_epoch, num_epoch, batch_size, dtype=bool)
        for i in range(batch_size):
            for j in range(num_epoch):
                mask[i, j, :, i] = 1
        mask = mask.cuda(self.device)

        logits_v1 = torch.einsum('ijk,mnk->ijnm', [feature_q, feature_q])
        pos_v1 = torch.exp(logits_v1.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg_v1 = torch.exp(logits_v1.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                                 batch_size * num_epoch - num_epoch)).sum(
            -1)

        loss_v1 = (-torch.log(pos_v1 / (pos_v1 + neg_v1))).mean()

        logits_v2 = torch.einsum('ijk,mnk->ijnm', [feature_k, feature_k])
        pos_v2 = torch.exp(logits_v2.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg_v2 = torch.exp(logits_v2.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                                 batch_size * num_epoch - num_epoch)).sum(
            -1)

        loss_v2 = (-torch.log(pos_v2 / (pos_v2 + neg_v2))).mean()

        loss_sim = cmd(feature_q, feature_k)

        return loss_v1, loss_v2, loss_sim


class FusionClassifier(nn.Module):
    def __init__(self, input_channels_v1, input_channels_v2, hidden_channels, feature_dim, num_class,
                 use_l2_norm, use_dropout, use_batch_norm, device):
        super(FusionClassifier, self).__init__()

        self.input_channels_v1 = input_channels_v1
        self.input_channels_v2 = input_channels_v2
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm

        self.encoder = R1DNet(input_channels_v1, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)
        self.sampler = R2DNet(input_channels_v2, hidden_channels, feature_dim, final_fc=True)

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(feature_dim * 2))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))
        final_fc.append(nn.Linear(feature_dim * 2, num_class))
        self.final_fc = nn.Sequential(*final_fc)

        # self._initialize_weights(self.final_fc)

    def forward(self, x1, x2):
        batch_size, num_epoch, channel, time_len = x1.shape
        x1 = x1.view(batch_size * num_epoch, channel, time_len)
        x2 = x2.view(batch_size * num_epoch, *x2.shape[2:])
        feature_q = self.encoder(x1)
        feature_k = self.sampler(x2)
        # feature = feature.view(batch_size, num_epoch, self.feature_dim)

        if self.use_l2_norm:
            feature_q = F.normalize(feature_q, p=2, dim=1)
            feature_k = F.normalize(feature_k, p=2, dim=1)

        out = self.final_fc(torch.cat([feature_q, feature_k], dim=-1))
        out = out.view(batch_size, num_epoch, -1)

        # print('3. Out: ', out.shape)

        return out
