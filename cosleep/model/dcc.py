"""
@Time    : 2020/11/10 21:27
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dcc.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import R1DNet


class DCC(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, use_temperature, temperature,
                 device):
        super(DCC, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.device = device
        self.encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)

        self.targets = None

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, channel, time_len = x.shape
        x = x.view(batch_size * num_epoch, channel, time_len)
        feature = self.encoder(x)
        feature = F.normalize(feature, p=2, dim=1)
        feature = feature.view(batch_size, num_epoch, self.feature_dim)

        #################################################################
        #                       Multi-InfoNCE Loss                      #
        #################################################################
        mask = torch.zeros(batch_size, num_epoch, num_epoch, batch_size, dtype=bool)
        for i in range(batch_size):
            for j in range(num_epoch):
                mask[i, j, :, i] = 1
        mask = mask.cuda(self.device)

        logits = torch.einsum('ijk,mnk->ijnm', [feature, feature])
        # if self.use_temperature:
        #     logits /= self.temperature

        pos = torch.exp(logits.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg = torch.exp(logits.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                           batch_size * num_epoch - num_epoch)).sum(-1)

        loss = (-torch.log(pos / (pos + neg))).mean()

        return loss

        # Compute scores
        # logits = torch.einsum('ijk,kmn->ijmn', [pred, feature])  # (batch, pred_step, num_seq, batch)
        # logits = logits.view(batch_size * self.pred_steps, num_epoch * batch_size)

        # logits = torch.einsum('ijk,mnk->ijnm', [feature, feature])
        # # print('3. Logits: ', logits.shape)
        # logits = logits.view(batch_size * num_epoch, num_epoch * batch_size)
        # if self.use_temperature:
        #     logits /= self.temperature
        #
        # if self.targets is None:
        #     targets = torch.zeros(batch_size, num_epoch, num_epoch, batch_size)
        #     for i in range(batch_size):
        #         for j in range(num_epoch):
        #             targets[i, j, :, i] = 1
        #     targets = targets.view(batch_size * num_epoch, num_epoch * batch_size)
        #     targets = targets.argmax(dim=1)
        #     targets = targets.cuda(device=self.device)
        #     self.targets = targets
        #
        # return logits, self.targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class DCCClassifier(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, num_class,
                 use_l2_norm, use_dropout, use_batch_norm, device):
        super(DCCClassifier, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm

        self.encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(feature_dim))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))
        final_fc.append(nn.Linear(feature_dim, num_class))
        self.final_fc = nn.Sequential(*final_fc)

        # self._initialize_weights(self.final_fc)

    def forward(self, x):
        batch_size, num_epoch, channel, time_len = x.shape
        x = x.view(batch_size * num_epoch, channel, time_len)
        feature = self.encoder(x)
        # feature = feature.view(batch_size, num_epoch, self.feature_dim)

        if self.use_l2_norm:
            feature = F.normalize(feature, p=2, dim=1)

        out = self.final_fc(feature)
        out = out.view(batch_size, num_epoch, -1)

        # print('3. Out: ', out.shape)

        return out
