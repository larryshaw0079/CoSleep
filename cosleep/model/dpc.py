"""
@Time    : 2020/11/10 21:27
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dpc.py
@Software: PyCharm
@Desc    : 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import R1DNet


class CPC(nn.Module):
    def __init__(self, input_channels, feature_dim, pred_steps, use_temperature, temperature,
                 device):
        super(CPC, self).__init__()

        # self.input_size = input_size
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.device = device

        self.encoder = R1DNet(input_channels, 16, feature_dim, stride=2, kernel_size=[7, 11, 11, 7], final_fc=True)
        self.agg = nn.GRU(input_size=feature_dim, hidden_size=feature_dim, batch_first=True)
        self.predictors = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for i in range(pred_steps)])

        self.relu = nn.ReLU(inplace=True)
        self.targets = None

        # self._initialize_weights(self.agg)
        # self._initialize_weights(self.predictor)

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, channel, time_len = x.shape
        x = x.view(batch_size * num_epoch, channel, time_len)
        feature = self.encoder(x)

        feature = feature.view(batch_size, num_epoch, self.feature_dim)
        feature_relu = self.relu(feature)

        ### aggregate, predict future ###
        _, hidden = self.agg(feature_relu[:, :num_epoch - self.pred_steps, :].contiguous())
        hidden = hidden[-1]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        preds = [self.predictors[i](hidden) for i in range(self.pred_steps)]
        preds = torch.stack(preds, dim=1)

        # Feature: (batch_size, num_epoch, feature_size, last_size)
        # Pred: (batch_size, pred_steps, feature_size, last_size)
        if self.use_temperature:
            feature = F.normalize(feature, p=2, dim=-1)
            preds = F.normalize(preds, p=2, dim=-1)

        logits = torch.einsum('ijk,mnk->ijnm', [feature, preds])
        # print('3. Logits: ', logits.shape)
        logits = logits.view(batch_size * num_epoch, self.pred_steps * batch_size)
        if self.use_temperature:
            logits /= self.temperature

        if self.targets is None:
            targets = torch.zeros(batch_size, num_epoch, self.pred_steps, batch_size)
            for i in range(batch_size):
                for j in range(self.pred_steps):
                    targets[i, num_epoch - self.pred_steps + j, j, i] = 1
            targets = targets.view(batch_size * num_epoch, self.pred_steps * batch_size)
            targets = targets.argmax(dim=1)
            targets = targets.cuda(device=self.device)
            self.targets = targets

        return logits, self.targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class CPCClassifier(nn.Module):
    def __init__(self, input_channels, feature_dim, pred_steps, num_class,
                 use_l2_norm, use_dropout, use_batch_norm, device):
        super(CPCClassifier, self).__init__()

        # self.input_size = input_size
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.device = device
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm

        self.encoder = R1DNet(input_channels, 16, feature_dim, stride=2, kernel_size=[7, 11, 11, 7], final_fc=True)
        # self.agg = nn.GRU(input_size=feature_dim, hidden_size=feature_dim, batch_first=True)

        # self.relu = nn.ReLU(inplace=True)

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(self.feature_dim))
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

        # print('2. Context: ', context.shape)

        if self.use_l2_norm:
            feature = F.normalize(feature, p=2, dim=-1)

        out = self.final_fc(feature)

        return out


class DPC(nn.Module):
    def __init__(self, input_channels, feature_dim, pred_steps, use_temperature, temperature,
                 device):
        super(DPC, self).__init__()

        # self.input_size = input_size
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.device = device

        self.encoder = R1DNet(input_channels, 16, feature_dim, stride=2, kernel_size=[7, 11, 11, 7], final_fc=True)
        self.agg = nn.GRU(input_size=feature_dim, hidden_size=feature_dim, batch_first=True)
        self.predictor = nn.Linear(feature_dim, feature_dim)

        self.relu = nn.ReLU(inplace=True)
        self.targets = None

        # self._initialize_weights(self.agg)
        # self._initialize_weights(self.predictor)

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, channel, time_len = x.shape
        x = x.view(batch_size * num_epoch, channel, time_len)
        feature = self.encoder(x)

        feature = feature.view(batch_size, num_epoch, self.feature_dim)
        feature_relu = self.relu(feature)

        ### aggregate, predict future ###
        _, hidden = self.agg(feature_relu[:, 0:num_epoch - self.pred_steps, :].contiguous())
        hidden = hidden[-1, :, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        # print('first hidden: ', hidden.shape)

        pred = []
        for i in range(self.pred_steps):
            # sequentially pred future
            p_tmp = self.predictor(hidden)
            # print('p_tmp', p_tmp.shape)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            # print('hidden', hidden.shape)
            hidden = hidden[-1, :, :]
        pred = torch.stack(pred, 1)

        # Feature: (batch_size, num_epoch, feature_size, last_size)
        # Pred: (batch_size, pred_steps, feature_size, last_size)
        # feature = feature.permute(0, 1, 3, 2).contiguous()
        feature = F.normalize(feature, p=2, dim=-1)

        # pred = pred.permute(0, 1, 3, 2).contiguous()
        pred = F.normalize(pred, p=2, dim=-1)

        # print(feature.shape, pred.shape)
        logits = torch.einsum('ijk,mnk->ijnm', [feature, pred])
        # print('3. Logits: ', logits.shape)
        logits = logits.view(batch_size * num_epoch, self.pred_steps * batch_size)
        if self.use_temperature:
            logits /= self.temperature

        if self.targets is None:
            targets = torch.zeros(batch_size, num_epoch, self.pred_steps, batch_size)
            for i in range(batch_size):
                for j in range(self.pred_steps):
                    targets[i, num_epoch - self.pred_steps + j, j, i] = 1
            targets = targets.view(batch_size * num_epoch, self.pred_steps * batch_size)
            targets = targets.argmax(dim=1)
            targets = targets.cuda(device=self.device)
            self.targets = targets

        return logits, self.targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class DPCClassifier(nn.Module):
    def __init__(self, input_channels, feature_dim, num_class,
                 use_l2_norm, use_dropout, use_batch_norm, device):
        super(DPCClassifier, self).__init__()

        # self.input_size = input_size
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.device = device
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm

        self.encoder = R1DNet(input_channels, 16, feature_dim, stride=2, kernel_size=[7, 11, 11, 7], final_fc=True)
        # self.agg = nn.GRU(input_size=feature_dim, hidden_size=feature_dim, batch_first=True)

        # self.relu = nn.ReLU(inplace=True)

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(self.feature_dim))
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

        # print('2. Context: ', context.shape)

        if self.use_l2_norm:
            feature = F.normalize(feature, p=2, dim=-1)

        out = self.final_fc(feature)

        # print('3. Out: ', out.shape)

        return out
