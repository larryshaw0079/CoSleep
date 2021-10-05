"""
@Time    : 2021/1/12 16:49
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : ts.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn

from ..backbone import R1DNet


class TemporalShuffling(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, device='cuda'):
        super(TemporalShuffling, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)

        self.linear_head = nn.Linear(2 * feature_dim, 1, bias=True)

    def forward(self, x1, x2, x3):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z3 = self.encoder(x3)

        diff1 = torch.abs(z1 - z2)
        diff2 = torch.abs(z2 - z3)

        out = torch.cat([diff1, diff2], dim=-1)
        out = self.linear_head(out)

        return out
