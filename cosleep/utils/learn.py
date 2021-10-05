"""
@Time    : 2020/12/16 16:51
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : learn.py
@Software: PyCharm
@Desc    : 
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def adjust_learning_rate(optimizer, lr, epoch, total_epochs, args):
    """Decay the learning rate based on schedule"""
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    else:  # stepwise lr schedule
        for milestone in args.lr_schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# class MultiNCELoss(nn.Module):
#     def __init__(self, reduction = 'mean'):
#         super(MultiNCELoss, self).__init__()
#
#         assert reduction in ['none', 'mean', 'sum']
#         self.reduction = reduction
#
#     def forward(self, outputs, targets):
#         mask_sum = targets.sum(1)
#         loss = - torch.log((F.softmax(outputs, dim=1) * targets).sum(1))
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

def multi_nce_loss(logits, mask):
    mask_sum = mask.sum(1)
    loss = - torch.log((F.softmax(logits, dim=1) * mask).sum(1))
    return loss.mean()


class MultiNCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MultiNCELoss, self).__init__()

        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, logits, targets):
        loss = - torch.log((F.softmax(logits, dim=1) * targets).sum(1))

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            raise ValueError


class SoftLogitLoss(nn.Module):
    def __init__(self):
        super(SoftLogitLoss, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        # print(output.shape, target.shape)
        if len(target.shape) == 1:
            target = target.view(-1, 1)
        loss = torch.log(1 + torch.exp(-target * output)).mean()

        return loss
