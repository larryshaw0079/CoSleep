"""
@Time    : 2020/11/10 21:27
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : moco.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.encoder import R1DNet, R2DNet


class Moco(nn.Module):
    def __init__(self, network='r1d', device=0, in_channel=2, mid_channel=16, dim=128, K=2048, m=0.999, T=0.07):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(Moco, self).__init__()

        assert network in ['r1d', 'r2d']

        self.network = network
        self.device = device
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        # create the encoders (including non-linear projection head: 2 FC layers)
        if network == 'r1d':
            backbone = R1DNet(in_channel, mid_channel, dim, stride=2, kernel_size=[7, 11, 11, 7], final_fc=False)
            feature_size = backbone.feature_size
            self.encoder_q = nn.Sequential(
                backbone,
                nn.AdaptiveAvgPool1d((1,)),
                nn.Conv1d(feature_size, feature_size, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(feature_size, dim, kernel_size=1, bias=True)
            )
            backbone = R1DNet(in_channel, mid_channel, dim, stride=2, kernel_size=[7, 11, 11, 7], final_fc=False)
            self.encoder_k = nn.Sequential(
                backbone,
                nn.AdaptiveAvgPool1d((1,)),
                nn.Conv1d(feature_size, feature_size, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(feature_size, dim, kernel_size=1, bias=True)
            )
        elif network == 'r2d':
            backbone = R2DNet(in_channel, mid_channel, dim, kernel_size=(3, 3),
                              stride=[(2, 2), (1, 1), (1, 1), (1, 1)], final_fc=False)
            feature_size = backbone.feature_size
            self.encoder_q = nn.Sequential(
                backbone,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(feature_size, feature_size, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_size, dim, kernel_size=1, bias=True)
            )
            backbone = R2DNet(in_channel, mid_channel, dim, kernel_size=(3, 3),
                              stride=[(2, 2), (1, 1), (1, 1), (1, 1)], final_fc=False)
            self.encoder_k = nn.Sequential(
                backbone,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(feature_size, feature_size, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_size, dim, kernel_size=1, bias=True)
            )
        else:
            raise ValueError

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Notes: for handling sibling videos, e.g. for UCF101 dataset

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''Momentum update of the key encoder'''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x1, x2):
        '''Output: logits, targets'''
        B, *_ = x1.shape

        # compute query features
        q = self.encoder_q(x1)
        q = F.normalize(q, dim=1)
        q = q.view(B, self.dim)

        # in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #             x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = F.normalize(k, dim=1)

            # undo shuffle
        #             k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (B, 1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # (B, K)

        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)  # (B, K+1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        if torch.cuda.is_available():
            labels = labels.cuda(self.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


class MocoClassifier(nn.Module):
    def __init__(self, network='r1d', device=0, in_channel=2, mid_channel=16, dim=128,
                 num_class=5,
                 dropout=0.5,
                 use_dropout=True,
                 use_l2_norm=False,
                 use_final_bn=False):
        super(MocoClassifier, self).__init__()

        assert network in ['r1d', 'r2d']

        self.network = network
        self.device = device
        self.dim = dim
        self.num_class = num_class
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.use_l2_norm = use_l2_norm
        self.use_final_bn = use_final_bn

        # create the encoders (including non-linear projection head: 2 FC layers)
        if network == 'r1d':
            self.backbone = R1DNet(in_channel, mid_channel, dim, stride=2, kernel_size=[7, 11, 11, 7], final_fc=False)
            self.feature_size = self.backbone.feature_size
        elif network == 'r2d':
            self.backbone = R2DNet(in_channel, mid_channel, dim, stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
                                   final_fc=False)
            self.feature_size = self.backbone.feature_size
        else:
            raise ValueError

        if use_final_bn:
            self.final_bn = nn.BatchNorm1d(self.feature_size)
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()

        if use_dropout:
            self.final_fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_size, self.num_class))
        else:
            self.final_fc = nn.Sequential(
                nn.Linear(self.feature_size, self.num_class))

        self._initialize_weights(self.final_fc)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)

    def forward(self, x):
        B, *_ = x.shape
        out = self.backbone(x)
        if self.network == 'r1d':
            out = F.adaptive_avg_pool1d(out, (1,))
        elif self.network == 'r2d':
            out = F.adaptive_avg_pool2d(out, (1, 1))
        else:
            raise ValueError

        out = out.view(B, self.feature_size)

        if self.use_l2_norm:
            out = F.normalize(out, p=2, dim=1)

        if self.use_final_bn:
            out = self.final_fc(self.final_bn(out))
        else:
            out = self.final_fc(out)

        return out
