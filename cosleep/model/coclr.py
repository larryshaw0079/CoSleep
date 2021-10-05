"""
@Time    : 2020/11/10 21:30
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : coclr.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .moco import Moco
from ..backbone.encoder import R1DNet, R2DNet


class CoCLR(Moco):
    '''
    CoCLR: using another view of the data to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''

    def __init__(self, network='r1d', second_network='r2d', device=0, in_channel=2, mid_channel=16, dim=128, K=2048,
                 m=0.999, T=0.07, topk=5):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(CoCLR, self).__init__(network, device, in_channel, mid_channel, dim, K, m, T)

        self.device = device
        self.topk = topk

        if second_network == 'r1d':
            backbone = R1DNet(in_channel, mid_channel, dim, stride=2, kernel_size=3, final_fc=False)
            self.feature_size = backbone.feature_size
            self.sampler = nn.Sequential(
                backbone,
                nn.AdaptiveAvgPool1d((1,)),
                nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.feature_size, dim, kernel_size=1, bias=True)
            )
        elif second_network == 'r2d':
            backbone = R2DNet(in_channel, mid_channel, dim, stride=[(2, 2), (1, 1), (1, 1), (1, 1)], final_fc=False)
            self.feature_size = backbone.feature_size
            self.sampler = nn.Sequential(
                backbone,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(self.feature_size, self.feature_size, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feature_size, dim, kernel_size=1, bias=True)
            )
        else:
            raise ValueError

        for param_s in self.sampler.parameters():
            param_s.requires_grad = False  # not update by gradient

        # create another queue, for the second view of the data
        self.register_buffer("queue_second", torch.randn(dim, K))
        self.queue_second = F.normalize(self.queue_second, dim=0)

        self.register_buffer("queue_vname", torch.ones(K, dtype=torch.long) * -1)
        self.queue_is_full = False

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_second, idx):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)
        #         keys_second = concat_all_gather(keys_second)
        #         vnames = concat_all_gather(vnames)
        # labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_second[:, ptr:ptr + batch_size] = keys_second.T
        self.queue_vname[ptr:ptr + batch_size] = idx
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x1, x2, f1, f2, idx):
        '''Output: logits, targets'''
        (B, *_) = x1.shape

        # x1, x2: two augmentations for the first view
        # f1, f2: two augmentations for the second view

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C
        q = F.normalize(q, dim=1)
        q = q.view(B, self.dim)

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

            # compute key feature for second view
            kf = self.sampler(f2)  # keys: B,C,1,1,1
            kf = F.normalize(kf, dim=1)
            kf = kf.view(B, self.dim)

        # if queue_second is full: compute mask & train CoCLR, else: train InfoNCE

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: N,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        mask_init = idx.unsqueeze(1) == self.queue_vname.unsqueeze(0)
        mask = mask_init.clone()

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_vname != -1)
            if self.queue_is_full:
                print('\n===== queue is full now =====')

        if self.queue_is_full and (self.topk != 0):
            mask_sim = kf.matmul(self.queue_second.clone().detach())
            mask_sim[mask_init] = - np.inf  # mask out self (and sibling videos)
            _, topkidx = torch.topk(mask_sim, self.topk, dim=1)
            topk_onehot = torch.zeros_like(mask_sim)
            topk_onehot.scatter_(1, topkidx, 1)
            mask[topk_onehot.bool()] = True

        mask = torch.cat([torch.ones((mask.shape[0], 1), dtype=torch.long, device=mask.device).bool(), mask], dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, kf, idx)

        return logits, mask.detach()
