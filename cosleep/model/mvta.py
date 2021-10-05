"""
@Time    : 2021/1/2 23:14
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : mc3.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import R1DNet, ResNet, GRU


class MVTA(nn.Module):
    def __init__(self, network, input_channels_v1, input_channels_v2, hidden_channels, feature_dim, pred_steps, reverse,
                 temperature, m, K, prop_iter, num_prop, single_view, device):
        super(MVTA, self).__init__()

        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.reverse = reverse
        self.temperature = temperature
        self.m = m
        self.K = K
        self.prop_iter = prop_iter
        self.num_prop = num_prop
        self.single_view = single_view
        self.device = device

        if network == 'r1d':
            if reverse:
                self.encoder_q = ResNet(input_channels=input_channels_v2, num_classes=feature_dim)
                self.encoder_k = ResNet(input_channels=input_channels_v2, num_classes=feature_dim)
                self.sampler = R1DNet(input_channels_v1, hidden_channels, feature_dim, stride=2,
                                      kernel_size=[7, 11, 11, 7],
                                      final_fc=True)
            else:
                self.encoder_q = R1DNet(input_channels_v1, hidden_channels, feature_dim, stride=2,
                                        kernel_size=[7, 11, 11, 7],
                                        final_fc=True)
                self.encoder_k = R1DNet(input_channels_v1, hidden_channels, feature_dim, stride=2,
                                        kernel_size=[7, 11, 11, 7],
                                        final_fc=True)
                self.sampler = ResNet(input_channels=input_channels_v2, num_classes=feature_dim)
        else:
            if reverse:
                self.encoder_q = R1DNet(input_channels_v2, hidden_channels, feature_dim, stride=2,
                                        kernel_size=[7, 11, 11, 7],
                                        final_fc=True)
                self.encoder_k = R1DNet(input_channels_v2, hidden_channels, feature_dim, stride=2,
                                        kernel_size=[7, 11, 11, 7],
                                        final_fc=True)
                self.sampler = ResNet(input_channels=input_channels_v1, num_classes=feature_dim)
            else:
                self.encoder_q = ResNet(input_channels=input_channels_v1, num_classes=feature_dim)
                self.encoder_k = ResNet(input_channels=input_channels_v1, num_classes=feature_dim)
                self.sampler = R1DNet(input_channels_v2, hidden_channels, feature_dim, stride=2,
                                      kernel_size=[7, 11, 11, 7],
                                      final_fc=True)

        for param_s in self.sampler.parameters():
            param_s.requires_grad = False
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

        self.relu = nn.ReLU(inplace=True)
        # self.targets_pred = None

        self.register_buffer("queue_first", torch.randn(feature_dim, K))
        self.queue_first = F.normalize(self.queue_first, dim=0)

        self.register_buffer("queue_second", torch.randn(feature_dim, K))
        self.queue_second = F.normalize(self.queue_second, dim=0)

        self.register_buffer("queue_idx", torch.ones(K, dtype=torch.long) * -1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_is_full = False

        self._initialize_weights(self.predictor)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feature_q, feature_k, idx):
        batch_size = feature_q.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_first[:, ptr:ptr + batch_size] = feature_q.T
        self.queue_second[:, ptr:ptr + batch_size] = feature_k.T
        self.queue_idx[ptr:ptr + batch_size] = idx
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x1, x2, idx):
        assert x1.shape[:2] == x2.shape[:2] and x1.shape[:2] == idx.shape[:2], \
            f'x1: {x1.shape}, x2: {x2.shape}, idx: {idx.shape}'

        (B1, num_epoch, *epoch_shape1) = x1.shape
        (B2, num_epoch, *epoch_shape2) = x2.shape

        # Compute query features for the first view
        x1 = x1.view(B1 * num_epoch, *epoch_shape1)
        feature_q = self.encoder_q(x1)
        feature_q = F.normalize(feature_q, p=2, dim=-1)
        feature_q = feature_q.view(B1, num_epoch, self.feature_dim)

        # Get predictions
        feature_relu = self.relu(feature_q)
        out, h_n = self.agg(feature_relu[:, :-self.pred_steps, :].contiguous())
        pred = []
        h_next = h_n
        c_next = out[:, -1, :].squeeze(1)
        for i in range(self.pred_steps):
            z_pred = self.predictor(c_next)
            pred.append(z_pred)
            c_next, h_next = self.agg(z_pred.unsqueeze(1), h_next)
            c_next = c_next[:, -1, :].squeeze(1)
        pred = torch.stack(pred, 1)  # (batch, pred_step, feature_dim)
        # Compute scores
        pred = pred.contiguous()
        pred = F.normalize(pred, p=2, dim=-1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            # Compute key features for the first view
            feature_k = self.encoder_k(x1)
            feature_k = F.normalize(feature_k, p=2, dim=-1)
            feature_k = feature_k.view(B1, num_epoch, self.feature_dim)

            # Compute key features for the second view
            x2 = x2.view(B2 * num_epoch, *epoch_shape2)
            feature_kf = self.sampler(x2)
            feature_kf = F.normalize(feature_kf, p=2, dim=-1)
            feature_kf = feature_kf.view(B2, num_epoch, self.feature_dim)

        # Compute logits
        # logits_pred = torch.einsum('ijk,mnk->ijnm', [pred, feature_k])
        # logits_pred = logits_pred.view(B1 * self.pred_steps, num_epoch * B1)
        #
        # logits_mem = torch.einsum('mnk,ki->mni', [pred, self.queue_first.clone().detach()])
        # logits_mem = logits_mem.view(B1 * self.pred_steps, self.K)
        # logits = torch.cat([logits_pred, logits_mem], dim=-1)

        logits_pos = torch.einsum('ijk,ijk->ij', [pred, feature_k[:, -self.pred_steps:, :]])
        logits_pos = logits_pos.view(B1 * self.pred_steps, 1)

        logits_neg = torch.einsum('ijk,km->ijm', [pred, self.queue_first.clone().detach()])
        logits_neg = logits_neg.view(B1 * self.pred_steps, self.K)

        logits = torch.cat([logits_pos, logits_neg], dim=-1)

        logits /= self.temperature

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_idx != -1)
            if self.queue_is_full:
                print('[INFO] ===== Queue is full now =====')

        # if self.targets_pred is None:
        #     targets_pred = torch.zeros(B1, num_epoch, self.pred_steps, B1)
        #     for i in range(B1):
        #         for j in range(self.pred_steps):
        #             targets_pred[i, num_epoch - self.pred_steps + j, j, i] = 1
        #     targets_pred = targets_pred.view(B1 * num_epoch, self.pred_steps * B1)
        #     targets_pred = targets_pred.t()
        #     targets_pred = targets_pred.cuda(device=self.device)
        #     self.targets_pred = targets_pred
        #
        # targets_mem = torch.zeros(B1, self.pred_steps, self.K)
        # targets_mem = targets_mem.cuda(self.device)

        targets_mem = idx.unsqueeze(-1)[:, -self.pred_steps:] == self.queue_idx.unsqueeze(0)

        if self.queue_is_full:
            if self.single_view:
                mem_sim = torch.einsum('ijk,km->ijm', [feature_k[:, -self.pred_steps:, :],
                                                       self.queue_first.clone().detach()])
            else:
                mem_sim = torch.einsum('ijk,km->ijm', [feature_kf[:, -self.pred_steps:, :],
                                                       self.queue_second.clone().detach()])  # (B, num_epoch, K)
            mem_sim[targets_mem] = -np.inf  # (B, num_epoch), (K), exclude self
            targets_mem = targets_mem.float()
            _, topk_idx = torch.topk(mem_sim, k=self.num_prop, dim=-1)
            targets_mem.scatter_(-1, topk_idx, 1)

            queue_sim = torch.einsum('km,kn->mn',
                                     [self.queue_first.clone().detach(), self.queue_first.clone().detach()])
            queue_sim_second = torch.einsum('km,kn->mn',
                                            [self.queue_second.clone().detach(), self.queue_second.clone().detach()])
            _, topk_idx = torch.topk(queue_sim, k=self.num_prop, dim=-1)
            _, topk_idx_second = torch.topk(queue_sim_second, k=self.num_prop, dim=-1)
            del queue_sim, queue_sim_second
            neighbor_mat = torch.zeros(self.K, self.K).cuda(self.device)
            neighbor_mat.scatter_(-1, topk_idx, 1)
            neighbor_mat_second = torch.zeros(self.K, self.K).cuda(self.device)
            neighbor_mat_second.scatter_(-1, topk_idx_second, 1)
            del topk_idx, topk_idx_second
            queue_tmp_mat = torch.eye(self.K).cuda(self.device)  # for matrix power
            queue_idx = torch.zeros(self.K, self.K).cuda(self.device)
            for i in range(1, self.prop_iter):
                if self.single_view:
                    queue_tmp_mat = torch.mm(queue_tmp_mat, neighbor_mat)
                else:
                    if i % 2 == 1:
                        queue_tmp_mat = torch.mm(queue_tmp_mat, neighbor_mat)
                    else:
                        queue_tmp_mat = torch.mm(queue_tmp_mat, neighbor_mat_second)

                queue_idx += queue_tmp_mat

            targets_mem.matmul(queue_idx)

        targets_mem = targets_mem.view(self.pred_steps * B1, self.K)
        targets = torch.cat([torch.ones(targets_mem.shape[0], 1).long().cuda(self.device), targets_mem], dim=-1)

        self._dequeue_and_enqueue(feature_k.view(B1 * num_epoch, self.feature_dim),
                                  feature_kf.view(B1 * num_epoch, self.feature_dim),
                                  idx.view(B1 * num_epoch))

        return logits, targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
