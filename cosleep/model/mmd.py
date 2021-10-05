"""
@Time    : 2021/2/6 15:24
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : model.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import R1DNet, R1DNetSimple, GRU


class DPCMMD(nn.Module):
    def __init__(self, input_size, input_channels, hidden_channels, feature_dim, pred_steps, use_temperature,
                 temperature,
                 backbone, use_memory_pool=False, stop_memory=False, m=None, K=None, device='cuda'):
        super(DPCMMD, self).__init__()

        assert backbone in ['normal', 'simple']

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.use_memory_pool = use_memory_pool
        self.stop_memory = stop_memory
        self.m = m
        self.K = K
        self.device = device

        if use_memory_pool:
            assert m is not None
            assert K is not None

        if backbone == 'simple':
            self.encoder_q = R1DNetSimple(input_size, input_channels, feature_dim)
        else:
            self.encoder_q = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                                    final_fc=True)
        if use_memory_pool:
            if backbone == 'simple':
                self.encoder_k = R1DNetSimple(input_size, input_channels, feature_dim)
            else:
                self.encoder_k = R1DNet(input_channels, hidden_channels, feature_dim, stride=2,
                                        kernel_size=[7, 11, 11, 7],
                                        final_fc=True)

        # feature_size = self.encoder_q.feature_size
        # self.feature_size = feature_size
        self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

        if use_memory_pool:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        self.relu = nn.ReLU(inplace=True)
        self.targets = None

        if use_memory_pool:
            self.register_buffer("queue", torch.randn(feature_dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self._initialize_weights(self.predictor)

    def start_memory(self):
        assert self.stop_memory
        print('[INFO] Start using memory...')
        self.stop_memory = False
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.targets = None

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = keys.view(-1, self.feature_dim)
        batch_size, *_ = keys.shape

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f'{self.K}, {batch_size}'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, *x_shape = x.shape
        x = x.view(batch_size * num_epoch, *x_shape)
        feature_q = self.encoder_q(x)  # (batch_size, num_epoch, feature_size)
        feature_q = feature_q.view(batch_size, num_epoch, self.feature_dim)
        feature_relu = self.relu(feature_q)

        if self.use_memory_pool and not self.stop_memory:
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                feature_k = self.encoder_k(x)
                feature_k = feature_k.view(batch_size, num_epoch, self.feature_dim)

        out, h_n = self.agg(feature_relu[:, :-self.pred_steps, :].contiguous())

        # Get predictions
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

        feature_q = F.normalize(feature_q, p=2, dim=-1)
        pred = F.normalize(pred, p=2, dim=-1)
        if self.use_memory_pool and not self.stop_memory:
            feature_k = F.normalize(feature_k, p=2, dim=-1)

        # feature (batch_size, num_epoch, feature_size)
        # pred (batch_size, pred_steps, feature_size)
        if self.use_memory_pool and not self.stop_memory:
            logits_pos = torch.einsum('ijk,ijk->ij', [pred, feature_k[:, -self.pred_steps:, :]])
            logits_pos = logits_pos.view(batch_size * self.pred_steps, 1)

            logits_neg = torch.einsum('ijk,km->ijm', [pred, self.queue.clone().detach()])
            logits_neg = logits_neg.view(batch_size * self.pred_steps, self.K)

            logits = torch.cat([logits_pos, logits_neg], dim=-1)
        else:
            logits = torch.einsum('ijk,mnk->ijnm', [pred, feature_q])
            logits = logits.view(batch_size * self.pred_steps, num_epoch * batch_size)

        if self.use_temperature:
            logits /= self.temperature

        if self.targets is None:
            if self.use_memory_pool and not self.stop_memory:
                targets = torch.zeros(logits.shape[0]).long().cuda(self.device)
                self.targets = targets
            else:
                targets = torch.zeros(batch_size, num_epoch, self.pred_steps, batch_size)
                for i in range(batch_size):
                    for j in range(self.pred_steps):
                        targets[i, num_epoch - self.pred_steps + j, j, i] = 1
                targets = targets.view(batch_size * num_epoch, self.pred_steps * batch_size)
                targets = targets.t()
                targets = targets.argmax(dim=1)
                targets = targets.cuda(device=self.device)
                self.targets = targets

        if self.use_memory_pool and not self.stop_memory:
            self._dequeue_and_enqueue(feature_k)

        return logits, self.targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class MMDCL(nn.Module):
    def __init__(self, input_size: int, input_channel_1: int, input_channel_2: int, feature_dim: int, pred_steps: int,
                 temperature: float = None, use_temperature: bool = False, w: float = 0.5, m: float = None,
                 K: int = None,
                 pos_mmd: bool = False, backbone: str = 'simple', device=0):
        super(MMDCL, self).__init__()

        assert 0.0 <= w <= 1.0

        self.temperature = temperature
        self.use_temperature = use_temperature
        self.pred_steps = pred_steps
        self.pos_mmd = pos_mmd
        self.w = w
        self.m = m
        self.K = K
        self.feature_dim = feature_dim
        self.device = device

        self.mask_batch = None

        if backbone == 'simple':
            self.encoder_q = R1DNetSimple(input_size, input_channel_1, feature_dim)
            self.encoder_k = R1DNetSimple(input_size, input_channel_1, feature_dim)
            self.sampler = R1DNetSimple(input_size, input_channel_2, feature_dim)
        elif backbone == 'normal':
            self.encoder_q = R1DNet(in_channel=input_channel_1, mid_channel=16, feature_dim=feature_dim, stride=2,
                                    kernel_size=[7, 11, 11, 7], final_fc=True)
            self.encoder_k = R1DNet(in_channel=input_channel_1, mid_channel=16, feature_dim=feature_dim, stride=2,
                                    kernel_size=[7, 11, 11, 7], final_fc=True)
            self.sampler = R1DNet(in_channel=input_channel_2, mid_channel=16, feature_dim=feature_dim, stride=2,
                                  kernel_size=[7, 11, 11, 7], final_fc=True)
        else:
            raise ValueError

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

        self.register_buffer("queue_first", torch.randn(feature_dim, K))
        self.queue_first = F.normalize(self.queue_first, dim=0)

        self.register_buffer("queue_second", torch.randn(feature_dim, K))
        self.queue_second = F.normalize(self.queue_second, dim=0)

        self.register_buffer("queue_idx", torch.ones(K, dtype=torch.long) * -1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_is_full = False

        self._initialize_weights(self.predictor)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, idx: torch.Tensor):
        assert x1.shape[:2] == x2.shape[:2] and x1.shape[:2] == idx.shape[:2], \
            f'x1: {x1.shape}, x2: {x2.shape}, idx: {idx.shape}'

        (B1, num_epoch, *epoch_shape1) = x1.shape
        (B2, num_epoch, *epoch_shape2) = x2.shape

        # Compute query features for the first view
        x1 = x1.view(B1 * num_epoch, *epoch_shape1)
        feature_q = self.encoder_q(x1)
        feature_q = F.normalize(feature_q, p=2, dim=-1)
        feature_q = feature_q.view(B1, num_epoch, -1)

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
            feature_k = feature_k.view(B1, num_epoch, -1)

            # Compute key features for the second view
            x2 = x2.view(B2 * num_epoch, *epoch_shape2)
            feature_kf = self.sampler(x2)
            feature_kf = F.normalize(feature_kf, p=2, dim=-1)
            feature_kf = feature_kf.view(B2, num_epoch, -1)

        # logits_pos = torch.einsum('ijk,imk->ij', [pred, feature_k[:, -self.pred_steps:, :]]).unsqueeze(-1)
        # logits_neg = torch.einsum('ijk,km->ijm', [pred, self.queue_first.clone().detach()])

        # logits_batch = torch.einsum('ijk,mnk->ijnm', [pred, feature_k[:, -self.pred_steps:, :]])
        logits_batch = torch.einsum('ijk,imk->ij', [pred, feature_k[:, -self.pred_steps:, :]]).unsqueeze(-1)
        logits_mem = torch.einsum('ijk,km->ijm', [pred, self.queue_first.clone().detach()])
        # logits = torch.cat([logits_batch.view(*logits_batch.shape[:2], -1), logits_mem], dim=-1)
        # logits = torch.cat([logits_batch, logits_mem], dim=-1)

        if self.use_temperature:
            logits_batch /= self.temperature
            logits_mem /= self.temperature

        pos = logits_batch
        neg = logits_mem

        # if self.mask_batch is None:
        #     mask_batch = torch.zeros(B1, num_epoch, num_epoch, B1, dtype=bool)
        #     for i in range(B1):
        #         for j in range(num_epoch):
        #             mask_batch[i, j, :, i] = 1
        #     mask_batch = mask_batch.cuda(self.device)
        #     mask_batch = mask_batch.view(*mask_batch.shape[:2], -1)
        #     self.mask_batch = mask_batch

        # mask_mem = idx.unsqueeze(-1)[:, -self.pred_steps:] == self.queue_idx.unsqueeze(0)
        # mask = torch.cat([mask_batch, mask_mem], dim=-1)
        # mask_mem = torch.zeros(B1, num_epoch, self.K, dtype=torch.bool).cuda(self.device)
        # mask = torch.cat([torch.ones(*mask_mem.shape[:2], 1, dtype=torch.bool).cuda(self.device), mask_mem], dim=-1)

        # assert logits.shape == mask.shape

        # pos = logits.masked_select(mask).view()
        # neg = logits.masked_select(torch.logical_not(mask)).view()

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_idx != -1)
            if self.queue_is_full:
                print('[INFO] ===== Queue is full now =====')

        # targets_mem = torch.zeros(B1, self.pred_steps, self.K).cuda(self.device)
        # targets_mem = idx.unsqueeze(-1)[:, -self.pred_steps:] == self.queue_idx.unsqueeze(0)
        # targets = torch.cat([torch.ones(targets_mem.shape[0], 1).long().cuda(self.device), targets_mem], dim=-1)

        if self.queue_is_full:
            sim_v1 = torch.einsum('ijk,km->ijm',
                                  [feature_k[:, -self.pred_steps:, :], self.queue_first.clone().detach()])
            sim_v2 = torch.einsum('ijk,km->ijm',
                                  [feature_kf[:, -self.pred_steps:, :], self.queue_second.clone().detach()])

            sim_profile = self.w * sim_v1 + (1 - self.w) * sim_v2
            sim_profile = torch.exp(sim_profile)

            expand_factor = sim_profile.shape[-1] / sim_profile.sum(-1).unsqueeze(-1)
            weight = sim_profile * expand_factor

            pos = torch.exp(pos).sum(-1)
            neg = torch.exp(neg * weight).sum(-1)
        else:
            pos = torch.exp(pos).sum(-1)
            neg = torch.exp(neg).sum(-1)

        loss = (-torch.log(pos / (pos + neg))).mean()

        self._dequeue_and_enqueue(feature_k.view(B1 * num_epoch, self.feature_dim),
                                  feature_kf.view(B1 * num_epoch, self.feature_dim),
                                  idx.view(B1 * num_epoch))

        return loss


class MMDClassifier(nn.Module):
    def __init__(self, input_size: int, input_channel_1: int, input_channel_2: int, feature_dim: int, num_class: int,
                 use_l2_norm: bool, use_dropout: bool, use_batch_norm: bool, backbone: str = 'simple'):
        super(MMDClassifier, self).__init__()

        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_bath_norm = use_batch_norm

        self.feature_dim = feature_dim

        if backbone == 'simple':
            self.encoder_q = R1DNetSimple(input_size, input_channel_1, feature_dim)
            self.encoder_k = R1DNetSimple(input_size, input_channel_2, feature_dim)
        elif backbone == 'normal':
            self.encoder_q = R1DNet(in_channel=input_channel_1, mid_channel=16, feature_dim=feature_dim, stride=2,
                                    kernel_size=[7, 11, 11, 7], final_fc=True)
            self.encoder_k = R1DNet(in_channel=input_channel_2, mid_channel=16, feature_dim=feature_dim, stride=2,
                                    kernel_size=[7, 11, 11, 7], final_fc=True)
        else:
            raise ValueError

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(feature_dim * 2))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))

        final_fc.append(nn.Linear(feature_dim * 2, num_class))
        self.final_fc = nn.Sequential(*final_fc)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        batch_size, num_epoch, *_ = x1.shape

        x1 = x1.view(x1.shape[0] * x1.shape[1], *x1.shape[2:])
        x2 = x2.view(x2.shape[0] * x2.shape[1], *x2.shape[2:])

        feature1 = self.encoder_q(x1)
        feature2 = self.encoder_k(x2)

        if self.use_l2_norm:
            feature1 = F.normalize(feature1, p=2, dim=1)
            feature2 = F.normalize(feature2, p=2, dim=1)

        feature = torch.cat([feature1, feature2], dim=-1)

        out = self.final_fc(feature)
        # out = out.view(batch_size, num_epoch, -1)

        return out
