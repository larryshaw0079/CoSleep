"""
@Time    : 2020/12/21 14:04
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : moco_profile.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import wandb
from sklearn.model_selection import KFold

from cosleep.data import transformation
from cosleep.model import Moco
from cosleep.utils import model_summary

AVAILABLE_1D_TRANSFORMATIONS = [
    'perturbation',
    'jittering',
    'flipping',
    'negating',
    'scaling',
    'mwarping',
    'twarping',
    'cshuffling',
    'cropping'
]

AVAILABLE_2D_TRANSFORMATIONS = [
    'jittering2d',
    'flipping2d',
    'negating2d',
    'scaling2d',
    'mwarping2d',
    'cshuffling2d',
    'cropping2d'
]


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    # Dataset & saving & loading
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--data-name', type=str, default='sleepedf', choices=['sleepedf', 'isruc'])
    parser.add_argument('--save-path', type=str, default='cache/')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--meta-file', type=str, required=True)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--time-len', type=int, default=3000)
    parser.add_argument('--freq-len', type=int, default=None)
    parser.add_argument('--num-extend', type=int, default=500)
    parser.add_argument('--classes', type=int, default=5)

    # Model
    parser.add_argument('--network', type=str, default='r1d', choices=['r1d', 'r2d'])
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--aug', dest='augmentation', type=str, nargs='+', default=None)

    # Training
    parser.add_argument('--only-pretrain', action='store_true')
    parser.add_argument('--devices', type=int, nargs='+', default=None)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)
    parser.add_argument('--finetune-mode', type=str, default='freeze', choices=['freeze', 'smaller', 'all'])
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)

    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

    # MOCO specific configs:
    parser.add_argument('--moco-k', default=2048, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # Misc
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--disp-interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)

    args_parsed = parser.parse_args()

    if verbose:
        message = ''
        message += '-------------------------------- Args ------------------------------\n'
        for k, v in sorted(vars(args_parsed).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------------------- End ----------------------------------'
        print(message)

    return args_parsed


def get_augmentations(augmentation_list, two_crop=False):
    augmentation = []

    for aug_param in augmentation_list:
        if aug_param == 'perturbation':
            augmentation.append(transformation.Perturbation(min_perturbation=10, max_perturbation=300))
        elif aug_param == 'jittering':
            augmentation.append(transformation.Jittering())
        elif aug_param == 'flipping':
            augmentation.append(transformation.Flipping(randomize=True))
        elif aug_param == 'negating':
            augmentation.append(transformation.Negating(randomize=True))
        elif aug_param == 'scaling':
            augmentation.append(transformation.Scaling(randomize=True))
        elif aug_param == 'mwarping':
            augmentation.append(transformation.MagnitudeWarping())
        elif aug_param == 'twarping':
            augmentation.append(transformation.TimeWarping())
        elif aug_param == 'cshuffling':
            augmentation.append(transformation.ChannelShuffling())
        elif aug_param == 'cropping':
            augmentation.append(transformation.RandomCropping(size=2000))
        elif aug_param == 'jittering2d':
            augmentation.append(transformation.Jittering2d())
        elif aug_param == 'flipping2d':
            augmentation.append(transformation.Flipping2d(axis='both', randomize=True))
        elif aug_param == 'negating2d':
            augmentation.append(transformation.Negating2d(randomize=True))
        elif aug_param == 'scaling2d':
            augmentation.append(transformation.Scaling2d(randomize=True))
        elif aug_param == 'mwarping2d':
            augmentation.append(transformation.MagnitudeWarping2d())
        elif aug_param == 'cshuffling2d':
            augmentation.append(transformation.ChannelShuffling2d())
        elif aug_param == 'cropping2d':
            augmentation.append(transformation.RandomCropping2d(size=(80, 20)))
        else:
            raise ValueError(f'Invalid augmentation `{aug_param}`!')

    if two_crop:
        return transformation.TwoCropsTransform(transformation.Compose(augmentation))
    else:
        return transformation.Compose(augmentation)


def main_worker(run_id, device, train_patients, test_patients, args):
    # Pretraining
    model = Moco(network=args.network, device=device, in_channel=args.channels, mid_channel=16, dim=args.feature_dim,
                 K=args.moco_k, m=args.moco_m, T=args.moco_t)
    model.cuda(device)

    criterion = nn.CrossEntropyLoss().cuda(device)

    if args.network == 'r1d':
        tensor_shape = (args.batch_size, args.channels, args.time_len)
    else:
        tensor_shape = (args.batch_size, args.channels, args.freq_len, args.time_len)

    model_summary(model.encoder_q, input_size=tensor_shape[1:], title='Encoder q', device=f'cuda:{args.devices[0]}')
    model_summary(model.encoder_k, input_size=tensor_shape[1:], title='Encoder k', device=f'cuda:{args.devices[0]}')

    model.train()
    with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
        q = torch.randn(*tensor_shape)
        k = torch.randn(*tensor_shape)

        q, k = k.cuda(device, non_blocking=True), q.cuda(device, non_blocking=True)

        output, target = model(q, k)
        loss = criterion(output, target)

        loss.backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace('cache/trace.json')


if __name__ == '__main__':
    args = parse_args()

    if args.wandb:
        with open('./data/wandb.txt', 'r') as f:
            os.environ['WANDB_API_KEY'] = f.readlines()[0]
        wandb.init(project='MVC', group=f'MOCO_{args.network}', config=args)

    if args.seed is not None:
        setup_seed(args.seed)

    devices = args.devices
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    print(f'[INFO] Using devices {devices}...')

    with open(args.meta_file, 'rb') as f:
        meta_info = pickle.load(f)
        patients = np.unique(meta_info['patient'])

    assert args.kfold <= len(patients)
    assert args.fold < args.kfold
    kf = KFold(n_splits=args.kfold)
    for i, (train_index, test_index) in enumerate(kf.split(patients)):
        if i == args.fold:
            print(f'[INFO] Running cross validation for {i + 1}/{args.kfold} fold...')
            train_patients, test_patients = patients[train_index], patients[test_index]
            main_worker(i, devices[0], train_patients.tolist(), test_patients.tolist(), args)
            break
