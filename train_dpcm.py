"""
@Time    : 2021/4/20 16:37
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : train_dpcm.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import random
import shutil
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as TF
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.std import tqdm

sys.path.append('.')

from cosleep.data import SleepDataset, SleepDatasetImg
from cosleep.model import DPCMemory
from cosleep.utils import (
    logits_accuracy,
    adjust_learning_rate
)


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
    parser.add_argument('--data-name', type=str, default='sleepedf', choices=['sleepedf', 'isruc', 'deap', 'amigos'])
    parser.add_argument('--save-path', type=str, default='cache/tmp')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--time-len', type=int, default=3000)
    parser.add_argument('--num-epoch', type=int, default=10, help='The number of epochs in a sequence')
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--write-embedding', action='store_true')
    parser.add_argument('--preprocessing', choices=['none', 'quantile', 'standard'], default='standard')

    # Model
    parser.add_argument('--network', type=str, default='r1d', choices=['r1d', 'r2d'])
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--pred-steps', type=int, default=5)
    parser.add_argument('--reg-weight', type=float, default=1.0)
    parser.add_argument('--mem-m', type=float, default=0.999)
    parser.add_argument('--mem-k', type=int, default=65536)
    parser.add_argument('--disable-memory', action='store_true')

    # Training
    parser.add_argument('--only-pretrain', action='store_true')
    parser.add_argument('--devices', type=int, nargs='+', default=None)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)
    parser.add_argument('--finetune-mode', type=str, default='freeze', choices=['freeze', 'smaller', 'all'])
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--use-temperature', action='store_true')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

    # Misc
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--disp-interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2020)

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


def pretrain(model, dataset, device, run_id, writer, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09,
                               amsgrad=True)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    else:
        raise ValueError('Invalid optimizer!')

    criterion = nn.CrossEntropyLoss().cuda(device)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    model.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        accuracies = []
        adjust_learning_rate(optimizer, args.lr, epoch, args.pretrain_epochs, args)
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.pretrain_epochs}]') as progress_bar:
            for x, _ in progress_bar:
                x = x.cuda(device, non_blocking=True)

                output, target = model(x)
                loss = criterion(output, target)

                acc = logits_accuracy(output, target, topk=(1,))[0]
                accuracies.append(acc)

                writer.add_scalar('Loss/pretrain', loss.item(), epoch)
                writer.add_scalar('Accuracy/pretrain', acc, epoch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                progress_bar.set_postfix({'Loss': np.mean(losses), 'ACC': np.mean(accuracies)})
        if (epoch + 1) % args.save_interval == 0:
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       os.path.join(args.save_path, f'dpc_{args.network}_{run_id}_pretrain_{epoch}.pth.tar'))


def main_worker(run_id, device, train_patients, test_patients, args):
    writer = SummaryWriter(os.path.join(args.save_path, 'logs'))

    model = DPCMemory(network=args.network, input_channels=args.channels, hidden_channels=16,
                      feature_dim=args.feature_dim, pred_steps=args.pred_steps, use_temperature=False,
                      temperature=args.temperature, use_memory_pool=not args.disable_memory, stop_memory=False,
                      m=args.mem_m, K=args.mem_k,
                      device=device)

    model.cuda(device)

    print(f'[INFO] Loading training dataset...')
    if args.network == 'r1d':
        train_dataset = SleepDataset(args.data_path, args.data_name, args.num_epoch, train_patients,
                                     preprocessing=args.preprocessing)
    elif args.network == 'r2d':
        transform = TF.Compose(
            [TF.Resize((64, 64)), TF.ToTensor()]
        )
        train_dataset = SleepDatasetImg(args.data_path, args.data_name, args.num_epoch, transform, train_patients)
    else:
        raise ValueError

    print(f'[INFO] Start pretraining ...')
    pretrain(model, train_dataset, device, run_id, writer, args)

    torch.save(model.state_dict(),
               os.path.join(args.save_path, f'dpc_{args.network}_{run_id}_pretrain_final.pth.tar'))


if __name__ == '__main__':
    args = parse_args()

    # torch.autograd.set_detect_anomaly(True)

    if args.seed is not None:
        setup_seed(args.seed)

    devices = args.devices
    if devices is None:
        devices = list(range(torch.cuda.device_count()))

    if not os.path.exists(args.save_path):
        warnings.warn(f'The path {args.save_path} dost not existed, created...')
        os.makedirs(args.save_path)
    elif not args.resume:
        warnings.warn(f'The path {args.save_path} already exists, deleted...')
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    print(f'[INFO] Using devices {devices}...')

    files = os.listdir(args.data_path)
    patients = []
    for a_file in files:
        if a_file.endswith('.npz'):
            patients.append(a_file)

    patients = sorted(patients)
    patients = np.asarray(patients)

    assert args.kfold <= len(patients)
    assert args.fold < args.kfold
    kf = KFold(n_splits=args.kfold)
    for i, (train_index, test_index) in enumerate(kf.split(patients)):
        if i == args.fold:
            print(f'[INFO] Running cross validation for {i + 1}/{args.kfold} fold...')
            train_patients, test_patients = patients[train_index].tolist(), patients[test_index].tolist()
            main_worker(i, devices[0], train_patients, test_patients, args)
            break
