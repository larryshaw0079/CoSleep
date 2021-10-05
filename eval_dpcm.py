"""
@Time    : 2021/5/27 15:42
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : eval_dpcm.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import pickle
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
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.std import tqdm

sys.path.append('.')

from cosleep.data import SleepDataset, SleepDatasetImg, TwoDataset
from cosleep.model import DPCFusionClassifier
from cosleep.utils import (
    logits_accuracy,
    get_performance
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
    parser.add_argument('--data-path-v1', type=str, required=True)
    parser.add_argument('--data-path-v2', type=str, required=True)
    parser.add_argument('--data-name', type=str, default='sleepedf', choices=['sleepedf', 'isruc'])
    parser.add_argument('--save-path', type=str, default='cache/tmp')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load-path-v1', type=str, default=None)
    parser.add_argument('--load-path-v2', type=str, default=None)
    parser.add_argument('--channels-v1', type=int, default=2)
    parser.add_argument('--channels-v2', type=int, default=6)
    parser.add_argument('--time-len', type=int, default=3000)
    parser.add_argument('--num-epoch', type=int, default=10, help='The number of epochs in a sequence')
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--write-embedding', action='store_true')
    parser.add_argument('--preprocessing', choices=['none', 'quantile', 'standard'], default='standard')

    # Model
    parser.add_argument('--network', type=str, default='r1d', choices=['r1d', 'r2d'])
    parser.add_argument('--second-network', type=str, default='r2d', choices=['r1d', 'r2d'])
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--pred-steps', type=int, default=5)
    parser.add_argument('--reg-weight', type=float, default=1.0)
    parser.add_argument('--mem-m', type=float, default=0.999)
    parser.add_argument('--mem-k', type=int, default=65536)

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


def finetune(classifier, dataset, sampler, device, args):
    params = []
    if args.finetune_mode == 'freeze':
        print('[INFO] Finetune classifier only for the last layer...')
        for name, param in classifier.named_parameters():
            if 'encoder' in name or 'agg' in name or 'sampler' in name:
                param.requires_grad = False
            else:
                params.append({'params': param})
    elif args.finetune_mode == 'smaller':
        print('[INFO] Finetune the whole classifier where the backbone have a smaller lr...')
        for name, param in classifier.named_parameters():
            if 'encoder' in name or 'agg' in name or 'sampler' in name:
                params.append({'params': param, 'lr': args.lr / 10})
            else:
                params.append({'params': param})
    else:
        print('[INFO] Finetune the whole classifier...')
        for name, param in classifier.named_parameters():
            params.append({'params': param})

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09,
                               amsgrad=True)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    else:
        raise ValueError('Invalid optimizer!')

    criterion = nn.CrossEntropyLoss().cuda(device)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True, drop_last=True,
                             sampler=sampler)

    classifier.train()
    for epoch in range(args.finetune_epochs):
        losses = []
        accuracies = []
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.finetune_epochs}]') as progress_bar:
            for x1, y, x2, _ in progress_bar:
                x1, y, x2 = x1.cuda(device, non_blocking=True), y.cuda(device, non_blocking=True), x2.cuda(device,
                                                                                                           non_blocking=True)
                out = classifier(x1, x2)
                loss = criterion(out, y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accuracies.append(
                    logits_accuracy(out.view(args.batch_size * args.num_epoch, -1), y.view(-1), topk=(1,))[0])

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})


def evaluate(classifier, dataset, device, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    targets = []
    scores = []

    classifier.eval()
    with torch.no_grad():
        for x1, y, x2, _ in data_loader:
            x1, x2 = x1.cuda(device, non_blocking=True), x2.cuda(device, non_blocking=True)

            out = classifier(x1, x2)
            scores.append(out.cpu().numpy())
            targets.append(y.view(-1).numpy())

    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    return scores, targets


def main_worker(run_id, device, train_patients, test_patients, args):
    print(f'[INFO] Loading evaluating dataset...')
    transform = TF.Compose(
        [TF.Resize((64, 64)), TF.ToTensor()]
    )

    train_dataset_v1 = SleepDataset(args.data_path_v1, args.data_name, args.num_epoch, train_patients,
                                    preprocessing=args.preprocessing)
    train_dataset_v2 = SleepDatasetImg(args.data_path_v2, args.data_name, args.num_epoch, transform, train_patients)
    train_dataset = TwoDataset(train_dataset_v1, train_dataset_v2)

    test_dataset_v1 = SleepDataset(args.data_path_v1, args.data_name, args.num_epoch, test_patients,
                                   preprocessing=args.preprocessing)

    test_dataset_v2 = SleepDatasetImg(args.data_path_v2, args.data_name, args.num_epoch, transform, test_patients)
    test_dataset = TwoDataset(test_dataset_v1, test_dataset_v2)

    sampled_indices = np.arange(len(train_dataset))
    np.random.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:int(len(sampled_indices) * args.finetune_ratio)]
    sampler = SubsetRandomSampler(sampled_indices)

    # Finetuning parameters
    if args.finetune_mode == 'freeze':
        use_dropout = False
        if args.use_temperature:
            use_l2_norm = True
        else:
            use_l2_norm = False
        use_final_bn = True
    else:
        use_dropout = True
        use_l2_norm = False
        use_final_bn = False

    classifier = DPCFusionClassifier(first_network=args.network, second_network=args.second_network,
                                     first_channels=args.channels_v1, second_channels=args.channels_v2,
                                     hidden_channels=16, feature_dim=args.feature_dim, num_class=args.classes,
                                     use_dropout=use_dropout, use_l2_norm=use_l2_norm, use_batch_norm=use_final_bn,
                                     device=device)
    classifier.cuda(device)

    if args.load_path_v1 is None:
        warnings.warn('The first view is not trained, using random weights...')
        state_dict_v1 = {}
    else:
        state_dict_v1 = torch.load(args.load_path_v1)
        new_dict = {}
        for key, value in state_dict_v1.items():
            if 'encoder_q.' in key:
                key = key.replace('encoder_q.', 'encoder.')
                new_dict[key] = value
        state_dict_v1 = new_dict

    if args.load_path_v2 is None:
        warnings.warn('The second view is not trained, using random weights...')
        state_dict_v2 = {}
    else:
        state_dict_v2 = torch.load(args.load_path_v2)
        new_dict = {}
        for key, value in state_dict_v2.items():
            if 'encoder_q.' in key:
                key = key.replace('encoder_q.', 'sampler.')
                new_dict[key] = value
        state_dict_v2 = new_dict

    state_dict = {**state_dict_v1, **state_dict_v2}

    classifier.load_state_dict(state_dict, strict=False)
    finetune(classifier, train_dataset, sampler, device, args)
    torch.save(classifier.state_dict(),
               os.path.join(args.save_path, f'dpc_run_{run_id}_finetune_final.pth.tar'))

    del train_dataset_v1, train_dataset_v2

    scores, targets = evaluate(classifier, test_dataset, device, args)
    performance, performance_dict = get_performance(scores, targets)

    with open(os.path.join(args.save_path, f'statistics_{run_id}_final.pkl'), 'wb') as f:
        pickle.dump({'performance': performance, 'args': vars(args)}, f)
    print(performance)


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

    files = os.listdir(args.data_path_v1)
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
