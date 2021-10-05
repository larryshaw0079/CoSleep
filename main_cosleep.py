"""
@Time    : 2021/5/15 16:01
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main_cosleep.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import copy
import os
import pickle
import random
import shutil
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as TF
import wandb
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.std import tqdm

from cosleep.data import SleepDataset, SleepDatasetImg, TwoDataset
from cosleep.model import CoSleep, DPCFusionClassifier
from cosleep.utils import get_performance, logits_accuracy, mask_accuracy, MultiNCELoss


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
    parser.add_argument('--data-name', type=str, default='sleepedf', choices=['sleepedf', 'isruc'])
    parser.add_argument('--data-path-v1', type=str, required=True)
    parser.add_argument('--data-path-v2', type=str, required=True)
    parser.add_argument('--load-path-v1', type=str, required=True)
    parser.add_argument('--load-path-v2', type=str, required=True)
    parser.add_argument('--channels-v1', type=int, default=2)
    parser.add_argument('--channels-v2', type=int, default=6)
    parser.add_argument('--time-len-v1', type=int, default=3000)
    parser.add_argument('--time-len-v2', type=int, default=30)
    parser.add_argument('--freq-len-v1', type=int, default=None)
    parser.add_argument('--freq-len-v2', type=int, default=100)
    parser.add_argument('--num-epoch', type=int, default=10, help='The number of epochs in a sequence')
    parser.add_argument('--pred-steps', type=int, default=5)
    parser.add_argument('--save-path', type=str, default='cache/tmp')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--preprocessing', choices=['none', 'quantile', 'standard'], default='standard')

    # Model
    parser.add_argument('--network', type=str, default='r1d', choices=['r1d', 'r2d'])
    parser.add_argument('--second-network', type=str, default='r2d', choices=['r1d', 'r2d'])
    parser.add_argument('--feature-dim', type=int, default=128)

    # Training
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--devices', type=int, nargs='+', default=None)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--iter', dest='iteration', type=int, default=5)
    parser.add_argument('--pretrain-epochs', type=int, default=10)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)
    parser.add_argument('--finetune-mode', type=str, default='freeze', choices=['freeze', 'smaller', 'all'])
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

    parser.add_argument('--mem-k', default=2048, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--mem-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--iteration', type=int, default=5)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--dynamic-lam', action='store_true')
    parser.add_argument('--lower-bound', type=float, default=0.1)
    parser.add_argument('--upper-bound', type=float, default=0.9)
    parser.add_argument('--single-view', action='store_true')
    parser.add_argument('--num-prop', type=int, default=5)

    # Misc
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--disp-interval', type=int, default=20)
    parser.add_argument('--wandb', action='store_true')
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


def pretrain(model, train_dataset_v1, train_dataset_v2, device, run_id, it, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    else:
        raise ValueError('Invalid optimizer!')

    assert len(train_dataset_v1) == len(train_dataset_v2), f'{len(train_dataset_v1)} - {len(train_dataset_v2)}'
    dataset = TwoDataset(train_dataset_v1, train_dataset_v2)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    criterion = MultiNCELoss(reduction='mean').cuda(device)

    if args.dynamic_lam:
        assert args.lam <= 0.5

    model.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        accuracies = []

        if args.dynamic_lam:
            model.lam = args.lam + epoch / args.pretrain_epochs * (0.5 - args.lam)

        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.pretrain_epochs}]',
                  total=len(train_dataset_v1) // args.batch_size) as progress_bar:
            for x, _, idx1, f, __, idx2 in progress_bar:
                assert (idx1 == idx2).all()
                x = x.cuda(device, non_blocking=True)
                f = f.cuda(device, non_blocking=True)
                idx = idx1.cuda(device, non_blocking=True)

                logits, targets = model(x, f, idx)

                loss = criterion(logits, targets)

                acc = mask_accuracy(logits, targets, topk=(1,))[0]
                accuracies.append(acc)

                # if random.random() < 0.9:
                #     # because model has been pretrained with infoNCE,
                #     # in this stage, self-similarity is already very high,
                #     # randomly mask out the self-similarity for optimization efficiency,
                #     targets_clone = targets.clone()
                #     targets_sum = targets.sum(-1)
                #     targets_clone[targets_sum != 1, 0] = 0  # mask out self-similarity
                #     loss = criterion(logits, targets_clone)
                # else:
                #     loss = criterion(logits, targets)

                # loss = criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                progress_bar.set_postfix(
                    {'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})
        if args.wandb:
            wandb.log({f'pretrain_loss_it{it}': np.mean(loss), f'pretrain_acc_it{it}': np.mean(accuracies)})


def finetune(classifier, dataset, device, it, args):
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

    sampled_indices = np.arange(len(dataset))
    np.random.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:int(len(sampled_indices) * args.finetune_ratio)]
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True, drop_last=True,
                             sampler=SubsetRandomSampler(sampled_indices))

    classifier.train()
    for epoch in range(args.finetune_epochs):
        losses = []
        accuracies = []
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.finetune_epochs}]') as progress_bar:
            for x1, y, _, x2, __, ___ in progress_bar:
                x1, y, x2 = x1.cuda(device, non_blocking=True), y.cuda(device, non_blocking=True), x2.cuda(device,
                                                                                                           non_blocking=True)

                out = classifier(x1, x2)
                # out = out.view(-1, out.shape[-1])

                loss = criterion(out, y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accuracies.append(logits_accuracy(out, y.view(-1), topk=(1,))[0])

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})
        if args.wandb:
            if it is None:
                wandb.log(
                    {'finetune_acc_it': np.mean(accuracies), 'finetune_loss': np.mean(losses)})
            else:
                wandb.log({f'finetune_acc_it{it}': np.mean(accuracies), f'finetune_loss_it{it}': np.mean(losses)})


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


def test(state_dict, dataset_v1, dataset_v2, test_dataset_v1, test_dataset_v2, reverse, device, it, args):
    # Finetuning
    if args.finetune_mode == 'freeze':
        use_dropout = False
        use_l2_norm = True
        use_final_bn = True
    else:
        use_dropout = True
        use_l2_norm = False
        use_final_bn = False

    if not reverse:
        classifier = DPCFusionClassifier(first_network=args.network, second_network=args.second_network,
                                         first_channels=args.channels_v1, second_channels=args.channels_v2,
                                         hidden_channels=16,
                                         feature_dim=args.feature_dim,
                                         num_class=5,
                                         use_dropout=use_dropout,
                                         use_l2_norm=use_l2_norm,
                                         use_batch_norm=use_final_bn, device=device)
    else:
        classifier = DPCFusionClassifier(first_network=args.second_network, second_network=args.network,
                                         first_channels=args.channels_v2, second_channels=args.channels_v1,
                                         hidden_channels=16,
                                         feature_dim=args.feature_dim,
                                         num_class=5,
                                         use_dropout=use_dropout,
                                         use_l2_norm=use_l2_norm,
                                         use_batch_norm=use_final_bn, device=device)
    classifier.cuda(device)

    new_state_dict = {}
    for key, param in state_dict.items():
        if 'encoder_q.' in key:
            key = key.replace('encoder_q.', 'encoder.')
            new_state_dict[key] = param
        elif 'sampler.' in key:
            new_state_dict[key] = param
    state_dict = new_state_dict

    classifier.load_state_dict(state_dict, strict=False)

    dataset = TwoDataset(dataset_v1, dataset_v2)
    finetune(classifier, dataset, device, it, args)

    # transform = TF.Compose(
    #     [TF.Resize((64, 64)), TF.ToTensor()]
    # )
    # if (args.network == 'r1d' and not reverse) or (args.network == 'r2d' and reverse):
    #     test_dataset_v1 = SleepDataset(args.data_path_v1, args.data_name, args.num_epoch, test_patients,
    #                                    preprocessing=args.preprocessing)
    #     test_dataset_v2 = SleepDatasetImg(args.data_path_v2, args.data_name, args.num_epoch, transform=transform,
    #                                       patients=test_patients)
    # else:
    #     test_dataset_v1 = SleepDatasetImg(args.data_path_v1, args.data_name, args.num_epoch, transform=transform,
    #                                       patients=test_patients)
    #     test_dataset_v2 = SleepDataset(args.data_path_v2, args.data_name, args.num_epoch, test_patients,
    #                                    preprocessing=args.preprocessing)

    test_dataset = TwoDataset(test_dataset_v1, test_dataset_v2)

    scores, targets = evaluate(classifier, test_dataset, device, args)
    performance = get_performance(scores, targets)
    print(performance)
    return performance


def main_worker(run_id, device, train_patients, test_patients, args):
    assert os.path.isfile(args.load_path_v1), f'Invalid file path {args.load_path_v1}!'
    assert os.path.isfile(args.load_path_v2), f'Invalid file path {args.load_path_v2}!'

    state_dict_v1 = torch.load(args.load_path_v1)
    state_dict_v2 = torch.load(args.load_path_v2)

    print('[INFO] Loading training dataset...')
    if args.network == 'r1d':
        train_dataset_v1 = SleepDataset(args.data_path_v1, args.data_name, args.num_epoch, train_patients,
                                        preprocessing=args.preprocessing, return_idx=True)
    else:
        transform = TF.Compose(
            [TF.Resize((64, 64)), TF.ToTensor()]
        )
        train_dataset_v1 = SleepDatasetImg(args.data_path_v1, args.data_name, args.num_epoch, transform=transform,
                                           patients=train_patients, return_idx=True)
    print(train_dataset_v1)

    if args.second_network == 'r1d':
        train_dataset_v2 = SleepDataset(args.data_path_v2, args.data_name, args.num_epoch, train_patients,
                                        preprocessing=args.preprocessing, return_idx=True)
    else:
        transform = TF.Compose(
            [TF.Resize((64, 64)), TF.ToTensor()]
        )
        train_dataset_v2 = SleepDatasetImg(args.data_path_v2, args.data_name, args.num_epoch, transform=transform,
                                           patients=train_patients, return_idx=True)
    print(train_dataset_v2)

    print('[INFO] Loading evaluation dataset...')

    transform = TF.Compose(
        [TF.Resize((64, 64)), TF.ToTensor()]
    )
    if args.network == 'r1d':
        test_dataset_v1 = SleepDataset(args.data_path_v1, args.data_name, args.num_epoch, test_patients,
                                       preprocessing=args.preprocessing)
        test_dataset_v2 = SleepDatasetImg(args.data_path_v2, args.data_name, args.num_epoch, transform=transform,
                                          patients=test_patients)
    else:
        test_dataset_v1 = SleepDatasetImg(args.data_path_v1, args.data_name, args.num_epoch, transform=transform,
                                          patients=test_patients)
        test_dataset_v2 = SleepDataset(args.data_path_v2, args.data_name, args.num_epoch, test_patients,
                                       preprocessing=args.preprocessing)

    # The last iteration should train the first view
    assert args.iteration % 2 == 1

    # Refine pretraining
    for it in range(args.iteration):
        if it % 2 == 1:
            reverse = True
        else:
            reverse = False

        if reverse:
            print(f'[INFO] Iteration {it + 1}, train the second view...')
        else:
            print(f'[INFO] Iteration {it + 1}, train the first view...')

        if reverse:
            model = CoSleep(first_network=args.second_network, first_channels=args.channels_v2,
                            second_network=args.network, second_channels=args.channels_v1,
                            hidden_channels=16, feature_dim=args.feature_dim, pred_steps=args.pred_steps,
                            temperature=args.temperature, m=args.mem_m, K=args.mem_k,
                            num_prop=args.num_prop, lam=args.lam, lower_bound=args.lower_bound,
                            upper_bound=args.upper_bound, use_temperature=False, device=device)
        else:
            model = CoSleep(first_network=args.network, first_channels=args.channels_v1,
                            second_network=args.second_network, second_channels=args.channels_v2,
                            hidden_channels=16, feature_dim=args.feature_dim, pred_steps=args.pred_steps,
                            temperature=args.temperature, m=args.mem_m, K=args.mem_k,
                            num_prop=args.num_prop, lam=args.lam, lower_bound=args.lower_bound,
                            upper_bound=args.upper_bound, use_temperature=False, device=device)

        model.cuda(device)

        # Second view as sampler
        new_dict = {}
        new_state_dict_v2 = copy.deepcopy(state_dict_v2)
        for k, v in new_state_dict_v2.items():
            if 'encoder_q.' in k:
                k = k.replace('encoder_q.', 'sampler.')
                new_dict[k] = v
        new_state_dict_v2 = new_dict
        new_dict = {}
        # Remove queue
        for k, v in new_state_dict_v2.items():
            if 'queue' not in k:
                new_dict[k] = v
        new_state_dict_v2 = new_dict

        # First view as encoder k
        new_dict = {}  # remove queue, queue_ptr
        new_state_dict_v1 = copy.deepcopy(state_dict_v1)
        for k, v in new_state_dict_v1.items():
            if 'queue' not in k:
                new_dict[k] = v
        new_state_dict_v1 = new_dict

        state_dict = {**new_state_dict_v1, **new_state_dict_v2}
        model.load_state_dict(state_dict, strict=False)

        print(f'Test for iteration [{it}]...')
        if reverse:
            test(model.state_dict(), train_dataset_v2, train_dataset_v1, test_dataset_v2, test_dataset_v1,
                 reverse, device, it, args)
        else:
            test(model.state_dict(), train_dataset_v1, train_dataset_v2, test_dataset_v1, test_dataset_v2,
                 reverse, device, it, args)

        if reverse:
            pretrain(model, train_dataset_v2, train_dataset_v1, device, run_id, it, args)
        else:
            pretrain(model, train_dataset_v1, train_dataset_v2, device, run_id, it, args)

        # Update the state dict
        state_dict_v1 = model.state_dict()
        state_dict_v1, state_dict_v2 = state_dict_v2, state_dict_v1

        # Saving
        if reverse:
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, f'cosleep_second_run_{run_id}_iter_{it}.pth.tar'))
            # test(model.state_dict(), train_dataset_v2, train_dataset_v1, test_patients, reverse, device, it, args)
        else:
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, f'cosleep_first_run_{run_id}_iter_{it}.pth.tar'))
            # test(model.state_dict(), train_dataset_v1, train_dataset_v2, test_patients, reverse, device, it, args)

    # Finetuning
    if args.finetune_mode == 'freeze':
        use_dropout = False
        use_l2_norm = True
        use_final_bn = True
    else:
        use_dropout = True
        use_l2_norm = False
        use_final_bn = False
    classifier = DPCFusionClassifier(first_network=args.network, second_network=args.second_network,
                                     first_channels=args.channels_v1, second_channels=args.channels_v2,
                                     hidden_channels=16, feature_dim=args.feature_dim, num_class=5,
                                     use_dropout=use_dropout, use_l2_norm=use_l2_norm, use_batch_norm=use_final_bn,
                                     device=device)
    classifier.cuda(device)

    state_dict = model.state_dict()
    new_state_dict = {}
    for key, param in state_dict.items():
        if 'encoder_q.' in key:
            key = key.replace('encoder_q.', 'encoder.')
            new_state_dict[key] = param
        elif 'sampler.' in key:
            new_state_dict[key] = param
    state_dict = new_state_dict
    classifier.load_state_dict(state_dict, strict=False)

    train_dataset = TwoDataset(train_dataset_v1, train_dataset_v2)
    finetune(classifier, train_dataset, device, None, args)
    torch.save(classifier.state_dict(), os.path.join(args.save_path, f'mc3_run_{run_id}_finetuned.pth.tar'))

    # Evaluation
    del train_dataset_v1, train_dataset_v2

    test_dataset = TwoDataset(test_dataset_v1, test_dataset_v2)

    scores, targets = evaluate(classifier, test_dataset, device, args)
    performance, _ = get_performance(scores, targets)
    with open(os.path.join(args.save_path, f'statistics_{run_id}.pkl'), 'wb') as f:
        pickle.dump({'performance': performance, 'args': vars(args), 'cmd': sys.argv}, f)
    performance.to_csv(os.path.join(args.save_path, 'performance.csv'), index=False)
    print(performance)


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    if args.wandb:
        with open('../data/wandb.txt', 'r') as f:
            os.environ['WANDB_API_KEY'] = f.readlines()[0]
        name = 'mc3'
        name += f'_fold{args.fold}'
        name += datetime.now().strftime('_%m-%d_%H-%M')
        wandb.init(project='MC3', group='MC3', name=name, config=args)

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
    # patients = np.asarray(patients)

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
