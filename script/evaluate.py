"""
@Time    : 2020/10/5 14:54
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : evaluate.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import track
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from cosleep.data import prepare_sleepedf_dataset, SleepEDFDataset
from cosleep.model import SleepContrast, SleepClassifier


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data-path', type=str, default='./data/sleepedf')
    parser.add_argument('--save-path', type=str, default='./cache/checkpoints')
    parser.add_argument('--load-path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2020)

    # Model params
    parser.add_argument('--num-patient', type=int, default=5)
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--input-channels', type=int, default=2)
    parser.add_argument('--hidden-channels', type=int, default=16)
    parser.add_argument('--num-seq', type=int, default=20)
    parser.add_argument('--pred-steps', type=int, default=5)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)

    # Training params
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr-step', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train-ratio', type=float, default=0.7)

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


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    warnings.warn(f'You have chosen to seed ({seed}) training. '
                  f'This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! '
                  f'You may see unexpected behavior when restarting '
                  f'from checkpoints.')


def finetune(classifier, finetune_loader, args):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()),
                           lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09,
                           weight_decay=1e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    classifier.train()

    for epoch in range(args.finetune_epochs):
        for x, y in track(finetune_loader, description='Finetune'):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            y_hat = classifier(x)
            loss = criterion(y_hat, y[:, -1])

            loss.backward()
            optimizer.step()


def evaluate(classifier, test_loader, args):
    classifier.eval()

    predictions = []
    labels = []
    for x, y in track(test_loader, description='Evaluation'):
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            y_hat = classifier(x)

        labels.append(y.cpu().numpy())
        predictions.append(y_hat.cpu().numpy())

    labels = np.concatenate(labels, axis=0)[:, -1]
    predictions = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')

    return {'accuracy': accuracy, 'f1_micro': f1_micro, 'f1_macro': f1_macro}


if __name__ == '__main__':
    args = parse_args()

    setup_seed(args.seed)

    data, targets = prepare_sleepedf_dataset(path=args.data_path, patients=args.num_patient)
    train_x, test_x, train_y, test_y = train_test_split(data, targets, train_size=args.train_ratio)
    train_dataset = SleepEDFDataset(train_x, train_y, seq_len=args.seq_len,
                                    stride=args.stride, return_label=True)
    test_dataset = SleepEDFDataset(test_x, test_y, seq_len=args.seq_len,
                                   stride=args.stride, return_label=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              drop_last=True, shuffle=True, pin_memory=True)

    model = SleepContrast(input_channels=args.input_channels, hidden_channels=args.hidden_channels,
                          feature_dim=args.feature_dim, pred_steps=args.pred_steps,
                          batch_size=args.batch_size, num_seq=args.num_seq, kernel_sizes=[7, 11, 7])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    cache_dict = torch.load(args.load_path)
    model.load_state_dict(cache_dict['state_dict'])

    classifier = SleepClassifier(input_channels=args.input_channels, hidden_channels=args.hidden_channels,
                                 num_classes=args.num_classes, feature_dim=args.feature_dim,
                                 pred_steps=args.pred_steps, batch_size=args.batch_size,
                                 num_seq=args.num_seq, kernel_sizes=[7, 11, 7])
    classifier.cuda()

    # Copying encoder params
    for finetune_param, pretraining_param in zip(classifier.encoder_q.parameters(), model.encoder_q.parameters()):
        finetune_param.data = pretraining_param.data

    # Copying gru params
    for finetune_param, pretraining_param in zip(classifier.gru.parameters(), model.gru.parameters()):
        finetune_param.data = pretraining_param.data

    classifier.freeze_parameters()

    finetune_size = int(len(train_dataset) * args.finetune_ratio)
    finetune_idx = np.random.choice(np.arange(len(train_dataset)), size=finetune_size, replace=False)
    finetune_x, finetune_y = train_x[finetune_idx], train_y[finetune_idx]
    finetune_dataset = SleepEDFDataset(finetune_x, finetune_y, return_label=True)
    finetune_loader = DataLoader(finetune_dataset, batch_size=args.batch_size,
                                 drop_last=True, shuffle=True, pin_memory=True)

    finetune(classifier, finetune_loader, args)
    torch.save(classifier.state_dict(), os.path.join(args.save_path, 'classifier.pth.tar'))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True)
    results = evaluate(classifier, test_loader, args)
    print(results)
