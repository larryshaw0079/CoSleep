"""
@Time    : 2020/11/28 13:21
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : generate_augmentation.py
@Software: PyCharm
@Desc    : 
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from tqdm.std import tqdm


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default='data/sleepedf')
    parser.add_argument('--dest-path', type=str, default='data/sleepedf_processed')
    parser.add_argument('--max-pert', dest='max_perturbation', type=int, default=500)
    parser.add_argument('--num-sampling', type=int, default=5)
    parser.add_argument('--sampling-rate', type=int, default=100)

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


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.dest_path):
        warnings.warn(f'The path {args.dest_path} does not existed, created.')
        os.makedirs(args.dest_path)

    meta_df = {'path': [], 'class': [], 'patient': []}

    file_list = os.listdir(args.data_path)
    for i_patient, file in enumerate(file_list):
        print(f'Processing {file}...')
        file_name = os.path.join(args.data_path, file)
        file_prefix = file.split('.')[0]

        if not os.path.exists(os.path.join(args.dest_path, file_prefix)):
            os.makedirs(os.path.join(args.dest_path, file_prefix))

        data = np.load(file_name)

        recordings = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=0)  # (channel, num_epoch, length)
        annotations = data['annotation']
        _, num_epochs, epoch_length = recordings.shape
        recordings = recordings.reshape(-1, num_epochs * epoch_length)  # (channel, num_epoch*length)

        for idx in tqdm(range(num_epochs), desc='EPOCH'):
            data_q = recordings[:, idx * epoch_length:(idx + 1) * epoch_length].astype(np.float32)  # (channel, length)
            data_k = []
            candidates = np.concatenate([np.arange(idx * epoch_length - args.max_perturbation, idx),
                                         np.arange(idx + 1, idx * epoch_length + args.max_perturbation + 1)])
            candidates = np.clip(candidates, a_min=0, a_max=(num_epochs - 1) * epoch_length)
            indices = np.random.choice(candidates, size=args.num_sampling, replace=False)

            for i_sample in range(args.num_sampling):
                data_k.append(recordings[:, indices[i_sample]:indices[i_sample] + epoch_length].astype(np.float32))

            data_k = np.stack(data_k, axis=0)  # (num_sample, channel, length)

            np.savez(os.path.join(args.dest_path, file_prefix, f'{idx}.npz'), data_q=data_q, data_k=data_k)
            meta_df['path'].append(os.path.join(file_prefix, f'{idx}.npz'))
            meta_df['class'].append(annotations[idx])
            meta_df['patient'].append(i_patient)

    meta_df = pd.DataFrame(meta_df)
    meta_df.to_csv(os.path.join(args.dest_path, 'meta.csv'), index=False)
