"""
@Time    : 2020/12/14 23:03
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : epoch_split.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import pickle
import warnings

import numpy as np
from tqdm.std import tqdm

UNIT_CONVERT_FACTOR = 1e6  # V -> uV


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--dest-path', type=str, required=True)
    parser.add_argument('--data-name', type=str, required=True)
    parser.add_argument('--convert-unit', action='store_true')
    parser.add_argument('--extend-edges', action='store_true')
    parser.add_argument('--extend-num', type=int, default=None)

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

    assert args.data_name in ['sleepedf_39', 'sleepedf_153', 'sleepedf_39_hht',
                              'sleepedf_153_hht', 'isruc']

    if args.extend_edges and args.extend_num is None:
        raise ValueError('Console parameter `extend_num` must be specified when `extend_edges` enabled!')

    if not os.path.exists(args.dest_path):
        warnings.warn(f'The path {args.dest_path} does not existed, created.')
        os.makedirs(args.dest_path)

    meta_info = {'path': [], 'class': [], 'patient': [], 'convert_unit': args.convert_unit,
                 'extend_edges': args.extend_edges, 'extend_num': args.extend_num}

    file_list = os.listdir(args.data_path)
    for i_patient, file in enumerate(file_list):
        print(f'Processing {file}...')
        file_name = os.path.join(args.data_path, file)
        file_prefix = file.split('.')[0]

        if not os.path.exists(os.path.join(args.dest_path, file_prefix)):
            os.makedirs(os.path.join(args.dest_path, file_prefix))

        data = np.load(file_name)

        if args.data_name in ['sleepedf_39', 'sleepedf_153']:
            recordings = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=1)  # (num_epoch, channel, length)
            annotations = data['annotation']
        elif args.data_name in ['sleepedf_39_hht', 'sleepedf_153_hht']:
            recordings = data['data']
            annotations = data['annotation']
        elif args.data_name in ['isruc']:
            recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                   data['O1_A2'], data['O2_A1']], axis=1)
            annotations = data['label'].reshape(-1)
        else:
            raise ValueError

        if args.convert_unit:
            warnings.warn('You choose to convert the unit from V to uV...')
            recordings *= UNIT_CONVERT_FACTOR

        for idx in tqdm(range(recordings.shape[0])):
            if args.extend_edges:
                if idx == 0:
                    # Pad heading with zeros
                    head = np.zeros(shape=(*(recordings[idx].shape[:-1]), args.extend_num))
                    mid = recordings[idx]
                    # (num_epoch, channel, time) for 1d inputs
                    # (num_epoch, channel, freq, time) for 2d inputs
                    tail = recordings[idx + 1][..., :args.extend_num]  # God, numpy's indexing is so awesome!
                elif idx == recordings.shape[0] - 1:
                    # Pad tail with zeros
                    head = recordings[idx - 1][..., -args.extend_num:]
                    mid = recordings[idx]
                    tail = np.zeros(shape=(*(recordings[idx].shape[:-1]), args.extend_num))
                else:
                    # No padding
                    head = recordings[idx - 1][..., -args.extend_num:]
                    mid = recordings[idx]
                    tail = recordings[idx + 1][..., :args.extend_num]
                sample = np.concatenate([head, mid, tail], axis=-1)
            else:
                sample = recordings[idx]
            np.savez(os.path.join(args.dest_path, file_prefix, f'{idx}.npz'), data=sample)
            meta_info['path'].append(os.path.join(file_prefix, f'{idx}.npz'))
            meta_info['class'].append(annotations[idx])
            meta_info['patient'].append(i_patient)

    with open(os.path.join(args.dest_path, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta_info, f)
