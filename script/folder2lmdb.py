"""
@Time    : 2020/12/8 12:22
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : folder2lmdb.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.append('../')
from cosleep.utils.data import folder_to_lmdb
from cosleep.data import LmdbDataset


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--dest-file', type=str, required=True)
    parser.add_argument('--commit-interval', type=int, default=100)
    parser.add_argument('--test', action='store_true')

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

    folder_to_lmdb(args.data_path, args.dest_file, args.commit_interval)

    if args.test:
        dataset = LmdbDataset(args.dest_file, os.path.join(args.data_path, 'meta.pkl'), num_channel=2)
        with open(os.path.join(args.data_path, 'meta.pkl'), 'rb') as f:
            meta_info = pickle.load(f)
        data = np.load(os.path.join(args.data_path, meta_info['path'][0]))

        ori_data = data['data'].astype(np.float32)
        lmdb_data = dataset[0][0].numpy().astype(np.float32)
        print(ori_data.shape, ori_data)
        print(lmdb_data.shape, lmdb_data)

        assert (ori_data == lmdb_data).all()
