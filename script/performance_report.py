"""
@Time    : 2020/12/31 9:54
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : performance_report.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--base-path', type=str, required=True)
    parser.add_argument('-s', '--suffix', type=str, required=True)

    args_parsed = parser.parse_args()

    return args_parsed


if __name__ == '__main__':
    args = parse_args()

    files = glob.glob(os.path.join(args.base_path, args.suffix + '*'))
    files = sorted(files)

    results = []
    for i, a_file in enumerate(files):
        with open(os.path.join(a_file, f'statistics_{i}.pkl'), 'rb') as f:
            data = pickle.load(f)

        results.append(data['performance'])

    df = pd.DataFrame(np.concatenate([res.values for res in results], axis=1), index=results[0].index,
                      columns=[f'{i}' for i in range(len(results))])
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)
