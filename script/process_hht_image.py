"""
@Time    : 2021/4/26 21:33
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : process_hht_image.py
@Software: PyCharm
@Desc    : 
"""
import os
import warnings

import numpy as np
from PIL import Image
from tqdm.std import tqdm

BASE_PATH = '/data/DataHub/SleepClassification/sleepedf153/deprecated/hht_image'
DEST_PATH = '/data/DataHub/SleepClassification/sleepedf153/sleepedf153_img'

if __name__ == '__main__':
    subjects1 = sorted(os.listdir(os.path.join(BASE_PATH, 'fpz_cz')))
    subjects2 = sorted(os.listdir(os.path.join(BASE_PATH, 'pz_oz')))
    assert subjects1 == subjects2
    subjects = subjects1
    # pz_oz_subjects = sorted(os.listdir('/data/DataHub/SleepClassification/sleepedf153/deprecated/hht_image/pz_oz'))

    for subject in subjects:
        files1 = sorted(os.listdir(os.path.join(BASE_PATH, 'fpz_cz', subject)))
        files2 = sorted(os.listdir(os.path.join(BASE_PATH, 'pz_oz', subject)))
        assert files1 == files2
        files = files1

        assert len(files) > 0, f'{subject}'

        if len(files) < 700:
            warnings.warn(f'Subject {subject} contains less than 700 files.')

        subject_data = []
        for a_file in tqdm(files, desc=subject):
            fpz_cz_img = Image.open(os.path.join(BASE_PATH, 'fpz_cz', subject, a_file))
            pz_oz_img = Image.open(os.path.join(BASE_PATH, 'pz_oz', subject, a_file))
            fpz_cz_array = np.asarray(fpz_cz_img)
            pz_oz_array = np.asarray(pz_oz_img)

            data = np.stack([fpz_cz_array, pz_oz_array], axis=0)
            subject_data.append(data)
        subject_data = np.stack(subject_data, axis=0)
        label = np.load(os.path.join(BASE_PATH, '..', '..', 'sleepedf153', f'{subject}.npz'))['annotation']
        np.savez(os.path.join(DEST_PATH, f'{subject}.npz'), data=subject_data, label=label)
