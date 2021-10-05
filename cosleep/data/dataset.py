"""
@Time    : 2020/9/17 19:09
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import os
import pickle
import warnings
from typing import List, Union, Tuple

import lmdb
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import Dataset, Sampler
from tqdm.std import tqdm

from .transformation import Transformation

EPS = 1e-8


def tackle_denominator(x: np.ndarray):
    x[x == 0.0] = EPS
    return x


def tensor_standardize(x: np.ndarray, dim=-1):
    x_mean = np.expand_dims(x.mean(axis=dim), axis=dim)
    x_std = np.expand_dims(x.std(axis=dim), axis=dim)
    return (x - x_mean) / tackle_denominator(x_std)


class DEAPDataset(Dataset):
    num_subject = 32
    fs = 128

    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0, modal='eeg', transform=None,
                 return_idx=False):
        self.label_dim = label_dim
        self.transform = transform
        self.return_idx = return_idx

        assert modal in ['eeg', 'pps']

        files = sorted(os.listdir(data_path))
        assert len(files) == self.num_subject
        files = [files[i] for i in subject_list]
        all_data = []
        all_labels = []
        for a_file in tqdm(files):
            data = sio.loadmat(os.path.join(data_path, a_file))
            subject_data = data['data']  # trial x channel x data
            subject_label = data['labels']  # trial x label (valence, arousal, dominance, liking)
            # subject_data = tensor_standardize(subject_data, dim=-1)

            if modal == 'eeg':
                subject_data = subject_data[:, :32, :]
            elif modal == 'pps':
                subject_data = subject_data[:, 32:, :]
            else:
                raise ValueError

            subject_data = subject_data.reshape(*subject_data.shape[:2], subject_data.shape[-1] // self.fs,
                                                self.fs)  # (trial, channel, num_sec, time_len)
            subject_data = np.swapaxes(subject_data, 1, 2)  # (trial, num_sec, channel, time_len)
            if num_seq == 0:
                subject_data = np.expand_dims(subject_data, axis=2)
            else:
                if subject_data.shape[1] % num_seq != 0:
                    subject_data = subject_data[:, :subject_data.shape[1] // num_seq * num_seq]
                subject_data = subject_data.reshape(subject_data.shape[0], subject_data.shape[1] // num_seq, num_seq,
                                                    *subject_data.shape[-2:])

            subject_label = np.repeat(np.expand_dims(subject_label, axis=1), subject_data.shape[1], axis=1)
            subject_label = np.repeat(np.expand_dims(subject_label, axis=2), subject_data.shape[2], axis=2)

            subject_data = subject_data.reshape(subject_data.shape[0] * subject_data.shape[1], *subject_data.shape[2:])
            subject_label = subject_label.reshape(subject_label.shape[0] * subject_label.shape[1],
                                                  *subject_label.shape[2:])

            all_data.append(subject_data)
            all_labels.append(subject_label)
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if num_seq == 0:
            all_data = np.squeeze(all_data)
            # all_labels = np.squeeze(all_labels)

        self.data = all_data
        self.labels = all_labels

        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])

    def __getitem__(self, item):
        x = self.data[item].astype(np.float32)
        label = self.labels[item].astype(np.long)[:, self.label_dim]
        y = np.zeros_like(label, dtype=np.long)
        y[label >= 5] = 1

        if self.transform is not None:
            x = self.transform(x)

        x, y = torch.from_numpy(x), torch.from_numpy(y)

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)


class AMIGOSDataset(Dataset):
    num_subject = 40
    fs = 128

    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0, modal='eeg', transform=None,
                 return_idx=False):
        self.transform = transform
        self.label_dim = label_dim

        self.return_idx = return_idx

        files = sorted(os.listdir(data_path))
        assert len(files) == self.num_subject
        files = [files[i] for i in subject_list]

        all_data = []
        all_labels = []
        for a_file in tqdm(files):
            data = sio.loadmat(os.path.join(data_path, a_file))

            subject_data = []
            subject_label = []
            for i in range(data['joined_data'].shape[1]):
                trial_data = data['joined_data'][0, i]
                trial_label = data['labels_selfassessment'][0, i]
                trial_data = trial_data[:trial_data.shape[0] // self.fs * self.fs]
                trial_data = trial_data.reshape(trial_data.shape[0] // self.fs, self.fs,
                                                trial_data.shape[-1])
                trial_data = np.swapaxes(trial_data, 1, 2)

                if np.isnan(trial_data).any():
                    warnings.warn(
                        f"The array of {a_file} - {i} contains {np.sum(np.isnan(trial_data))} NaN of total {np.prod(trial_data.shape)} points, dropped.")
                    # trial_data[np.isnan(trial_data)] = 0
                    continue

                if modal == 'eeg':
                    trial_data = trial_data[:, :14]
                elif modal == 'pps':
                    trial_data = trial_data[:, 14:]
                else:
                    raise ValueError

                if trial_data.shape[0] % num_seq != 0:
                    trial_data = trial_data[:trial_data.shape[0] // num_seq * num_seq]

                # Standardize
                # mean_value = np.expand_dims(trial_data.mean(axis=0), axis=0)
                # std_value = np.expand_dims(trial_data.std(axis=0), axis=0)
                # trial_data = (trial_data - mean_value) / std_value

                trial_data = trial_data.reshape(trial_data.shape[0] // num_seq, num_seq, *trial_data.shape[1:])

                if 0 in trial_data.shape:
                    warnings.warn(f"The array of shape {data['joined_data'][0, i].shape} is too small, dropped.")
                    continue

                trial_label = np.repeat(trial_label, trial_data.shape[1], axis=0)
                trial_label = np.repeat(np.expand_dims(trial_label, axis=0), trial_data.shape[0], axis=0)

                if 0 in trial_label.shape:
                    warnings.warn(f"The label of {a_file} - {i} is malfunctioned, dropped.")
                    continue

                subject_data.append(trial_data)
                subject_label.append(trial_label)

            subject_data = np.concatenate(subject_data, axis=0)
            subject_label = np.concatenate(subject_label, axis=0)

            all_data.append(subject_data)
            all_labels.append(subject_label)
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        print(all_data.shape)
        print(all_labels.shape)

        self.data = all_data
        self.labels = all_labels

        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])

    def __getitem__(self, item):
        x = self.data[item].astype(np.float32)
        label = self.labels[item].astype(np.long)[:, self.label_dim]
        y = np.zeros_like(label, dtype=np.long)
        y[label >= 5] = 1

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)


class SleepEDFDataset(Dataset):
    def __init__(self, data_path, num_seq, transform=None, patients: List = None, modal='eeg', return_idx=False,
                 verbose=True):
        assert isinstance(patients, list)
        assert modal in ['eeg', 'pps']

        self.data_path = data_path
        self.transform = transform
        self.patients = patients
        self.modal = modal
        self.return_idx = return_idx

        self.data = []
        self.labels = []

        for i, patient in enumerate(patients):
            if verbose:
                print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            if modal == 'eeg':
                recordings = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=1)
            # elif modal == 'emg':
            #     recordings = np.expand_dims(data['emg'], axis=1)
            # elif modal == 'eog':
            #     recordings = np.expand_dims(data['eog'], axis=1)
            elif modal == 'pps':
                recordings = np.stack([data['emg'], data['eog']], axis=1)
            else:
                raise ValueError

            # print(f'[INFO] Convert the unit from V to uV...')
            recordings *= 1e6

            annotations = data['annotation']
            recordings = recordings[:(recordings.shape[0] // num_seq) * num_seq].reshape(-1, num_seq,
                                                                                         *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_seq) * num_seq].reshape(-1, num_seq)

            assert recordings.shape[:2] == annotations.shape[:2]

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = x.astype(np.float32)
        y = y.astype(np.long)

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)


class ISRUCDataset(Dataset):
    num_subject = 99
    fs = 200

    def __init__(self, data_path, num_epoch, transform=None, patients: List = None, modal='eeg', return_idx=False,
                 verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.transform = transform
        self.patients = patients
        self.modal = modal
        self.return_idx = return_idx

        assert modal in ['eeg', 'pps']

        self.data = []
        self.labels = []

        for i, patient in enumerate(patients):
            data = np.load(os.path.join(data_path, patient))
            if modal == 'eeg':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
            # elif modal == 'emg':
            #     recordings = np.stack([data['X1'], data['X3']], axis=1)
            # elif modal == 'eog':
            #     recordings = np.stack([data['LOC_A2'], data['ROC_A1']], axis=1)
            elif modal == 'pps':
                recordings = np.stack([data['X1'], data['X2'], data['X3'], data['LOC_A2'], data['ROC_A1']], axis=1)
            else:
                raise ValueError

            annotations = data['label'].flatten()

            if verbose:
                print(
                    f'[INFO] Processing the {i + 1}-th patient {patient} [shape: {recordings.shape} - {annotations.shape}] ...')

            if verbose:
                print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')
            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2], f'{patient}: {recordings.shape} - {annotations.shape}'

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = x.astype(np.float32)
        y = y.astype(np.long)

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)


class LmdbDataset(Dataset):
    def __init__(self, lmdb_path, meta_file, num_channel, length):
        self.lmdb_path = lmdb_path
        with open(meta_file, 'rb') as f:
            self.meta_info = pickle.load(f)
        self.num_channel = num_channel
        self.length = length

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.keys = self.meta_info['path']
        self.labels = torch.from_numpy(np.asarray(self.meta_info['class'], dtype=np.long))
        self.size = len(self.keys)

    def __getitem__(self, item):
        with self.env.begin(write=False) as txn:
            buffer = txn.get(self.keys[item].encode('ascii'))
        data = np.frombuffer(buffer, dtype=np.float32).copy().reshape(self.num_channel, self.length)
        data = torch.from_numpy(data)
        label = self.labels[item]

        return data, label

    def __len__(self):
        return self.size


class SleepDataset(Dataset):
    def __init__(self, data_path, data_name, num_epoch, patients: List = None, preprocessing: str = 'none', modal='eeg',
                 return_idx=False, verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
        self.patients = patients
        self.preprocessing = preprocessing
        self.modal = modal
        self.return_idx = return_idx

        assert preprocessing in ['none', 'quantile', 'standard']
        assert modal in ['eeg', 'emg', 'eog']

        self.data = []
        self.labels = []

        for i, patient in enumerate(tqdm(patients, desc='::: LOADING EEG DATA :::')):
            # if verbose:
            #     print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            if data_name == 'sleepedf':
                if modal == 'eeg':
                    recordings = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=1)
                elif modal == 'emg':
                    recordings = np.expand_dims(data['emg'], axis=1)
                elif modal == 'eog':
                    recordings = np.expand_dims(data['eog'], axis=1)
                else:
                    raise ValueError

                annotations = data['annotation']
            elif data_name == 'isruc':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
                annotations = data['label'].flatten()
            else:
                raise ValueError

            if preprocessing == 'standard':
                # print(f'[INFO] Applying standard scaler...')
                # scaler = StandardScaler()
                # recordings_old = recordings
                # recordings = []
                # for j in range(recordings_old.shape[0]):
                #     recordings.append(scaler.fit_transform(recordings_old[j].transpose()).transpose())
                # recordings = np.stack(recordings, axis=0)

                recordings = tensor_standardize(recordings, dim=-1)
            elif preprocessing == 'quantile':
                # print(f'[INFO] Applying quantile scaler...')
                scaler = QuantileTransformer(output_distribution='normal')
                recordings_old = recordings
                recordings = []
                for j in range(recordings_old.shape[0]):
                    recordings.append(scaler.fit_transform(recordings_old[j].transpose()).transpose())
                recordings = np.stack(recordings, axis=0)
            else:
                # print(f'[INFO] Convert the unit from V to uV...')
                recordings *= 1e6

            # if verbose:
            #     print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')
            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2]

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return """
**********************************************************************
Dataset Summary:
Preprocessing: {}
# Instance: {}
Shape of an Instance: {}
Selected patients: {}
**********************************************************************
            """.format(self.preprocessing, len(self.data), self.full_shape, self.patients)


class SleepDataset2d(Dataset):
    def __init__(self, data_path, data_name, num_epoch, patients: List = None, preprocessing: str = 'none',
                 verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
        self.patients = patients
        self.preprocessing = preprocessing

        assert preprocessing in ['none', 'quantile', 'standard']

        self.data = []
        self.labels = []

        for i, patient in enumerate(patients):
            if verbose:
                print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            if data_name == 'sleepedf':
                recordings = data['data']
                annotations = data['annotation']
            elif data_name == 'isruc':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
                annotations = data['label'].flatten()
            else:
                raise ValueError

            assert preprocessing == 'none'

            if verbose:
                print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')
            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2]

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))

        return x, y

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return """
    **********************************************************************
    Dataset Summary:
    Preprocessing: {}
    # Instance: {}
    Shape of an Instance: {}
    Selected patients: {}
    **********************************************************************
                """.format(self.preprocessing, len(self.data), self.full_shape, self.patients)


class SleepDatasetImg(Dataset):
    def __init__(self, data_path, data_name, num_epoch, transform, patients: List = None, return_idx=False,
                 verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
        self.num_epoch = num_epoch
        self.transform = transform
        self.patients = patients
        self.return_idx = return_idx

        self.data = []
        self.labels = []

        for i, patient in enumerate(tqdm(patients, desc='::: LOADING IMG DATA :::')):
            # if verbose:
            #     print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            recordings = data['data']
            annotations = data['label']

            # if verbose:
            #     print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')
            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2]

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        # x = torch.stack([self.transform(x[0]), self.transform(x[1])], dim=0)
        # print(x.shape, x[:, 0].shape, '-------------')
        if self.data_name == 'isruc':
            x_all = []
            for k in range(x.shape[1]):
                x_all.append(torch.stack([self.transform(Image.fromarray(x[i][k])) for i in range(x.shape[0])], dim=0))
            x = torch.cat(x_all, dim=1)
        else:
            x1 = torch.stack([self.transform(Image.fromarray(x[i][0])) for i in range(x.shape[0])],
                             dim=0)  # TODO for temp
            x2 = torch.stack([self.transform(Image.fromarray(x[i][1])) for i in range(x.shape[0])], dim=0)
            x = torch.cat([x1, x2], dim=1)
        y = torch.from_numpy(y.astype(np.long))

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return """
    **********************************************************************
    Dataset Summary:
    # Instance: {}
    Shape of an Instance: {}
    Selected patients: {}
    **********************************************************************
                """.format(len(self.data), self.full_shape, self.patients)


class LmdbDatasetWithEdges(Dataset):
    def __init__(self, lmdb_path, meta_file, num_channel, size: Union[int, Tuple[int]], num_extend,
                 patients: List = None,
                 transform: Transformation = None, return_idx: bool = False):
        self.lmdb_path = lmdb_path
        with open(meta_file, 'rb') as f:
            self.meta_info = pickle.load(f)
        self.num_channel = num_channel
        if isinstance(size, int):
            size = (size,)
            self.full_shape = (num_channel, *size)
        elif isinstance(size, tuple):
            assert len(size) == 2
            self.full_shape = (num_channel, *size)
        else:
            raise ValueError('Invalid` length`!')
        self.size = size
        self.num_extend = num_extend
        self.transform = transform
        self.patients = patients
        self.return_idx = return_idx

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        if self.patients is None:
            self.keys = self.meta_info['path']
        else:
            self.keys = []
            for i, p in enumerate(self.meta_info['path']):
                if self.meta_info['patient'][i] in self.patients:
                    self.keys.append(p)

        self.labels = torch.from_numpy(np.asarray(self.meta_info['class'], dtype=np.long))
        self.len = len(self.keys)

    def __getitem__(self, item):
        with self.env.begin(write=False) as txn:
            buffer = txn.get(self.keys[item].encode('ascii'))
        data = np.frombuffer(buffer, dtype=np.float32).copy().reshape(*self.full_shape[:-1],
                                                                      self.full_shape[-1] + 2 * self.num_extend)
        data = {'head': data[..., :self.num_extend],
                'mid': data[..., self.num_extend:-self.num_extend],
                'tail': data[..., -self.num_extend:]}

        if self.transform is not None:
            data = self.transform(data)

        if isinstance(data, list):
            data = [torch.from_numpy(data[0]['mid'].astype(np.float32)),
                    torch.from_numpy(data[1]['mid'].astype(np.float32))]
        else:
            data = torch.from_numpy(data['mid'].astype(np.float32))

        label = self.labels[item]

        if self.return_idx:
            return data, label, item
        else:
            return data, label

    def __len__(self):
        return self.len

    def __repr__(self):
        return """
**********************************************************************
Dataset Summary:
# Instance: {}
Shape of an Instance: {}
Selected patients: {}
**********************************************************************
        """.format(self.len, self.full_shape, self.patients)


class SleepDatasetSampling(Dataset):
    def __init__(self, data_path, data_name, num_sampling, mode, dis, patients: List = None,
                 preprocessing: str = 'none', verbose=True):
        assert isinstance(patients, list)
        assert mode in ['pair', 'triplet']

        self.data_path = data_path
        self.data_name = data_name
        self.patients = patients
        self.num_sampling = num_sampling
        self.mode = mode
        self.dis = dis
        self.preprocessing = preprocessing

        assert preprocessing in ['none', 'quantile', 'standard']

        self.data = []
        self.labels = []
        self.annotations = []
        self.indices = []

        for i, patient in enumerate(patients):
            if verbose:
                print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            if data_name == 'sleepedf':
                recordings = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=1)
                annotations = data['annotation']
            elif data_name == 'isruc':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
                annotations = data['label'].flatten()
            else:
                raise ValueError

            if preprocessing == 'standard':
                print(f'[INFO] Applying standard scaler...')

                recordings = tensor_standardize(recordings, dim=-1)
            elif preprocessing == 'quantile':
                print(f'[INFO] Applying quantile scaler...')
                scaler = QuantileTransformer(output_distribution='normal')
                recordings_old = recordings
                recordings = []
                for j in range(recordings_old.shape[0]):
                    recordings.append(scaler.fit_transform(recordings_old[j].transpose()).transpose())
                recordings = np.stack(recordings, axis=0)
            else:
                print(f'[INFO] Convert the unit from V to uV...')
                recordings *= 1e6

            if verbose:
                print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')

            num_to_sample = num_sampling // len(patients)
            if mode == 'pair':
                anchor_indices = np.random.choice(np.arange(recordings.shape[0]), size=num_to_sample // 2,
                                                  replace=False)
                labels = []
                for k in tqdm(anchor_indices):
                    pos_idx = np.random.randint(np.clip(k - dis, 0, recordings.shape[0]),
                                                np.clip(k + dis, 0, recordings.shape[0]))
                    neg_idx = np.ones(recordings.shape[0], dtype=np.bool)
                    neg_idx[np.arange(np.clip(k - dis, 0, recordings.shape[0]),
                                      np.clip(k + dis, 0, recordings.shape[0]))] = False
                    neg_idx = np.arange(recordings.shape[0])[neg_idx]
                    neg_idx = np.random.choice(neg_idx, size=1)[0]
                    # self.data.append(np.stack([recordings[k], recordings[pos_idx]], axis=0))
                    self.indices.append(np.array([k, pos_idx]))
                    labels.append(1)
                    # self.data.append(np.stack([recordings[k], recordings[neg_idx]], axis=0))
                    self.indices.append(np.array([k, neg_idx]))
                    labels.append(-1)
                labels = np.array(labels)
            else:
                anchor_indices = np.random.choice(np.arange(0, recordings.shape[0] - dis), size=num_to_sample // 2,
                                                  replace=False)
                labels = []
                for k in tqdm(anchor_indices):
                    pos_idx = np.random.randint(k + 1, k + dis)
                    neg_idx = np.ones(recordings.shape[0], dtype=np.bool)
                    neg_idx[np.arange(k, k + dis + 1)] = False
                    neg_idx = np.arange(recordings.shape[0])[neg_idx]
                    neg_idx = np.random.choice(neg_idx, size=1)[0]
                    # self.data.append(np.stack([recordings[k], recordings[pos_idx], recordings[k + dis]], axis=0))
                    self.indices.append(np.array([k, pos_idx, k + dis]))
                    labels.append(1)
                    # self.data.append(np.stack([recordings[k], recordings[neg_idx], recordings[k + dis]], axis=0))
                    self.indices.append(np.array([k, neg_idx, k + dis]))
                    labels.append(-1)
            self.labels.append(labels)
            self.data.append(recordings)
            self.annotations.append(annotations)

        self.data = np.concatenate(self.data)
        self.indices = np.stack(self.indices, axis=0)
        self.labels = torch.from_numpy(np.concatenate(self.labels).astype(np.long))
        self.annotations = np.concatenate(self.annotations)
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        # x = self.data[item]
        idx = self.indices[item]
        y = self.labels[item]

        x = torch.from_numpy(self.data[idx].astype(np.float32))

        return x, y

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return """
**********************************************************************
Dataset Summary:
Preprocessing: {}
# Instance: {}
Shape of indices: {}
Shape of an instance: {}
Selected patients: {}
**********************************************************************
            """.format(self.preprocessing, len(self.indices), self.indices.shape, self.full_shape, self.patients)


class TwoDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)

        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, item):
        return (*self.dataset1[item], *self.dataset2[item])

    def __len__(self):
        return len(self.dataset1)


class ShuffleSampler(Sampler):
    def __init__(self, data_source, total_len):
        super(ShuffleSampler, self).__init__(data_source)

        self.total_len = total_len

    def __iter__(self):
        yield from torch.randperm(self.total_len)

    def __len__(self):
        return self.total_len
