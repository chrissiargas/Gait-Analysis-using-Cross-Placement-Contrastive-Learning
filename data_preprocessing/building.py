import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_parser import Parser
import os
from extracting import Extractor
from formatting import *
from data_info import Info
from typing import Tuple, Any
from preprocessing import virtual, smooth, rescale, impute, choose, finalize, clean
from splitting import split_train_test
from generating import batch_concat, Concatenator, to_generator, prefetch
from transformers import Temporal
import pickle
from transformations import time_warp, random_scale, add_noise

START = 2000
SEARCH = 40100
SEED = 19


def show_pd(x: pd.DataFrame, pos1: str, pos2: str,
            dataset: str = 'marea', subject: int = 1, activity: str = 'treadmill_walking',
            length: int = 100, start: int = 1000,
            transformations: Optional[List[str]] = None):
    x = x[x['dataset'] == dataset]
    x = x[x['subject'] == subject]
    x = x[x['activity'] == activity]


    all_acc = x.columns[x.columns.str.contains("acc")]
    positions = [acc_name[:-6] for acc_name in all_acc]
    positions = list(set(positions))

    if pos1 in positions and pos2 in positions:
        features1 = x.columns[x.columns.str.contains(pos1)]
        features2 = x.columns[x.columns.str.contains(pos2)]

        time = x['timestamp'].values
        sig1 = x[features1].values
        sig2 = x[features2].values

        t = time[start: start + length]
        w1 = sig1[start: start + length]
        w2 = sig2[start: start + length]

        w1 = np.array(w1, dtype=np.float32)
        w2 = np.array(w2, dtype=np.float32)

        if transformations is not None:
            aug_w1 = w1.copy()
            aug_w2 = w2.copy()
            for transformation in transformations:
                if transformation == 'jitter':
                    aug_w1 = np.expand_dims(aug_w1, axis=0)[..., :3]
                    aug_w2 = np.expand_dims(aug_w2, axis=0)[..., :3]

                    aug_w1 = add_noise(aug_w1).squeeze()
                    aug_w2 = add_noise(aug_w2).squeeze()
                elif transformation == 'scale':
                    aug_w1 = np.expand_dims(aug_w1, axis=0)[..., :3]
                    aug_w2 = np.expand_dims(aug_w2, axis=0)[..., :3]

                    aug_w1 = random_scale(aug_w1).squeeze()
                    aug_w2 = random_scale(aug_w2).squeeze()

            w1 = np.concatenate((aug_w1, w1[:, [-2]]), axis=-1)
            w2 = np.concatenate((aug_w2, w2[:, [-2]]), axis=-1)

            w1 = np.concatenate((w1, np.sqrt(np.sum(np.square(aug_w1), axis=-1))[:, np.newaxis]), axis=-1)
            w2 = np.concatenate((w2, np.sqrt(np.sum(np.square(aug_w2), axis=-1))[:, np.newaxis]), axis=-1)

        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(t, w1)
        axs[1].plot(t, w2)

        axs[0].set_ylabel(pos1)
        axs[1].set_ylabel(pos2)

        features = [pos_ft.replace(pos1 + '_', '') for pos_ft in features1]

        plt.legend(features)
        plt.show()


def show_np(S: Tuple[dict, dict], dicts: List, anchor: str, target: str,
            dataset: str = 'marea', subject: int = 1, activity: str = 'treadmill_walking',
            search: float = 0.):
    X, T = S
    X, T = X[target], T[target]
    ds_dict, act_dict = dicts

    idx = np.argwhere(np.all(T[:, :, 0] == ds_dict[dataset], axis=1)).squeeze()
    X = X[idx]
    T = T[idx]

    idx = np.argwhere(np.all(T[:, :, 1] == subject, axis=1)).squeeze()
    X = X[idx]
    T = T[idx]

    idx = np.argwhere(np.all(T[:, :, 2] == act_dict[activity], axis=1)).squeeze()
    X = X[idx]
    T = T[idx]

    idx = np.argwhere((T[:, 0, 3] < search) & (search < T[:, -1, 3])).squeeze()
    X = X[idx[1]]
    T = T[idx[1]]

    t = T[:, 3]
    w_anchor = X[0]
    w_target = X[1]

    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(t, w_anchor)
    axs[1].plot(t, w_target)

    axs[0].set_ylabel(anchor)
    axs[1].set_ylabel(target)

    plt.show()


def show_tf(data):
    pass


class Builder:

    def __init__(self, regenerate: bool = False):
        self.test_dicts = None
        self.train_dicts = None
        self.input_type = None
        self.input_shape = None
        self.transformer = None
        config = Parser()
        config.get_args()
        self.conf = config

        self.path = os.path.join(
            os.path.expanduser('~'),
            config.path,
            'gait-CLR'
        )

        self.info = Info()

        if regenerate:
            Bob = Extractor()
            self.data = Bob(load=True)

        else:
            filename = os.path.join(self.path, 'gaitCLR.csv')
            self.data = pd.read_csv(filename)

        self.anchor = 'left_lower_arm'
        self.pairs = [[self.anchor, 'left_lower_leg'],
                      [self.anchor, 'right_lower_leg']]
        self.targets = [pair[1] for pair in self.pairs]

        self.train_batches = {target: 0 for target in self.targets}
        self.test_batches = {target: 0 for target in self.targets}

        self.same_ds = True if self.conf.neg_ds == 'same' else False
        self.same_sub = True if self.conf.neg_sub == 'same' else False
        self.same_act = True if self.conf.neg_act == 'same' else False

    def prepare_data(self, verbose: bool = False) -> \
            Tuple[Tuple[dict, dict], Tuple[dict, dict]]:

        data = self.data.copy()
        data = data.drop(data.columns[0], axis=1)

        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', start=START)

        data = choose(data, self.conf.in_datasets, self.conf.in_positions, self.conf.in_activities)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', start=START)

        data = impute(data, self.conf.cleaner)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', start=START)

        data = virtual(data, self.conf.prod_features, self.conf.fs)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', start=START)

        data = smooth(data, self.conf.filter, self.conf.filter_window)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', start=START)

        data = rescale(data, self.conf.rescaler)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', start=START)

        train, test = split_train_test(data, self.conf.split_type, self.conf.hold_out, SEED)
        if verbose:
            show_pd(train, 'left_lower_arm', 'left_lower_leg', start=START)

        train, self.train_dicts = finalize(train, self.conf.length + self.conf.shift_length, self.conf.step, self.pairs, True)
        test, self.test_dicts = finalize(test, self.conf.length + self.conf.shift_length, self.conf.step, self.pairs, True)
        if verbose:
            show_np(train, self.train_dicts, 'left_lower_arm', 'left_lower_leg', search=SEARCH)

        train = clean(train, self.conf.selection_method, self.conf.tolerance)
        test = clean(test, self.conf.selection_method, self.conf.tolerance)
        if verbose:
            show_np(train, self.train_dicts, 'left_lower_arm', 'left_lower_leg', search=SEARCH)

        if 'time_warp' in self.conf.common_augmentations:
            X_train, T_train = train
            for target in self.targets:
                X_train[target] = time_warp(X_train[target])

            train = X_train, T_train

        return train, test

    def load_data(self, path):
        pass

    def save_data(self,
                  train: Tuple[np.ndarray, np.ndarray],
                  test: Tuple[np.ndarray, np.ndarray],
                  path: str):

        save_path = os.path.join(path, 'data.pkl')

        my_data = {'train_X': train[0],
                   'train_T': train[1],
                   'test_X': test[0],
                   'test_T': test[1],
                   'train_dicts': self.train_dicts,
                   'test_dicts': self.test_dicts}

        output = open(save_path, 'wb')
        pickle.dump(my_data, output)

        output.close()

    def load_data(self, path: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        load_path = os.path.join(path, 'data.pkl')

        pkl_file = open(load_path, 'rb')
        my_data = pickle.load(pkl_file)
        train = my_data['train_X'], my_data['train_T']
        test = my_data['test_X'], my_data['test_T']
        self.train_dicts = my_data['train_dicts']
        self.test_dicts = my_data['test_dicts']

        pkl_file.close()

        return train, test

    def get_transformers(self):
        self.transformer = Temporal()
        self.input_shape = self.transformer.get_shape()
        self.input_type = self.transformer.get_type()

    def __call__(self, path: Optional[str] = None, verbose: bool = False):

        if self.conf.load_data:
            train, test = self.load_data(path)

        else:
            train, test = self.prepare_data(verbose)
            self.save_data(train, test, path)

        self.get_transformers()

        train = batch_concat(train, self.conf.batch_method, self.conf.batch_size, self.train_dicts, self.transformer,
                             training=True, seed=SEED,
                             same_ds=self.same_ds, same_sub=self.same_sub, same_act=self.same_act)

        for target in self.targets:
            self.train_batches[target] = train[target].N_batches

        test = batch_concat(test, self.conf.batch_method, self.conf.batch_size, self.test_dicts, self.transformer,
                            training=False, seed=SEED,
                            same_ds=self.same_ds, same_sub=self.same_sub, same_act=self.same_act)

        for target in self.targets:
            self.test_batches[target] = test[target].N_batches

        print(self.train_batches)
        print(self.test_batches)

        for target in self.targets:
            train[target] = to_generator(train[target], self.input_type, self.input_shape)
            test[target] = to_generator(test[target], self.input_type, self.input_shape)

            train[target] = prefetch(train[target])
            test[target] = prefetch(test[target])

        return train, test


if __name__ == '__main__':
    data_dir = os.path.join('save', 'data', 'no_model' + '_data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    BD = Builder(regenerate=False)
    BD(path=data_dir, verbose=False)
