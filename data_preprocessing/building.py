import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_parser import Parser
import os
from extracting import Extractor
import tensorflow as tf
from formatting import *
from data_info import Info
from typing import Tuple, Any
from preprocessing import virtual, smooth, rescale, impute, choose, finalize, clean
from splitting import split_train_test
from generating import batch_concat, Concatenator, to_generator, prefetch
from transformers import Temporal
import pickle

START = 2000
SEARCH = 1435993199433.0
SEED = 43


def show_pd(x: pd.DataFrame, pos1: str, pos2: str,
            dataset: str = 'rwhar', subject: int = 1, activity: str = 'walking',
            length: int = 100, start: int = 1000):
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

        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(t, w1)
        axs[1].plot(t, w2)

        axs[0].set_ylabel(pos1)
        axs[1].set_ylabel(pos2)

        features = [pos_ft.replace(pos1 + '_', '') for pos_ft in features1]

        plt.legend(features)
        plt.show()


def show_np(S: Tuple[np.ndarray, np.ndarray], dicts: List, pos1: str, pos2: str,
            dataset: str = 'rwhar', subject: int = 1, activity: str = 'walking',
            search: int = 1000):
    X, T = S
    ds_dict, act_dict = dicts

    idx = np.argwhere(np.all(T[:, :, 1] == ds_dict[dataset], axis=1)).squeeze()
    X = X[idx]
    T = T[idx]

    idx = np.argwhere(np.all(T[:, :, 2] == subject, axis=1)).squeeze()
    X = X[idx]
    T = T[idx]

    idx = np.argwhere(np.all(T[:, :, 3] == act_dict[activity], axis=1)).squeeze()
    X = X[idx]
    T = T[idx]

    idx = np.argwhere((T[:, 0, -1] < search) & (search < T[:, -1, -1])).squeeze()
    X = X[idx[1]]
    T = T[idx[1]]

    t = T[:, -1]
    w1 = X[0]
    w2 = X[1]

    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(t, w1)
    axs[1].plot(t, w2)

    axs[0].set_ylabel(pos1)
    axs[1].set_ylabel(pos2)

    plt.show()


def show_tf(data):
    pass


class Builder:

    def __init__(self, regenerate: bool = False):
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
            self.data = Bob()

        else:
            filename = os.path.join(self.path, 'gaitCLR.csv')
            self.data = pd.read_csv(filename)

        self.pairs = [['left_lower_arm', 'left_lower_leg'],
                      ['right_lower_arm', 'right_lower_leg'],
                      ['dominant_lower_arm', 'dominant_lower_leg']]
        self.dicts = None

    def prepare_data(self, verbose: bool = False) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        data = self.data.copy()
        data = data.drop(data.columns[0], axis=1)

        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', START)

        data = choose(data, self.conf.in_datasets, self.conf.in_positions)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', START)

        data = impute(data, self.conf.cleaner)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', START)

        data = virtual(data, self.conf.prod_features, self.conf.fs)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', START)

        data = smooth(data, self.conf.filter, self.conf.filter_window)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', START)

        data = rescale(data, self.conf.rescaler)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', START)

        train, test = split_train_test(data, self.conf.split_type, self.conf.hold_out, SEED)
        if verbose:
            show_pd(data, 'left_lower_arm', 'left_lower_leg', START)

        train, self.dicts = finalize(train, self.conf.length, self.conf.step, self.pairs, True)
        test, _ = finalize(test, self.conf.length, self.conf.step, self.pairs, True)
        if verbose:
            show_np(train, self.dicts, 'left_lower_arm', 'left_lower_leg', SEARCH)

        train = clean(train, self.conf.selection_method, self.conf.tolerance)
        test = clean(test, self.conf.selection_method, self.conf.tolerance)
        if verbose:
            show_np(train, self.dicts, 'left_lower_arm', 'left_lower_leg', SEARCH)

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
                   'dicts': self.dicts}

        output = open(save_path, 'wb')
        pickle.dump(my_data, output)

        output.close()

    def load_data(self, path: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        load_path = os.path.join(path, 'data.pkl')

        pkl_file = open(load_path, 'rb')
        my_data = pickle.load(pkl_file)
        train = my_data['train_X'], my_data['train_T']
        test = my_data['test_X'], my_data['test_T']
        dicts = my_data['dicts']

        pkl_file.close()

        return train, test, dicts

    def get_transformers(self):
        self.transformer = Temporal()
        self.input_shape = self.transformer.get_shape()
        self.input_type = self.transformer.get_type()

    def __call__(self, path: Optional[str] = None, verbose: bool = False):

        if self.conf.load_data:
            train, test, self.dicts = self.load_data(path)

        else:
            train, test = self.prepare_data(verbose)
            self.save_data(train, test, path)

        self.get_transformers()

        train = batch_concat(train, self.conf.batch_method, self.conf.batch_size, self.dicts, self.transformer,
                             training=True, seed=SEED)
        test = batch_concat(test, self.conf.batch_method, self.conf.batch_size, self.dicts, self.transformer,
                            training=False, seed=SEED)

        train = to_generator(train, self.input_type, self.input_shape)
        test = to_generator(test, self.input_type, self.input_shape)

        train = prefetch(train)
        test = prefetch(test)

        for l, batch in enumerate(train):
            anchor, target = batch
            if l == 0:
                fig, axs = plt.subplots(2, sharex=True)
                axs[0].plot(anchor[0])
                axs[1].plot(target[0])
                plt.show()
                break

        return train, test


if __name__ == '__main__':
    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, 'save', 'data', 'no_model' + '_data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    BD = Builder(regenerate=False)
    BD(path=data_dir, verbose=False)
