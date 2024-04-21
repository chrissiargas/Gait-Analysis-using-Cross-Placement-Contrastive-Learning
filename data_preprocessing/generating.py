import numpy as np
from math import ceil, floor
from typing import Optional, Tuple, List
from transformers import Temporal
import tensorflow as tf
from config_parser import Parser


Aug_ANCHOR = False
Aug_TARGET = True


class Batcher:
    def __init__(self, S: Tuple[np.ndarray, np.ndarray], method: str, batch_size: int,
                 transformer: Optional[Temporal] = None, augment: bool = False,
                 seed: Optional[int] = None):
        X, T = S
        self.data = X.copy()
        self.info = T.copy()
        self.method = method
        self.batch_size = batch_size
        self.seed = seed
        self.n_wins = self.data.shape[0]
        self.length = self.data.shape[1]
        self.channels = self.data.shape[2]
        self.n_batches = floor(self.n_wins / self.batch_size)
        self.rng = np.random.default_rng(seed=seed)
        self.shuffled_data = None
        self.output_data = None
        self.transformer = transformer
        self.augment = augment

    def reset_data(self):
        index_list = np.arange(self.n_wins, dtype=int)
        self.rng.shuffle(index_list)
        self.shuffled_data = self.data[index_list]
        self.output_data = self.shuffled_data

    def __len__(self):
        return self.n_batches

    def get_shape(self):
        return self.transformer.get_shape()

    def __iter__(self):
        self.reset_data()

        def gen():
            for i in range(self.n_batches):
                if self.transformer:
                    batches = self.transformer(self.output_data[i * self.batch_size: (i + 1) * self.batch_size],
                                               augment=self.augment)
                else:
                    batches = self.output_data[i * self.batch_size: (i + 1) * self.batch_size]

                yield batches

        return gen()


class Zipper:
    def __init__(self, batched_datasets: List[Batcher], stack: bool = True, stack_axis: int = 0):
        self.datasets = batched_datasets
        self.stack = stack
        self.stack_axis = stack_axis
        self.n_batches = self.datasets[0].n_batches
        assert self.n_batches == self.datasets[1].n_batches
        self.batch_size = self.datasets[0].get_shape()[0]
        self.length = self.datasets[0].get_shape()[1]
        self.channels = self.datasets[0].get_shape()[2]

    def __iter__(self):
        def gen():
            if self.stack:
                for zipped_batch in zip(*tuple(self.datasets)):
                    yield np.stack(zipped_batch, axis=self.stack_axis)

            else:
                for zipped_batch in zip(*tuple(self.datasets)):
                    yield zipped_batch

        return gen()


class Concatenator:
    def __init__(self, zipped_datasets: List[Zipper]):
        self.datasets = zipped_datasets
        self.n_dss = len(self.datasets)
        self.N_batches = sum(zipped_dataset.n_batches for zipped_dataset in zipped_datasets)
        self.batch_size = self.datasets[0].batch_size
        self.length = self.datasets[0].length
        self.channels = self.datasets[0].channels

    def get_shape(self):
        return (self.N_batches, 2, self.batch_size, self.length, self.channels)

    def __iter__(self):
        def gen():
            for dataset in self.datasets:
                for batch in dataset:
                    yield batch

        return gen()


def batch_concat(S: Tuple[dict, dict], method: str, batch_size: int,
                 dicts: List[dict], transformer: Optional[Temporal] = None,
                 same_ds: bool = True, same_sub: bool = True, same_act: bool = True,
                 training: bool = False, seed: Optional[int] = None) -> dict:
    X, T = S
    ds_dict, act_dict = dicts
    X = X.copy()
    T = T.copy()

    aug_anchor = Aug_ANCHOR & training
    aug_target = Aug_TARGET & training

    batches = {}
    for tp in X.keys():
        groups = []
        X_tmp = X[tp]
        T_tmp = T[tp]

        if same_ds:
            for ds, ds_id in ds_dict.items():
                idx = np.argwhere(np.all(T_tmp[:, :, 0] == ds_id, axis=1)).squeeze()
                ds_X = X_tmp[idx]
                ds_T = T_tmp[idx]

                if same_sub:
                    sub_arr = ds_T[:, :, 1]
                    sub_ids = np.unique(sub_arr)

                    for sub_id in sub_ids:
                        idx = np.argwhere(np.all(ds_T[:, :, 1] == sub_id, axis=1)).squeeze()
                        ds_sub_X = ds_X[idx]
                        ds_sub_T = ds_T[idx]

                        if same_act:
                            for act, act_id in act_dict.items():
                                idx = np.argwhere(np.all(ds_sub_T[:, :, 2] == act_id, axis=1)).squeeze()
                                ds_sub_act_X = ds_sub_X[idx]
                                ds_sub_act_T = ds_sub_T[idx]
                                groups.append((ds_sub_act_X, ds_sub_act_T))

                        else:
                            groups.append((ds_sub_X, ds_sub_T))

                else:
                    groups.append((ds_X, ds_T))

        else:
            groups.append((X_tmp, T_tmp))

        gp_batches = []
        for group in groups:
            gp_X, gp_T = group
            anchor = (gp_X[:, 0], gp_T)
            target = (gp_X[:, 1], gp_T)

            batched_anchor = Batcher(anchor, method, batch_size, transformer, aug_anchor, seed)
            batched_target = Batcher(target, method, batch_size, transformer, aug_target, seed)
            batched_datasets = [batched_anchor, batched_target]

            gp_batches.append(Zipper(batched_datasets))

        batches[tp] = Concatenator(gp_batches)

    return batches


def to_generator(C: Concatenator, input_type: tf.dtypes.DType,
                 input_shape: np.ndarray):
    def gen():
        for batch in C:
            anchor_batch, target_batch = batch
            yield anchor_batch, target_batch

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(input_type, input_type),
        output_shapes=(input_shape, input_shape)
    )


def prefetch(data: tf.data.Dataset) -> tf.data.Dataset:
    return data.cache().repeat().prefetch(tf.data.AUTOTUNE)

