import numpy as np
from math import ceil, floor
from typing import Optional, Tuple, List
from transformers import Temporal
import tensorflow as tf


class Batcher:
    def __init__(self, S: Tuple[np.ndarray, np.ndarray], method: str, batch_size: int,
                 transformer: Optional[Temporal] = None, training: bool = False,
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
        self.training = training

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
                                               training=self.training)
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
        self.batch_size = self.datasets[0].batch_size
        self.length = self.datasets[0].length
        self.channels = self.datasets[0].channels

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
        self.N_batches = 0
        for zipped_dataset in zipped_datasets:
            self.N_batches += zipped_dataset.n_batches
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


def batch_concat(S: Tuple[np.ndarray, np.ndarray], method: str, batch_size: int,
                 dicts: List[dict], transformer: Optional[Temporal] = None,
                 same_ds: bool = True, same_sub: bool = True, same_act: bool = True,
                 training: bool = False, seed: Optional[int] = None) -> Concatenator:
    X, T = S
    ds_dict, act_dict = dicts
    X = X.copy()
    T = T.copy()

    groups = []
    if same_ds:
        for ds, ds_id in ds_dict.items():
            idx = np.argwhere(np.all(T[:, :, 1] == ds_id, axis=1)).squeeze()
            ds_X = X[idx]
            ds_T = T[idx]

            if same_sub:
                sub_arr = ds_T[:, :, 2]
                sub_ids = np.unique(sub_arr)

                for sub_id in sub_ids:
                    idx = np.argwhere(np.all(ds_T[:, :, 2] == sub_id, axis=1)).squeeze()
                    ds_sub_X = ds_X[idx]
                    ds_sub_T = ds_T[idx]

                    if same_act:
                        for act, act_id in act_dict.items():
                            idx = np.argwhere(np.all(ds_sub_T[:, :, 3] == act_id, axis=1)).squeeze()
                            ds_sub_act_X = ds_sub_X[idx]
                            ds_sub_act_T = ds_sub_T[idx]
                            groups.append((ds_sub_act_X, ds_sub_act_T))

                    else:
                        groups.append((ds_sub_X, ds_sub_T))

            else:
                groups.append((ds_X, ds_T))

    else:
        groups.append((X, T))

    gp_batches = []
    for group in groups:
        X, T = group
        anchor = (X[:, 0], T)
        target = (X[:, 1], T)

        batched_anchor = Batcher(anchor, method, batch_size, transformer, training, seed)
        batched_target = Batcher(target, method, batch_size, transformer, training, seed)
        batched_datasets = [batched_anchor, batched_target]

        gp_batches.append(Zipper(batched_datasets))

    batches = Concatenator(gp_batches)

    return batches


def to_generator(C: Concatenator, input_type: tf.dtypes.DType,
                 input_shape: np.ndarray, training: bool = False):
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

