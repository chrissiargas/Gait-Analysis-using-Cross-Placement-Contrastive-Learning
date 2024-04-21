import matplotlib.pyplot as plt
import numpy as np
from config_parser import Parser
from operator import itemgetter
from transformations import add_noise, random_scale, time_mask, time_shift, random_rotate
from filters import vectorized_lowpass_filter
import tensorflow as tf


class Temporal:
    def __init__(self):
        config = Parser()
        config.get_args()
        self.conf = config

        self.initial_features = {'acc_x': 0, 'acc_y': 1, 'acc_z': 2}
        self.initial_features.update({k: v+3 for v, k in enumerate(self.conf.prod_features)})

        self.available_augs = ['jitter', 'scale', 'mask', 'shift', 'rotate']
        self.features = self.conf.use_features
        self.augs = self.conf.augmentations

        if self.conf.augmentations is None:
            self.augs = []
        else:
            assert all(aug in self.available_augs for aug in self.augs)

        self.batch_size = self.conf.batch_size
        self.w_len = self.conf.length
        self.n_features = len(self.conf.use_features)
        self.data_type = tf.float32

    def get_shape(self):
        return self.batch_size, self.w_len, self.n_features

    def get_type(self):
        return self.data_type

    def __call__(self, batch_of_ws: np.ndarray, augment: bool = False):
        batch_of_ws = np.array(batch_of_ws, dtype=np.float64)
        x = self.initial_features['acc_x']
        y = self.initial_features['acc_y']
        z = self.initial_features['acc_z']
        output = None

        if augment:
            for aug in self.augs:
                if aug == 'jitter':
                    batch_of_ws = add_noise(batch_of_ws)
                elif aug == 'scale':
                    batch_of_ws = random_scale(batch_of_ws)
                elif aug == 'mask':
                    batch_of_ws = time_mask(batch_of_ws)
                elif aug == 'shift':
                    batch_of_ws = time_shift(batch_of_ws, final_length=self.w_len)
                elif aug == 'rotate':
                    batch_of_ws[..., [x, y, z]] = random_rotate(batch_of_ws[..., [x, y, z]])

        elif self.conf.shift_length > 0:
            batch_of_ws = batch_of_ws[:, self.conf.shift_length // 2: -self.conf.shift_length // 2, :]

        for feature in self.conf.use_features:
            if feature in self.initial_features:
                f_idx = self.initial_features[feature]
                batch_of_signals = batch_of_ws[:, :, f_idx]

            else:
                if feature == 'norm_xyz':
                    batch_of_signals = np.sqrt(np.sum(batch_of_ws[..., [x, y, z]] ** 2, axis=2))

                elif feature == 'norm_xy':
                    batch_of_signals = np.sqrt(np.sum(batch_of_ws[..., [x, y]] ** 2, axis=2))

                elif feature == 'norm_xz':
                    batch_of_signals = np.sqrt(np.sum(batch_of_ws[..., [x, z]] ** 2, axis=2))

                elif feature == 'norm_yz':
                    batch_of_signals = np.sqrt(np.sum(batch_of_ws[..., [y, z]] ** 2, axis=2))

                elif feature == 'jerk':
                    Js = np.array([np.array([(window[1:, orientation] - window[:-1, orientation]) * self.conf.fs
                                             for orientation in [x, y, z]]).transpose() for window in batch_of_ws])
                    batch_of_signals = np.sqrt(np.sum(Js ** 2, axis=2))
                    batch_of_signals = np.concatenate((batch_of_signals, np.zeros((batch_of_signals.shape[0], 1))),
                                                      axis=1)

                elif feature == 'grav_x':
                    vlowspass = vectorized_lowpass_filter()
                    batch_of_signals = vlowspass(batch_of_ws[..., x], 1., self.conf.fs / 2)

                elif feature == 'grav_y':
                    vlowspass = vectorized_lowpass_filter()
                    batch_of_signals = vlowspass(batch_of_ws[..., y], 1., self.conf.fs / 2)

                elif feature == 'grav_z':
                    vlowspass = vectorized_lowpass_filter()
                    batch_of_signals = vlowspass(batch_of_ws[..., z], 1., self.conf.fs / 2)

                elif feature == 'grav_xyz':
                    norm_xyz = np.sqrt(np.sum(batch_of_ws[..., [x, y, z]] ** 2, axis=2))
                    vlowspass = vectorized_lowpass_filter()
                    batch_of_signals = vlowspass(norm_xyz, 1., self.conf.fs / 2)

                elif feature == 'grav':
                    vlowspass = vectorized_lowpass_filter()
                    grav_x = vlowspass(batch_of_ws[..., x], 1., self.conf.fs / 2)
                    grav_y = vlowspass(batch_of_ws[..., y], 1., self.conf.fs / 2)
                    grav_z = vlowspass(batch_of_ws[..., z], 1., self.conf.fs / 2)
                    batch_of_signals = np.sqrt(grav_x ** 2 + grav_y ** 2 + grav_z ** 2)

            if output is None:
                output = batch_of_signals[:, :, np.newaxis]

            else:
                output = np.concatenate(
                    (output, batch_of_signals[:, :, np.newaxis]),
                    axis=2
                )

        return output
