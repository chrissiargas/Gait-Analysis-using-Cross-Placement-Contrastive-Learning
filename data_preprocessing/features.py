import pandas as pd
import numpy as np
from filters import butter_lowpass_filter


def get_norm_xyz(x: pd.DataFrame, position: str) -> pd.DataFrame:
    x = x.copy()

    acc_x = position + '_acc_x'
    acc_y = position + '_acc_y'
    acc_z = position + '_acc_z'

    x[position + '_norm_xyz'] = np.sqrt(x[acc_x] ** 2 + x[acc_y] ** 2 + x[acc_z] ** 2)

    return x


def get_norm_xy(x: pd.DataFrame, position: str) -> pd.DataFrame:
    x = x.copy()

    acc_x = position + '_acc_x'
    acc_y = position + '_acc_y'

    x[position + '_norm_xy'] = np.sqrt(x[acc_x] ** 2 + x[acc_y] ** 2)

    return x


def get_norm_yz(x: pd.DataFrame, position: str) -> pd.DataFrame:
    x = x.copy()

    acc_y = position + '_acc_y'
    acc_z = position + '_acc_z'

    x[position + '_norm_yz'] = np.sqrt(x[acc_y] ** 2 + x[acc_z] ** 2)

    return x


def get_norm_xz(x: pd.DataFrame, position: str) -> pd.DataFrame:
    x = x.copy()

    acc_x = position + '_acc_x'
    acc_z = position + '_acc_z'

    x[position + '_norm_xz'] = np.sqrt(x[acc_x] ** 2 + x[acc_z] ** 2)

    return x


def get_jerk(x: pd.DataFrame, position: str, fillna: bool = False) -> pd.DataFrame:
    x = x.copy()

    acc_x = position + '_acc_x'
    acc_y = position + '_acc_y'
    acc_z = position + '_acc_z'

    groups = x.groupby(['dataset', 'subject', 'activity'])
    acc_dx = (groups[acc_x].diff() / groups['timestamp'].diff()).values[:, np.newaxis]
    acc_dy = (groups[acc_y].diff() / groups['timestamp'].diff()).values[:, np.newaxis]
    acc_dz = (groups[acc_z].diff() / groups['timestamp'].diff()).values[:, np.newaxis]

    acc_di = np.concatenate((acc_dx, acc_dy, acc_dz), axis=1)
    jerk = np.sqrt(np.sum(np.square(acc_di), axis=1))

    x[position + '_jerk'] = jerk
    groups = x.groupby(['dataset', 'subject', 'activity'])

    if fillna:
        mask = groups.cumcount() == 0
        x[position + '_jerk'] = x[position + '_jerk'].where(~mask, 0)

    return x


def get_grav(x: pd.DataFrame, position: str, fs: int) -> pd.DataFrame:
    x = x.copy()
    x = x.interpolate()

    cutoff = 1.

    acc_x = position + '_acc_x'
    acc_y = position + '_acc_y'
    acc_z = position + '_acc_z'

    groups = x.groupby(['dataset', 'subject', 'activity'])

    low_x = groups.apply(lambda g: butter_lowpass_filter(g[acc_x].values, cutoff, fs / 2))
    low_y = groups.apply(lambda g: butter_lowpass_filter(g[acc_y].values, cutoff, fs / 2))
    low_z = groups.apply(lambda g: butter_lowpass_filter(g[acc_z].values, cutoff, fs / 2))

    low_x = np.concatenate(low_x.values)
    low_y = np.concatenate(low_y.values)
    low_z = np.concatenate(low_z.values)

    x[position + '_grav_x'] = low_x
    x[position + '_grav_y'] = low_y
    x[position + '_grav_z'] = low_z

    return x
