import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, List


def resample(ds: pd.DataFrame, old_fs: int, new_fs: int, thres: float = 1.) -> pd.DataFrame:
    resampled_ds = pd.DataFrame()
    acc_cols = ['acc_x', 'acc_y', 'acc_z']

    for sub_id, sub_df in ds.groupby('subject'):
        for act_id, act_df in sub_df.groupby('activity'):
            for pos_id, pos_df in act_df.groupby('position'):
                old_t = pos_df['timestamp'].values
                step = 1000. / new_fs
                e = 1e-4
                new_t = np.arange(start=old_t[0], stop=old_t[-1] + e, step=step)

                old_acc = pos_df[acc_cols].interpolate().values
                f = interp1d(old_t, old_acc, kind='linear', axis=0, fill_value='extrapolate')
                new_acc = f(new_t)
                resampled_df = pd.DataFrame(new_acc, columns=acc_cols)

                NaNs = pos_df.isna().any(axis=1).values.astype(int)
                f = interp1d(old_t, NaNs, kind='previous', axis=0, fill_value='extrapolate')
                prev_NaNs = f(new_t)
                f = interp1d(old_t, NaNs, kind='next', axis=0, fill_value='extrapolate')
                next_NaNs = f(new_t)
                resampled_df.loc[(prev_NaNs == 1) | (next_NaNs == 1)] = np.nan

                f = interp1d(old_t, old_t, kind='nearest', axis=0, fill_value='extrapolate')
                nearest_t = f(new_t)
                resampled_df.loc[abs(new_t - nearest_t) > thres * 1000. / old_fs] = np.nan

                resampled_df['timestamp'] = new_t
                resampled_df['position'] = pos_id
                resampled_df['activity'] = act_id
                resampled_df['subject'] = sub_id

                resampled_ds = pd.concat((resampled_ds, resampled_df),
                                         axis=0, ignore_index=True)

    return resampled_ds


def synchronize(ds: pd.DataFrame, pivots: Optional[List[str]], kind: str = 'linear') -> pd.DataFrame:
    for tmp_pivot in pivots:
        if tmp_pivot in ds.position.unique():
            pivot = tmp_pivot

    pivot_df = ds[ds['position'] == pivot]
    npivot_df = ds[ds['position'] != pivot]
    other_dfs = npivot_df.groupby('position')

    synced_ds = pd.DataFrame()
    pivot_cols = {'acc_x': pivot + '_acc_x',
                  'acc_y': pivot + '_acc_y',
                  'acc_z': pivot + '_acc_z'}

    for subject in pivot_df.subject.unique():
        pivot_sub = pivot_df[pivot_df['subject'] == subject]
        pivot_t = pivot_sub['timestamp']
        synced_df = pivot_sub.rename(columns=pivot_cols).copy().reset_index(drop=True)
        synced_df = synced_df.drop('position', axis=1)

        for pos, other_df in other_dfs:
            other_sub = other_df[other_df['subject'] == subject]
            other_acc = other_sub[pivot_cols.keys()].interpolate().values
            other_t = other_sub['timestamp']
            other_cols = [pos + '_acc_x', pos + '_acc_y', pos + '_acc_z']

            f = interp1d(other_t, other_acc, kind=kind, axis=0, fill_value='extrapolate')
            synced_acc = pd.DataFrame(f(pivot_t), columns=other_cols)

            NaNs = other_sub.isna().any(axis=1).values.astype(int)
            f = interp1d(other_t, NaNs, kind='previous', axis=0, fill_value='extrapolate')
            prev_NaNs = f(pivot_t)
            f = interp1d(other_t, NaNs, kind='next', axis=0, fill_value='extrapolate')
            next_NaNs = f(pivot_t)
            synced_acc.loc[(prev_NaNs == 1) | (next_NaNs == 1)] = np.nan

            synced_df = pd.concat([synced_df, synced_acc], axis=1)

        synced_ds = pd.concat((synced_ds, synced_df),
                              axis=0, ignore_index=True)

    return synced_ds
