import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from features import add_norm_xy, add_norm_xz, add_norm_xyz, add_norm_yz, add_jerk, add_grav
from filters import median_smoothing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_info import Info
import math


def fill_nan(x: pd.DataFrame, position: str, how: str) -> pd.DataFrame:
    x = x.copy()

    features = x.columns[x.columns.str.contains(position)]
    x[position + '_is_NaN'] = x[features].isnull().any(axis='columns')
    groups = x.groupby(['dataset', 'subject', 'activity'])

    for feature in features:
        if how == 'linear':
            cleaned_ft = groups.apply(lambda g: g[feature].interpolate(method='linear')).droplevel([0,1,2])
        elif how == 'spline':
            cleaned_ft = groups.apply(lambda g: g[feature].interpolate(method='spline')).droplevel([0,1,2])
        elif how == 'bfill':
            cleaned_ft = groups.apply(lambda g: g[feature].fillna(method='bfill')).droplevel([0,1,2])
        elif how == 'ffill':
            cleaned_ft = groups.apply(lambda g: g[feature].fillna(method='ffill')).droplevel([0,1,2])

        x[feature] = cleaned_ft

    return x


def drop_positions(x: pd.DataFrame, positions) -> pd.DataFrame:
    x = x.copy()

    all_acc = x.columns.str.contains("acc")

    pass_acc = []
    for position in positions:
        pass_acc.extend(x.columns.str.endswith(position + "_acc_x"))
        pass_acc.extend(x.columns.str.endswith(position + "_acc_y"))
        pass_acc.extend(x.columns.str.endswith(position + "_acc_z"))

    drop_acc = list(set(all_acc) - set(pass_acc))

    x = x.drop(drop_acc, axis=1)

    return x


def choose(x: pd.DataFrame, ds_names: List[str], positions: List[str], activities: List[str]) -> pd.DataFrame:
    x = x.copy()

    in_pos = []
    if positions is not None:
        for position in positions:
            parts = position.split('_')
            if len(parts) == 3:
                in_pos.append(position)

            if len(parts) == 2:
                if parts[0] == 'upper' or parts[0] == 'lower':
                    add = ['right_' + position, 'left_' + position, 'dominant_' + position]
                    in_pos.extend(add)
                elif parts[0] == 'right' or parts[0] == 'left':
                    add = [parts[0] + '_upper_' + parts[1], parts[0] + '_lower_' + parts[1]]
                    in_pos.extend(add)

            if len(parts) == 1:
                if position == 'leg' or position == 'arm':
                    add = ['right_lower' + position, 'right_upper' + position,
                           'left_lower' + position, 'left_upper' + position,
                           'dominant_lower' + position, 'dominant_upper' + position]
                    in_pos.extend(add)

    info = Info()

    all_pos = []
    drop_names = []
    for ds_name in ds_names:
        common_pos = list(set(in_pos) & set(info.ds_positions[ds_name]))
        if len(common_pos) < 2:
            drop_names.append(ds_name)
        all_pos.extend(info.ds_positions[ds_name])

    ds_names = [ds_name for ds_name in ds_names if ds_name not in drop_names]
    in_pos = list(set(in_pos) & set(all_pos))

    x = x[x['dataset'].isin(ds_names)]

    acc_cols = x.columns[x.columns.str.contains('acc')]
    in_cols = [acc_col for acc_col in acc_cols if any(pos for pos in in_pos if pos in acc_col)]
    out_cols = list(set(acc_cols) - set(in_cols))
    x = x.drop(out_cols, axis=1)
    x = x[x.activity.str.contains('|'.join(activities))]

    return x


def impute(x: pd.DataFrame, how: str) -> pd.DataFrame:
    if how is None:
        return x

    x = x.copy()

    all_acc = x.columns[x.columns.str.contains("acc")]
    positions = [acc_name[:-6] for acc_name in all_acc]
    positions = list(set(positions))

    for position in positions:
        x = fill_nan(x, position, how)

    return x


def virtual(x: pd.DataFrame, features: List[str], fs: int) -> pd.DataFrame:
    x = x.copy()

    all_acc = x.columns[x.columns.str.contains("acc")]
    positions = [acc_name[:-6] for acc_name in all_acc]
    positions = list(set(positions))

    if features is None:
        return x

    for position in positions:

        if 'norm_xyz' in features:
            x = add_norm_xyz(x, position)

        if 'norm_xy' in features:
            x = add_norm_xy(x, position)

        if 'norm_yz' in features:
            x = add_norm_yz(x, position)

        if 'norm_xz' in features:
            x = add_norm_xz(x, position)

        if 'jerk' in features:
            x = add_jerk(x, position, fillna=True)

        if 'grav_x' in features:
            x = add_grav(x, position, fs, 'x')

        if 'grav_y' in features:
            x = add_grav(x, position, fs, 'y')

        if 'grav_z' in features:
            x = add_grav(x, position, fs, 'z')

    return x


def smooth(x, filter_type, w):
    if filter_type is None:
        return x

    x = x.copy()

    all_acc = x.columns[x.columns.str.contains("acc")]
    positions = [acc_name[:-6] for acc_name in all_acc]
    positions = list(set(positions))

    for position in positions:

        if filter_type == 'median':
            x = median_smoothing(x, position, w)

    return x


def rescale(x: pd.DataFrame, how: str = 'standard') -> pd.DataFrame:
    if how is None:
        return x

    x = x.copy()

    all_acc = x.columns[x.columns.str.contains("acc")]
    positions = [acc_name[:-6] for acc_name in all_acc]
    positions = list(set(positions))

    if how == 'min-max':
        rescaler = MinMaxScaler()
    elif how == 'standard':
        rescaler = StandardScaler()

    for position in positions:
        features = x.columns[x.columns.str.contains(position)]
        features = list(filter(lambda f: 'NaN' not in f, features))
        x[features] = rescaler.fit_transform(x[features].values)

    return x


def to_categorical(x: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    x = x.copy()

    ds_factor = pd.factorize(x['dataset'])
    act_factor = pd.factorize(x['activity'])

    x['dataset_id'], ds_dict = ds_factor[0], ds_factor[1].values
    ds_dict = {k: v for v, k in enumerate(ds_dict)}
    x['subject_id'] = x['subject'].astype(int)
    x['activity_id'], act_dict = act_factor[0], act_factor[1].values
    act_dict = {k: v for v, k in enumerate(act_dict)}
    dicts = [ds_dict, act_dict]

    return x, dicts


def segment(X: pd.DataFrame, length: int, step: int) -> np.ndarray:
    X = X.values

    n_windows = math.ceil((X.shape[0] - length + 1) / step)
    n_windows = max(0, n_windows)

    X = np.lib.stride_tricks.as_strided(
        X,
        shape=(n_windows, length, X.shape[1]),
        strides=(step * X.strides[0], X.strides[0], X.strides[1]))

    return X


def stack(x: pd.DataFrame, length: int, step: int, anchors: List[str], targets: List[str]) -> Tuple[dict, dict]:
    x = x.copy()

    t_cols = ['dataset_id', 'subject_id', 'activity_id', 'timestamp']

    X = {}
    T = {}

    for it, (anchor, target) in enumerate(zip(anchors, targets)):
        anchor_cols = x.columns[x.columns.str.contains(anchor)]
        target_cols = x.columns[x.columns.str.contains(target)]

        if len(anchor_cols) == 0 or len(target_cols) == 0:
            return X, T

        groups = x.groupby(['dataset', 'subject', 'activity'])

        anchor_ws = groups.apply(lambda gp: segment(gp[anchor_cols], length, step))
        anchor_ws = np.concatenate(anchor_ws.values)
        target_ws = groups.apply(lambda gp: segment(gp[target_cols], length, step))
        target_ws = np.concatenate(target_ws.values)

        t_ws = groups.apply(lambda gp: segment(gp[t_cols], length, step))
        t_ws = np.concatenate(t_ws.values)

        anchor_ws = anchor_ws[:, np.newaxis, ...]
        target_ws = target_ws[:, np.newaxis, ...]
        x_ws = np.concatenate((anchor_ws, target_ws), axis=1)

        X[target] = x_ws
        T[target] = t_ws

    return X, T


def finalize(x: pd.DataFrame, length: int, step: int, pairs: List[List[str]], oversample: bool)\
        -> Tuple[Tuple[dict, dict],List]:
    x = x.copy()
    nan_cols = x.columns[x.columns.str.contains('NaN')].tolist()
    moved_cols = [col for col in x.columns if col not in nan_cols] + nan_cols
    x = x[moved_cols]

    ds_names = x['dataset'].unique()
    info = Info()

    X = {}
    T = {}

    x, dicts = to_categorical(x)

    for ds_name in ds_names:
        anchors, targets = [], []
        ds = x[x['dataset'] == ds_name]
        ds_pos = info.ds_positions[ds_name]

        for pair in pairs:
            if pair[0] in ds_pos and pair[1] in ds_pos:
                anchors.append(pair[0])
                targets.append(pair[1])
                if not oversample:
                    break

        ds_X, ds_T = stack(ds, length, step, anchors, targets)

        for target in targets:
            if target not in X:
                X[target] = ds_X[target]
                T[target] = ds_T[target]
            else:
                X[target] = np.concatenate((X[target], ds_X[target]))
                T[target] = np.concatenate((T[target], ds_T[target]))

    return (X, T), dicts


def clean(S: Tuple[dict, dict], how: str, tol: float) -> Tuple[dict, dict]:
    X, T = S

    for target in X.keys():

        X_tmp = X[target].copy()
        T_tmp = T[target].copy()

        length = X_tmp.shape[2]
        thres = int(tol * length)

        anchor_NaNs = X_tmp[:, 0, :, -1]
        target_NaNs = X_tmp[:, 1, :, -1]

        anchor_count = np.sum(anchor_NaNs, axis=1)
        target_count = np.sum(target_NaNs, axis=1)

        if how == 'in total':
            anchor_drop = np.argwhere(anchor_count > thres).squeeze()
            target_drop = np.argwhere(target_count > thres).squeeze()
            drop = np.union1d(anchor_drop, target_drop)
            X_tmp = np.delete(X_tmp, drop, axis=0)
            T[target] = np.delete(T_tmp, drop, axis=0)
            X[target] = X_tmp[..., :-1]

    return X, T
