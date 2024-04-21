import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
import random


def split_train_test(x: pd.DataFrame, split_type: str, hold_out: Union[List, int, float, str], seed: Optional[int] = None)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = x.copy()

    ds_names = x['dataset'].unique()

    subs = {}
    n_subs = 0
    for ds_name in ds_names:
        ds_subs = x[x['dataset'] == ds_name]['subject'].unique().tolist()
        subs[ds_name] = ds_subs
        n_subs += len(ds_subs)

    if split_type == 'lodo':
        if isinstance(hold_out, str):
            test = x[x['dataset'] == hold_out]
            train = x[x['dataset'] != hold_out]

    elif split_type == 'loso':
        rng = np.random.default_rng(seed=seed)

        test_ids = []
        train_ids = []
        for ds_name in ds_names:
            ds_subs = subs[ds_name]

            if isinstance(hold_out, float):
                r = int(len(ds_subs) * hold_out)
                test_subs = rng.choice(ds_subs, r, replace=False)
            elif isinstance(hold_out, int):
                r = hold_out
                test_subs = rng.choice(ds_subs, r, replace=False)
            elif isinstance(hold_out, list):
                test_subs = hold_out

            train_subs = list(set(ds_subs) - set(test_subs))

            print('train subjects: ', train_subs)
            print('test subjects: ', test_subs)

            test_ids.extend([ds_name + '_' + str(test_sub) for test_sub in test_subs])
            train_ids.extend([ds_name + '_' + str(train_sub) for train_sub in train_subs])

        x['ds_sub'] = x['dataset'] + '_' + x['subject'].astype(str)
        train = x[x['ds_sub'].isin(train_ids)]
        test = x[x['ds_sub'].isin(test_ids)]

    elif split_type == 'start' or split_type == 'end':
        train, test = pd.DataFrame(), pd.DataFrame()
        for name in ds_names:
            ds_subs = subs[name]
            for sub in ds_subs:
                DS = x[(x['dataset'] == ds_name) & (x['subject'] == sub)]
                ds_sub_acts = DS['activity'].unique()
                for act in ds_sub_acts:
                    ds = DS[DS['activity'] == act]

                    if isinstance(hold_out, float):
                        test_size = int(len(ds) * hold_out)
                        train_size = len(ds) - test_size

                    if split_type == 'start':
                        this_test, this_train = ds.iloc[:test_size], ds.iloc[test_size:]
                    elif split_type == 'end':
                        this_train, this_test = ds.iloc[:train_size], ds.iloc[train_size:]

                    train = pd.concat([train, this_train], axis=0)
                    test = pd.concat([test, this_test], axis=0)

    else:
        train = x
        test = pd.DataFrame()

    return train, test

