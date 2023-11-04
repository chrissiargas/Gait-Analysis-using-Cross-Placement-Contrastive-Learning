import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
import random


def split_train_test(x: pd.DataFrame, split_type: str, hold_out: Union[str, int, float], seed: Optional[int] = None)\
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
        assert isinstance(hold_out, str)
        test = x[x['dataset'] == hold_out]
        train = x[x['dataset'] != hold_out]

    elif split_type == 'loso':
        assert isinstance(hold_out, float)
        rng = np.random.default_rng(seed=seed)

        test_ids = []
        train_ids = []
        for ds_name in ds_names:
            ds_subs = subs[ds_name]
            r = int(len(ds_subs) * hold_out)

            test_subs = rng.choice(ds_subs, r, replace=False)
            train_subs = list(set(ds_subs) - set(test_subs))

            test_ids.extend([ds_name + '_' + str(test_sub) for test_sub in test_subs])
            train_ids.extend([ds_name + '_' + str(train_sub) for train_sub in train_subs])

        x['ds_sub'] = x['dataset'] + '_' + x['subject'].astype(str)
        train = x[x['ds_sub'].isin(train_ids)]
        test = x[x['ds_sub'].isin(test_ids)]

    else:
        train = x
        test = pd.DataFrame()

    return train, test

