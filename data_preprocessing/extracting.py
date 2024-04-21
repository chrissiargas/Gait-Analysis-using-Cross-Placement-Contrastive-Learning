import pandas as pd
from typing import List, Optional
import os

from config_parser import Parser

from formatting import resample, synchronize
from data_info import Info
from rwhar import rwhar_load_activity

import matplotlib.pyplot as plt
import numpy as np

class Extractor:
    def __init__(self):
        config = Parser()
        config.get_args()
        self.conf = config

        self.load_path = os.path.join(
            os.path.expanduser('~'),
            config.path,
            'gait-CLR'
        )

        self.info = Info()

    def build_pamap2(self, load: bool = False) -> pd.DataFrame:

        if load:
            path = os.path.join(self.load_path, 'pamap.csv')
            if os.path.exists(path):
                dataset = pd.read_csv(path)
                return dataset

        path = self.info.pamap2_path

        subjects = [*range(1, 10)]
        sub_files = []
        for subject in subjects:
            sub_path = os.path.join(
                path,
                'subject10' + str(subject) + '.dat'
            )
            sub_files.append(sub_path)

        colNames = ["timestamp", "activityID", "heartrate"]

        IMUhand = ['IMU_hand_temp', 'IMU_hand_ax1', 'IMU_hand_ay1', 'IMU_hand_az1',
                   'IMU_hand_ax2', 'IMU_hand_ay2', 'IMU_hand_az2',
                   'IMU_hand_rotx', 'IMU_hand_roty', 'IMU_hand_rotz',
                   'IMU_hand_magx', 'IMU_hand_magy', 'IMU_hand_magz',
                   'IMU_hand_oru', 'IMU_hand_orv', 'IMU_hand_orw', 'IMU_hand_orx']

        IMUchest = ['IMU_chest_temp', 'IMU_chest_ax1', 'IMU_chest_ay1', 'IMU_chest_az1',
                    'IMU_chest_ax2', 'IMU_chest_ay2', 'IMU_chest_az2',
                    'IMU_chest_rotx', 'IMU_chest_roty', 'IMU_chest_rotz',
                    'IMU_chest_magx', 'IMU_chest_magy', 'IMU_chest_magz',
                    'IMU_chest_oru', 'IMU_chest_orv', 'IMU_chest_orw', 'IMU_chest_orx']

        IMUankle = ['IMU_ankle_temp', 'IMU_ankle_ax1', 'IMU_ankle_ay1', 'IMU_ankle_az1',
                    'IMU_ankle_ax2', 'IMU_ankle_ay2', 'IMU_ankle_az2',
                    'IMU_ankle_rotx', 'IMU_ankle_roty', 'IMU_ankle_rotz',
                    'IMU_ankle_magx', 'IMU_ankle_magy', 'IMU_ankle_magz',
                    'IMU_ankle_oru', 'IMU_ankle_orv', 'IMU_ankle_orw', 'IMU_ankle_orx']

        columns = colNames + IMUhand + IMUchest + IMUankle

        data = pd.DataFrame()
        for file in sub_files:
            sub_data = pd.read_table(file, header=None, sep='\s+')
            sub_data.columns = columns
            sub_data['subject'] = int(file[-5])
            data = data.append(sub_data, ignore_index=True)

        data.reset_index(drop=True, inplace=True)

        data = data.drop(
            ['heartrate',
             'IMU_hand_temp', 'IMU_chest_temp', 'IMU_ankle_temp',
             'IMU_hand_rotx', 'IMU_hand_roty', 'IMU_hand_rotz',
             'IMU_hand_magx', 'IMU_hand_magy', 'IMU_hand_magz',
             'IMU_hand_oru', 'IMU_hand_orv', 'IMU_hand_orw', 'IMU_hand_orx',
             'IMU_chest_rotx', 'IMU_chest_roty', 'IMU_chest_rotz',
             'IMU_chest_magx', 'IMU_chest_magy', 'IMU_chest_magz',
             'IMU_chest_oru', 'IMU_chest_orv', 'IMU_chest_orw', 'IMU_chest_orx',
             'IMU_ankle_rotx', 'IMU_ankle_roty', 'IMU_ankle_rotz',
             'IMU_ankle_magx', 'IMU_ankle_magy', 'IMU_ankle_magz',
             'IMU_ankle_oru', 'IMU_ankle_orv', 'IMU_ankle_orw', 'IMU_ankle_orx'], axis=1)

        data['activity'] = data['activityID'].apply(self.info.pamap2_activities.get)
        data = data.loc[data['activity'].isin(['walking', 'running'])]
        data = data.drop(['activityID'], axis=1)
        data['timestamp'] = data['timestamp'] * 1000.  # convert to ms
        data.columns = data.columns.str.replace('IMU_', '')
        data.columns = data.columns.str.replace('ax1', 'acc_x')
        data.columns = data.columns.str.replace('ay1', 'acc_y')
        data.columns = data.columns.str.replace('az1', 'acc_z')

        dataset = pd.DataFrame()
        for position in self.info.pamap2_positions.keys():
            pos_ds = data[['timestamp', position + '_acc_x', position + '_acc_y', position + '_acc_z', 'activity',
                           'subject']].copy()
            pos_ds.columns = pos_ds.columns.str.replace(position + '_', '')
            pos_ds['position'] = position
            dataset = dataset.append(pos_ds)

        dataset = dataset.astype({'timestamp': int, 'subject': int,
                                  'activity': str, 'position': str,
                                  'acc_x': float, 'acc_y': float, 'acc_z': float})

        filepath = os.path.join(self.load_path, 'pamap.csv')
        dataset.to_csv(filepath, index=False, header=True)

        return dataset

    def build_rwhar(self, load: bool = False) -> pd.DataFrame:

        if load:
            path = os.path.join(self.load_path, 'rwhar.csv')
            if os.path.exists(path):
                dataset = pd.read_csv(path)
                return dataset

        path = self.info.rwhar_path

        subject_dir = os.listdir(path)
        dataset = pd.DataFrame()

        for sub in subject_dir:
            if "proband" not in sub:
                continue

            subject_num = int(sub[7:])  # proband is 7 letters long so subject num is number following that
            sub_pd = pd.DataFrame()

            for activity in self.info.rwhar_activities.keys():  # pair the acc and gyr zips of the same activity
                activity_name = "_" + activity + "_csv.zip"
                path_acc = os.path.join(path, sub, 'data', 'acc' + activity_name)
                table = rwhar_load_activity(path_acc)

                # add an activity column and fill it with activity num
                table["activity"] = self.info.rwhar_activities[activity]
                sub_pd = sub_pd.append(table)

            sub_pd["subject"] = subject_num  # add subject id to all entries
            dataset = dataset.append(sub_pd)

        dataset = dataset.astype({'timestamp': int, 'subject': int,
                                  'activity': str, 'position': str,
                                  'acc_x': float, 'acc_y': float, 'acc_z': float})

        filename = os.path.join(self.load_path, 'rwhar.csv')
        dataset.to_csv(filename, index=False, header=True)

        return dataset

    def build_mhealth(self, load: bool = False) -> pd.DataFrame:

        if load:
            path = os.path.join(self.load_path, 'mhealth.csv')
            if os.path.exists(path):
                dataset = pd.read_csv(path)
                return dataset

        path = self.info.mhealth_path

        subject_dir = os.listdir(path)
        dataset = pd.DataFrame()

        columns = {
            0: 'acc_chest_x',
            1: 'acc_chest_y',
            2: 'acc_chest_z',
            5: 'acc_ankle_x',
            6: 'acc_ankle_y',
            7: 'acc_ankle_z',
            14: 'acc_arm_x',
            15: 'acc_arm_y',
            16: 'acc_arm_z',
            23: 'activity'
        }

        for sub_file in subject_dir:
            if "subject" not in sub_file:
                continue

            sub_id = int(sub_file[15:-4])
            sub_path = os.path.join(path, sub_file)
            df = pd.read_csv(sub_path, header=None, sep='\t')

            df = df.loc[:, [0, 1, 2, 5, 6, 7, 14, 15, 16, 23]].rename(columns=columns)

            sub_df = pd.DataFrame()
            for position in self.info.mhealth_positions:
                acc_x = 'acc_' + position + '_x'
                acc_y = 'acc_' + position + '_y'
                acc_z = 'acc_' + position + '_z'

                pos_df = df[[acc_x, acc_y, acc_z, 'activity']]
                pos_df = pos_df.reset_index()
                pos_df['timestamp'] = df.index * (1000. / 50.)
                pos_df = pos_df.drop(['index'], axis=1)
                pos_df.columns = pos_df.columns.str.replace(position + '_', '')
                pos_df['position'] = position

                sub_df = sub_df.append(pos_df)

            sub_df['subject'] = str(sub_id)
            dataset = dataset.append(sub_df)

        dataset['activity'] = dataset['activity'].map(self.info.mhealth_activities)
        dataset = dataset.astype({'timestamp': int, 'subject': int,
                                  'activity': str, 'position': str,
                                  'acc_x': float, 'acc_y': float, 'acc_z': float})

        filename = os.path.join(self.load_path, 'mhealth.csv')
        dataset.to_csv(filename, index=False, header=True)

        return dataset

    def convert_activity(self, x):

        def row_split(row, place):
            if row[place + '_walknrun'] == 1 and row[place + '_walk'] == 0:
                return 1
            return 0

        places = ['treadmill', 'indoor', 'outdoor']
        for place in places:
            if place + '_walknrun' in x.columns:
                x[place + '_run'] = x.apply(lambda row: row_split(row, place), axis=1)

        walknrun_cols = [col for col in x if col.endswith('walknrun')]
        x = x.drop(walknrun_cols, axis=1)

        activities = x[x.columns.intersection(self.info.marea_activities)]
        x['activity'] = activities.idxmax(axis=1)
        x.loc[~activities.any(axis='columns'), 'activity'] = 'undefined'

        x = x.drop(x.columns.intersection(self.info.marea_activities), axis=1)

        return x

    def marea_load_subject(self, path):
        df = pd.read_csv(path)

        initial_activities = [
            "treadmill_walk", "treadmill_walknrun", "treadmill_slope_walk",
            "indoor_walk", "indoor_walknrun", "outdoor_walk", "outdoor_walknrun"
        ]

        sub_df = pd.DataFrame()
        for position in self.info.marea_positions:
            acc_x = 'accX_' + position
            acc_y = 'accY_' + position
            acc_z = 'accZ_' + position
            columns = [acc_x, acc_y, acc_z, *initial_activities]

            pos_df = df[df.columns.intersection(columns)].copy()

            pos_df = self.convert_activity(pos_df)

            pos_df = pos_df.reset_index()
            pos_df['timestamp'] = df.index * (1000. / 128.)
            pos_df = pos_df.drop(['index'], axis=1)
            pos_df.columns = pos_df.columns.str.replace('_' + position, '')
            pos_df['position'] = position

            sub_df = sub_df.append(pos_df)

        return sub_df

    def build_marea(self, load: bool = False) -> pd.DataFrame:

        if load:
            path = os.path.join(self.load_path, 'marea.csv')
            if os.path.exists(path):
                dataset = pd.read_csv(path)
                return dataset

        path = self.info.marea_path

        subject_dir = os.listdir(path)
        dataset = pd.DataFrame()

        for sub_file in subject_dir:
            if 'All' in sub_file:
                continue

            sub_id = int(sub_file[4:-4])
            sub_path = os.path.join(path, sub_file)

            sub_df = self.marea_load_subject(sub_path)

            sub_df['subject'] = sub_id
            dataset = dataset.append(sub_df)

        dataset = dataset.rename(columns={"accX": "acc_x", "accY": "acc_y", "accZ": "acc_z"})
        dataset = dataset.astype({'timestamp': int, 'subject': int,
                                  'activity': str, 'position': str,
                                  'acc_x': float, 'acc_y': float, 'acc_z': float})

        filename = os.path.join(self.load_path, 'marea.csv')
        dataset.to_csv(filename)

        return dataset

    def build_realdisp(self, load: bool = False) -> pd.DataFrame:

        if load:
            path = os.path.join(self.load_path, 'realdisp.csv')
            if os.path.exists(path):
                dataset = pd.read_csv(path)
                return dataset

        path = self.info.realdisp_path

        subject_dir = os.listdir(path)
        dataset = pd.DataFrame()

        for sub_file in subject_dir:
            if 'ideal' not in sub_file:
                continue

            sub_id = int(sub_file[7:-10])
            sub_path = os.path.join(path, sub_file)
            df = pd.read_csv(sub_path, header=None, sep='\t')

            sub_df = pd.DataFrame()
            for position, pos_id in self.info.realdisp_positions.items():
                offset = 2 + (pos_id - 1) * 13
                sec, microsecond = 0, 1
                acc_x, acc_y, acc_z = offset, offset + 1, offset + 2
                activity = 119

                columns = {
                    0: 's',
                    1: 'us',
                    offset: 'acc_x',
                    offset + 1: 'acc_y',
                    offset + 2: 'acc_z',
                    119: 'activity'
                }

                pos_df = df.loc[:, [sec, microsecond, acc_x, acc_y, acc_z, activity]].rename(columns=columns)
                pos_df['timestamp'] = pos_df['s'] * 1000. + pos_df['us'] / 1000.
                pos_df = pos_df.drop(['s', 'us'], axis=1)
                pos_df['position'] = position

                sub_df = sub_df.append(pos_df)

            sub_df['subject'] = sub_id
            dataset = dataset.append(sub_df)

        dataset['activity'] = dataset['activity'].map(self.info.realdisp_activities)
        dataset = dataset.astype({'timestamp': int, 'subject': int,
                                  'activity': str, 'position': str,
                                  'acc_x': float, 'acc_y': float, 'acc_z': float})

        filename = os.path.join(self.load_path, 'realdisp.csv')
        dataset.to_csv(filename)

        return dataset

    def prepare(self,
                name: str,
                ds: pd.DataFrame,
                positions: Optional[List] = None,
                activities: Optional[List] = None,
                fs: Optional[int] = None,
                sync: bool = False) -> pd.DataFrame:

        act_pairs = {}
        pos_pairs = {}
        old_fs = 0.
        if name == 'pamap2':
            act_pairs = self.info.pamap2_act_pairs
            pos_pairs = self.info.pamap2_pos_pairs
            old_fs = self.info.pamap2_fs

        elif name == 'rwhar':
            act_pairs = self.info.rwhar_act_pairs
            pos_pairs = self.info.rwhar_pos_pairs
            old_fs = self.info.rwhar_fs

        elif name == 'mhealth':
            act_pairs = self.info.mhealth_act_pairs
            pos_pairs = self.info.mhealth_pos_pairs
            old_fs = self.info.mhealth_fs

        elif name == 'marea':
            act_pairs = self.info.marea_act_pairs
            pos_pairs = self.info.marea_pos_pairs
            old_fs = self.info.marea_fs

        elif name == 'realdisp':
            act_pairs = self.info.realdisp_act_pairs
            pos_pairs = self.info.realdisp_pos_pairs
            old_fs = self.info.realdisp_fs

        ds['activity'] = ds['activity'].map(act_pairs)
        ds['position'] = ds['position'].map(pos_pairs)

        if activities is not None:
            ds = ds[ds['activity'].str.contains('|'.join(activities))]

        if positions is not None:
            ds = ds[ds['position'].str.contains('|'.join(positions))]

        if fs is not None:
            ds = resample(ds, old_fs, fs)

        if sync:
            ds = synchronize(ds, pivots=['left_lower_arm', 'right_lower_arm', 'dominant_lower_arm'])

        return ds

    def __call__(self,
                 datasets: Optional[List[str]] = None,
                 positions: Optional[List[str]] = None,
                 activities: Optional[List[str]] = None,
                 fs: Optional[int] = None,
                 sync: bool = True,
                 load: bool = False) -> pd.DataFrame:

        if datasets is None:
            datasets = self.conf.datasets
        if positions is None:
            positions = self.conf.positions
        if activities is None:
            activities = self.conf.activities
        if fs is None:
            fs = self.conf.fs

        merged_ds = pd.DataFrame()
        for dataset in datasets:
            print('preparing ' + dataset + '...')

            if dataset == 'pamap2':
                ds = self.build_pamap2(load)
            elif dataset == 'rwhar':
                ds = self.build_rwhar(load)
            elif dataset == 'mhealth':
                ds = self.build_mhealth(load)
            elif dataset == 'marea':
                ds = self.build_marea(load)
            elif dataset == 'realdisp':
                ds = self.build_realdisp(load)
            else:
                ds = pd.DataFrame()

            # old_ds = ds.copy()
            ds = self.prepare(dataset, ds, positions, activities, fs, sync)
            ds['dataset'] = dataset
            ds = ds.sort_values(by=['subject', 'timestamp'])

            filename = os.path.join('data_preprocessing', 'datasets', dataset + '.csv')
            ds.to_csv(filename)

            merged_ds = pd.concat([merged_ds, ds], axis=0, ignore_index=True)

            # old_ds = old_ds.sort_values(by=['subject', 'position', 'timestamp'])
            # A = old_ds.loc[(old_ds['subject'] == 3) & (old_ds['position'] == 'LF')]
            # B = ds.loc[ds['subject'] == 3]
            #
            # acc_cols = ['acc_x', 'acc_y', 'acc_z']
            # A = A[acc_cols].values
            #
            # acc_cols = ['left_lower_leg_acc_x', 'left_lower_leg_acc_y', 'left_lower_leg_acc_z']
            # B = B[acc_cols].values
            #
            # start = 150
            # old_fs = self.info.marea_fs
            # old_segment = A[start*old_fs:start*old_fs+old_fs]
            # plt.plot(old_segment)
            # plt.show()
            #
            # new_fs = self.conf.fs
            # new_segment = B[start * new_fs:start * new_fs + new_fs]
            # plt.plot(new_segment)
            # plt.show()
            #
            # plt.plot(np.linspace(0, old_fs//2-1, old_fs//2), np.abs(np.fft.fft(old_segment[:,0]))[:old_fs//2])
            # plt.plot(np.linspace(0, new_fs//2-1, new_fs//2), np.abs(np.fft.fft(new_segment[:,0]))[:new_fs//2] * old_fs / new_fs, 'r--')
            # plt.show()

        filename = os.path.join(self.load_path, 'gaitCLR.csv')
        merged_ds.to_csv(filename)

        return merged_ds


if __name__ == '__main__':

    Bob = Extractor()
    # pamap_ds = PP.build_pamap2()
    # print(pamap_ds)
    # rwhar_ds = PP.build_rwhar()
    # print(rwhar_ds)
    # mhealth_ds = PP.build_mhealth()
    # print(mhealth_ds)
    # marea_ds = PP.build_marea()
    # print(marea_ds)
    # realdisp_ds = PP.build_realdisp()
    # print(realdisp_ds)

    Bob(sync=True, load=False)
