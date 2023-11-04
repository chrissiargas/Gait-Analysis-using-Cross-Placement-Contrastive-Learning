import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config_parser import Parser
import os
from data_info import Info
from typing import Optional, List
from scipy import signal


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def pairwise_distance(this, other):
    norm = lambda x: np.sum(np.square(x), 1)

    return np.transpose(norm(np.expand_dims(this, 2) - np.transpose(other)))


def gaussian_rbf_kernel(this, other):
    sigmas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 100])

    beta = 1.0 / (2.0 * np.expand_dims(sigmas, 1))
    dist = pairwise_distance(this, other)

    s = np.matmul(beta, np.reshape(dist, (1, -1)))

    return np.reshape(np.sum(np.exp(-1.0 * s), 0), np.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_rbf_kernel):
    cost = (
            np.mean(kernel(x, x))
            + np.mean(kernel(y, y))
            - 2 * np.mean(kernel(x, y))
    )

    # We do not allow the loss to become negative.
    cost = np.where(cost > 0, cost, 0)
    return cost


class visualizer:
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

        filepath = os.path.join(self.load_path, 'gaitCLR.csv')
        self.data = pd.read_csv(filepath)

    def show_window(self, name: str, positions: List[str], activity: str, win_len: Optional[int] = None,
                    only_norm=False, only_low=False, all=False):
        lw = 1
        n_positions = len(positions)

        dataset = self.data[self.data['dataset'] == name]
        act_data = dataset[dataset['activity'] == activity]
        subs = act_data.subject.unique()

        acc_per_sub = {}
        time_per_sub = {}
        for subject in subs:
            sensordata = act_data[act_data['subject'] == subject]
            acc_per_pos = {}

            for position in positions:
                f = sensordata.interpolate()
                acc_x = f[position + "_acc_x"].to_numpy()[:, np.newaxis]
                acc_y = f[position + "_acc_y"].to_numpy()[:, np.newaxis]
                acc_z = f[position + "_acc_z"].to_numpy()[:, np.newaxis]
                acc = np.concatenate((acc_x, acc_y, acc_z), axis=1)

                norm = np.sqrt(np.sum(np.square(acc), axis=1))[:, np.newaxis]

                low_x = butter_lowpass_filter(acc_x[:, 0], 1., 50./2)[:, np.newaxis]
                low_y = butter_lowpass_filter(acc_y[:, 0], 1., 50. / 2)[:, np.newaxis]
                low_z = butter_lowpass_filter(acc_z[:, 0], 1., 50. / 2)[:, np.newaxis]
                low = np.concatenate((low_x, low_y, low_z), axis=1)

                low_norm = np.sqrt(np.sum(np.square(low), axis=1))[:, np.newaxis]

                acc = np.concatenate((acc, norm, low, low_norm), axis=1)
                acc_per_pos[position] = acc

            acc_per_sub[subject] = acc_per_pos
            time_per_sub[subject] = sensordata['timestamp'].to_numpy() / 1000.
            print((time_per_sub[subject][-1] - time_per_sub[subject][0]) / 60.)

        dists = np.zeros((n_positions, n_positions))
        for subject in subs:
            if all:
                fig, axs = plt.subplots(n_positions, 2, sharex=True, sharey=True)
            else:
                fig, axs = plt.subplots(n_positions, 1, sharex=True)

            if win_len is not None:
                u = np.random.randint(0, acc_per_sub[subject][positions[0]].shape[0] - win_len, 1)[0]
            else:
                u = 0
                win_len = acc_per_sub[subject][positions[0]].shape[0]

            time = time_per_sub[subject][u: u + win_len]
            for p1, position1 in enumerate(positions):
                acc = acc_per_sub[subject][position1][u: u + win_len]

                for p2, position2 in enumerate(positions):
                    cmp_acc = acc_per_sub[subject][position2][u: u + win_len]
                    dists[p1, p2] = maximum_mean_discrepancy(acc, cmp_acc)

                if n_positions == 1:
                    if only_norm:
                        axs.plot(time, acc[:, 3], linewidth=lw)
                    elif only_low:
                        axs.plot(time, acc[:, 4:], linewidth=lw)
                    elif all:
                        axs[0].plot(time, acc[:, :4], linewidth=lw)
                        axs[1].plot(time, acc[:, 4:], linewidth=lw)
                    else:
                        axs.plot(time, acc[:, :3], linewidth=lw)

                    axs.set_ylabel(position1)
                    axs.set_xticks([time[0], time[-1]])

                else:
                    if only_norm:
                        axs[p1].plot(time, acc[:, 3], linewidth=lw)
                    elif only_low:
                        axs[p1].plot(time, acc[:, 4:], linewidth=lw)
                    elif all:
                        axs[p1][0].plot(time, acc[:, :4], linewidth=lw)
                        axs[p1][1].plot(time, acc[:, 4:], linewidth=lw)
                    else:
                        axs[p1].plot(time, acc[:, :3], linewidth=lw)

                    axs[p1][0].set_ylabel(position1)
                    axs[p1][0].set_xticks([time[0], time[-1]])

            if only_norm:
                if n_positions == 1:
                    axs.legend(["acc norm"])

                else:
                    axs[0].legend(["acc norm"])

            elif only_low:
                if n_positions == 1:
                    axs.legend(["x", "y", "z"])

                else:
                    axs[0].legend(["x", "y", "z"])

            elif all:
                if n_positions == 1:
                    axs[0].legend(["x", "y", "z", "xyz"])

                else:
                    axs[0][0].legend(["x", "y", "z", "xyz"])

            else:
                if n_positions == 1:
                    axs.legend(["x", "y", "z"])

                else:
                    axs[0].legend(["acc x", "acc y", "z"])

            plt.suptitle('Subject: ' + str(subject))
            plt.show()


viz = visualizer()

viz.show_window('pamap2',
                ['dominant_lower_arm', 'dominant_lower_leg'],
                'walking',
                win_len=300, all=True)
#
# viz.show_window('rwhar',
#                 ['left_lower_arm', 'left_lower_leg'],
#                 'walking',
#                 win_len=300, all=True)

# viz.show_window('mhealth',
#                 ['right_lower_arm', 'left_lower_leg'],
#                 'walking',
#                 win_len=100, all=True)

# viz.show_window('marea',
#                 ['left_lower_arm', 'left_lower_leg'],
#                 'walking',
#                 win_len=300, all=True)

# viz.show_window('realdisp',
#                 ['left_lower_arm', 'right_lower_arm',
#                  'left_upper_arm', 'right_upper_arm',
#                  'left_lower_leg', 'right_lower_leg',
#                  'left_upper_leg', 'right_upper_leg'],
#                 'walking',
#                 win_len=200, only_norm=False)

# viz.show_window('realdisp',
#                 ['left_lower_arm', 'left_upper_arm'],
#                 'walking',
#                 win_len=300, all=True)
