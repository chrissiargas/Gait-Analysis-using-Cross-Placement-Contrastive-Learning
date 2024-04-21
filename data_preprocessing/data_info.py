import os
from config_parser import Parser


class Info:
    def __init__(self):
        config = Parser()
        config.get_args()
        self.conf = config

        self.pamap2_path = os.path.join(
            os.path.expanduser('~'),
            config.path,
            'PAMAP2',
            'PAMAP2_Dataset',
            'Protocol'
        )

        self.rwhar_path = os.path.join(
            os.path.expanduser('~'),
            config.path,
            'REAL-WORLD'
        )

        self.mhealth_path = os.path.join(
            os.path.expanduser('~'),
            config.path,
            'MHEALTH',
            'MHEALTHDATASET'
        )

        self.marea_path = os.path.join(
            os.path.expanduser('~'),
            config.path,
            'MAREA',
            'Data_csv format'
        )

        self.realdisp_path = os.path.join(
            os.path.expanduser('~'),
            config.path,
            'REALDISP'
        )

        self.pamap2_fs = 100

        self.pamap2_activities = {0: 'transient',
                                  1: 'lying',
                                  2: 'sitting',
                                  3: 'standing',
                                  4: 'walking',
                                  5: 'running',
                                  6: 'cycling',
                                  7: 'Nordic walking',
                                  9: 'watching TV',
                                  10: 'computer work',
                                  11: 'car driving',
                                  12: 'ascending stairs',
                                  13: 'descending stairs',
                                  16: 'vacuum cleaning',
                                  17: 'ironing',
                                  18: 'folding laundry',
                                  19: 'house cleaning',
                                  20: 'playing soccer',
                                  24: 'rope jumping'}

        self.pamap2_positions = {
            "chest": 1,
            "hand": 2,
            "ankle": 3
        }

        self.pamap2_act_pairs = {'transient': 'no pair',
                                 'lying': 'lying',
                                 'sitting': 'sitting',
                                 'standing': 'standing',
                                 'walking': 'walking',
                                 'running': 'running',
                                 'cycling': 'no pair',
                                 'Nordic walking': 'no pair',
                                 'watching TV': 'no pair',
                                 'computer work': 'no pair',
                                 'car driving': 'no pair',
                                 'ascending stairs': 'stairs',
                                 'descending stairs': 'stairs',
                                 'vacuum cleaning': 'no pair',
                                 'ironing': 'no pair',
                                 'folding laundry': 'no pair',
                                 'house cleaning': 'no pair',
                                 'playing soccer': 'no pair',
                                 'rope jumping': 'no pair'}

        self.pamap2_pos_pairs = {"chest": 'no pair',
                                 "hand": 'dominant_lower_arm',
                                 "ankle": 'dominant_lower_leg'}

        self.rwhar_fs = 50

        self.rwhar_activities = {
            "climbingdown": 'climbing down',
            "climbingup": 'climbing up',
            "jumping": 'jumping',
            "lying": 'lying',
            "running": 'running',
            "sitting": 'sitting',
            "standing": 'standing',
            "walking": 'walking',
        }

        self.rwhar_positions = {
            "chest": 1,
            "forearm": 2,
            "head": 3,
            "shin": 4,
            "thigh": 5,
            "upperarm": 6,
            "waist": 7
        }

        self.rwhar_act_pairs = {
            'climbing down': 'climbing',
            'climbing up': 'climbing',
            "jumping": 'jumping',
            "lying": 'lying',
            "running": 'running',
            "sitting": 'sitting',
            "standing": 'standing',
            "walking": 'walking',
        }

        self.rwhar_pos_pairs = {
            "chest": 'no pair',
            "forearm": 'left_lower_arm',
            "head": 'no pair',
            "shin": 'left_lower_leg',
            "thigh": 'left_upper_leg',
            "upperarm": 'left_upper_arm',
            "waist": 'no pair'
        }

        self.mhealth_fs = 50

        self.mhealth_activities = {
            0: "none",
            1: "standing",
            2: "sitting",
            3: "lying",
            4: "walking",
            5: "climbing",
            6: "waist bends forward",
            7: "frontal elevation of arms",
            8: "knees bending",
            9: "cycling",
            10: "jogging",
            11: "running",
            12: "jump front & back"
        }

        self.mhealth_positions = {
            "chest": 1,
            "arm": 2,
            "ankle": 3
        }

        self.mhealth_act_pairs = {
            "none": 'no pair',
            "standing": 'standing',
            "sitting": 'sitting',
            "lying": 'lying',
            "walking": 'walking',
            "climbing": 'climbing',
            "waist bends forward": 'no pair',
            "frontal elevation of arms": 'no pair',
            "knees bending": 'no pair',
            "cycling": 'no pair',
            "jogging": 'jogging',
            "running": 'running',
            "jump front & back": 'no pair'
        }

        self.mhealth_pos_pairs = {
            "chest": 'no pair',
            "arm": 'right_lower_arm',
            "ankle": 'left_lower_leg'
        }

        self.marea_fs = 128

        self.marea_activities = [
            "treadmill_walk", "treadmill_run", "treadmill_slope_walk",
            "indoor_walk", "indoor_run", "outdoor_walk", "outdoor_run"
        ]

        self.marea_positions = {
            'LF': 1,
            'RF': 2,
            'Waist': 3,
            'Wrist': 4
        }

        self.marea_act_pairs = {
            "undefined": 'no pair',
            "treadmill_walk": 'treadmill_walking',
            "treadmill_run": 'treadmill_running',
            "treadmill_slope_walk": 'treadmill_slope_walking',
            "indoor_walk": 'indoor_walking',
            "indoor_run": 'indoor_running',
            "outdoor_walk": 'outdoor_walking',
            "outdoor_run": 'outdoor_running'
        }

        self.marea_pos_pairs = {
            'LF': 'left_lower_leg',
            'RF': 'right_lower_leg',
            'Waist': 'no pair',
            'Wrist': 'left_lower_arm'
        }

        self.realdisp_fs = 50

        self.realdisp_positions = {'RLA': 1,
                                   'RUA': 2,
                                   'BACK': 3,
                                   'LUA': 4,
                                   'LLA': 5,
                                   'RC': 6,
                                   'RT': 7,
                                   'LT': 8,
                                   'LC': 9}

        self.realdisp_activities = {0: 'No Activity',
                                    1: "Walking",
                                    2: "Jogging",
                                    3: "Running",
                                    4: "Jump up",
                                    5: "Jump front & back",
                                    6: "Jump sideways",
                                    7: "Jump leg/arms open/closed",
                                    8: "Jump rope",
                                    9: "Trunk twist (arms outstretched)",
                                    10: "Trunk twist (elbows bent)",
                                    11: "Waist bends forward",
                                    12: "Waist rotation",
                                    13: "Waist bends (reach foot with opposite hand)",
                                    14: "Reach heels backwards",
                                    15: "Lateral bend (10_ to the left + 10_ to the right)",
                                    16: "Lateral bend with arm up (10_ to the left + 10_ to the right)",
                                    17: "Repetitive forward stretching",
                                    18: "Upper trunk and lower body opposite twist",
                                    19: "Lateral elevation of arms",
                                    20: "Frontal elevation of arms",
                                    21: "Frontal hand claps",
                                    22: "Frontal crossing of arms",
                                    23: "Shoulders high-amplitude rotation",
                                    24: "Shoulders low-amplitude rotation",
                                    25: "Arms inner rotation",
                                    26: "Knees (alternating) to the breast",
                                    27: "Heels (alternating) to the backside",
                                    28: "Knees bending (crouching)",
                                    29: "Knees (alternating) bending forward",
                                    30: "Rotation on the knees",
                                    31: "Rowing",
                                    32: "Elliptical bike",
                                    33: "Cycling"}

        self.realdisp_pos_pairs = {'RLA': 'right_lower_arm',
                                   'RUA': 'right_upper_arm',
                                   'BACK': 'no pair',
                                   'LUA': 'left_upper_arm',
                                   'LLA': 'left_lower_arm',
                                   'RC': 'right_lower_leg',
                                   'RT': 'right_upper_leg',
                                   'LT': 'left_upper_leg',
                                   'LC': 'left_lower_leg'}

        self.realdisp_act_pairs = {"No Activity": 'no pair',
                                   "Walking": 'walking',
                                   "Jogging": 'jogging',
                                   "Running": 'running',
                                   "Jump up": 'jumping',
                                   "Jump front & back": 'no pair',
                                   "Jump sideways": 'no pair',
                                   "Jump leg/arms open/closed": 'no pair',
                                   "Jump rope": 'no pair',
                                   "Trunk twist (arms outstretched)": 'no pair',
                                   "Trunk twist (elbows bent)": 'no pair',
                                   "Waist bends forward": 'no pair',
                                   "Waist rotation": 'no pair',
                                   "Waist bends (reach foot with opposite hand)": 'no pair',
                                   "Reach heels backwards": 'no pair',
                                   "Lateral bend (10_ to the left + 10_ to the right)": 'no pair',
                                   "Lateral bend with arm up (10_ to the left + 10_ to the right)": 'no pair',
                                   "Repetitive forward stretching": 'no pair',
                                   "Upper trunk and lower body opposite twist": 'no pair',
                                   "Lateral elevation of arms": 'no pair',
                                   "Frontal elevation of arms": 'no pair',
                                   "Frontal hand claps": 'no pair',
                                   "Frontal crossing of arms": 'no pair',
                                   "Shoulders high-amplitude rotation": 'no pair',
                                   "Shoulders low-amplitude rotation": 'no pair',
                                   "Arms inner rotation": 'no pair',
                                   "Knees (alternating) to the breast": 'no pair',
                                   "Heels (alternating) to the backside": 'no pair',
                                   "Knees bending (crouching)": 'no pair',
                                   "Knees (alternating) bending forward": 'no pair',
                                   "Rotation on the knees": 'no pair',
                                   "Rowing": 'no pair',
                                   "Elliptical bike": 'no pair',
                                   "Cycling": 'no pair'}

        self.ds_positions = {
            'pamap2': ['dominant_lower_arm', 'dominant_lower_leg'],
            'rwhar': ['left_lower_arm', 'left_upper_arm', 'left_lower_leg', 'left_upper_leg'],
            'mhealth': ['right_lower_arm', 'left_lower_leg'],
            'marea': ['left_lower_leg', 'right_lower_leg', 'left_lower_arm'],
            'realdisp': ['right_lower_arm', 'right_upper_arm',
                         'left_upper_arm', 'left_lower_arm',
                         'right_lower_leg', 'right_upper_leg',
                         'left_upper_leg', 'left_lower_leg']
        }
