import argparse
import os.path

import yaml
from os.path import dirname, abspath

class Parser:
    def __init__(self):
        self.encoder = None
        self.common_augmentations = None
        self.shift_length = None
        self.shift_pad = None
        self.cnn_blocks = None
        self.in_activities = None
        self.neg_pos = None
        self.neg_act = None
        self.neg_sub = None
        self.neg_ds = None
        self.attach_head = None
        self.clr_temp = None
        self.lr_decay = None
        self.decay_steps = None
        self.epochs = None
        self.optimizer = None
        self.load_model = None
        self.lr = None
        self.ssl_model = None
        self.augmentations = None
        self.use_features = None
        self.batch_method = None
        self.load_data = None
        self.tolerance = None
        self.selection_method = None
        self.step = None
        self.stride = None
        self.length = None
        self.duration = None
        self.hold_out = None
        self.split_type = None
        self.in_datasets = None
        self.in_positions = None
        self.prod_features = None
        self.cleaner = None
        self.rescaler = None
        self.filter_window = None
        self.filter = None
        self.batch_size = None
        self.activities = None
        self.positions = None
        self.fs = None
        self.path = None
        self.datasets = None

        self.parser = argparse.ArgumentParser(
            description="pre-processing and training parameters"
        )

    def __call__(self, *args, **kwargs):
        project_root = dirname(abspath(__file__))
        config_path = os.path.join(project_root, 'config.yaml')

        self.parser.add_argument(
            '--config',
            default=config_path,
            help='config file location'
        )

        self.parser.add_argument(
            '--data_args',
            default=dict(),
            type=dict,
            help='pre-processing arguments'
        )

        self.parser.add_argument(
            '--main_args',
            default=dict(),
            type=dict,
            help='training arguments'
        )

    def get_args(self):
        self.__call__()
        args = self.parser.parse_args(args=[])
        configFile = args.config

        assert configFile is not None

        with open(configFile, 'r') as cf:
            defaultArgs = yaml.load(cf, Loader=yaml.FullLoader)

        keys = vars(args).keys()

        for defaultKey in defaultArgs.keys():
            if defaultKey not in keys:
                print('WRONG ARG: {}'.format(defaultKey))
                assert (defaultKey in keys)

        self.parser.set_defaults(**defaultArgs)
        args = self.parser.parse_args(args=[])

        self.datasets = args.data_args['datasets']
        self.path = args.data_args['path']
        self.fs = args.data_args['fs']
        self.positions = args.data_args['positions']
        self.activities = args.data_args['activities']

        self.load_data = args.main_args['load_data']
        self.in_datasets = args.main_args['datasets']
        self.in_positions = args.main_args['positions']
        self.in_activities = args.main_args['activities']
        self.cleaner = args.main_args['cleaner']
        self.prod_features = args.main_args['produce_features']
        self.filter = args.main_args['filter']
        self.filter_window = args.main_args['filter_window']
        self.rescaler = args.main_args['rescaler']
        self.split_type = args.main_args['split_type']
        self.hold_out = args.main_args['hold_out']
        self.duration = args.main_args['duration']
        self.length = int(self.duration * self.fs)
        self.stride = args.main_args['stride']
        self.step = int(self.stride * self.fs)
        self.selection_method = args.main_args['selection_method']
        self.tolerance = args.main_args['tolerance']
        self.use_features = args.main_args['use_features']
        self.augmentations = args.main_args['augmentations']
        self.batch_method = args.main_args['batch_method']
        self.batch_size = args.main_args['batch_size']
        self.ssl_model = args.main_args['ssl_model']
        self.lr = args.main_args['learning_rate']
        self.load_model = args.main_args['load_model']
        self.clr_temp = args.main_args['clr_temperature']
        self.optimizer = args.main_args['optimizer']
        self.epochs = args.main_args['epochs']
        self.decay_steps = args.main_args['decay_steps']
        self.lr_decay = args.main_args['lr_decay']
        self.attach_head = args.main_args['attach_head']
        self.neg_ds = args.main_args['negative_dataset']
        self.neg_sub = args.main_args['negative_subject']
        self.neg_act = args.main_args['negative_activity']
        self.neg_pos = args.main_args['negative_position']
        self.cnn_blocks = args.main_args['cnn_blocks']
        self.shift_pad = args.main_args['shift_pad']
        self.shift_length = int(self.shift_pad * self.fs)
        self.common_augmentations = [] if args.main_args['common_augmentations'] is None else args.main_args['common_augmentations']
        self.encoder = args.main_args['encoder']
        
        
        return
