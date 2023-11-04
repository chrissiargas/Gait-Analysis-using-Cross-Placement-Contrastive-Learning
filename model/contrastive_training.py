import os
import shutil

import tensorflow as tf
from keras.layers import Input, LSTM, Conv2D, Conv1D, Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.metrics import binary_accuracy
import keras.backend as K

from config_parser import Parser
from data_preprocessing.building import Builder
from simCLR_utilities import get_simCLR


def CLR_train(dataset: Builder, summary=True, verbose=True) -> Model:
    config = Parser()
    config.get_args()
    parent_dir = os.path.dirname(os.getcwd())
    model_name = config.ssl_model

    data_dir = os.path.join(parent_dir, 'save', 'data', model_name + '_data')
    log_dir = os.path.join(parent_dir, 'logs', model_name + '_TB')
    model_dir = os.path.join(parent_dir, 'save', 'models', model_name)
    model_name = '%s.h5' % model_name
    model_file = os.path.join(model_dir, model_name)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    try:
        shutil.rmtree(log_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    try:
        os.remove(model_file)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    train, test = dataset(path=data_dir)

    model = Model()
    if config.ssl_model == 'simCLR':
        optimizer = Adam(learning_rate=float(config.learning_rate))
        loss = BinaryCrossentropy()
        metrics = [binary_accuracy]
        model = get_simCLR(dataset.input_shape)

    if summary and verbose:
        print(model.summary())

    model.compile(optimizer, loss, metrics)

    if config.load_model:
        model.load_weights(model_dir)


