import os
import shutil
from typing import *

import tensorflow as tf
from keras.layers import Input, LSTM, Conv2D, Conv1D, Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Optimizer
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.metrics import binary_accuracy
import keras.backend as K

from config_parser import Parser
from data_preprocessing.building import Builder
from simCLR_utilities import simCLR

import matplotlib.pyplot as plt


def CLR_train(dataset: Builder, summary=True, verbose=True) -> Tuple[Dict, Dict]:
    config = Parser()
    config.get_args()

    model_type = config.ssl_model
    data_dir = os.path.join('save', 'data', model_type + '_data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    train, test = dataset(path=data_dir, verbose=verbose)

    history = {}
    models = {}
    for target in dataset.targets:
        print(target)

        model_name = model_type + '_' + target
        log_dir = os.path.join('logs', model_name + '_TB')
        model_dir = os.path.join('save', 'models', model_name)
        model_name = '%s.h5' % model_name
        model_file = os.path.join(model_dir, model_name)

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

        models[target] = Model()

        if config.lr_decay == 'cosine':
            learning_rate = tf.keras.experimental.CosineDecay(initial_learning_rate=config.lr,
                                                              decay_steps=config.decay_steps)
        else:
            learning_rate = config.lr

        if config.optimizer == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = Optimizer()

        if config.ssl_model == 'simCLR':
            models[target] = simCLR(inputs_shape=dataset.input_shape)

        if summary:
            print(models[target].summary())

        models[target].compile(optimizer)

        if config.load_model:
            if not os.path.isdir(model_dir):
                return None

            models[target].load_weights(model_dir)

        train_steps = dataset.train_batches[target]
        test_steps = dataset.test_batches[target]

        tensorboard_callback = TensorBoard(log_dir, histogram_freq=1)

        save_model = ModelCheckpoint(
            filepath=model_file,
            monitor='val_c_loss',
            verbose=verbose,
            save_best_only=True,
            mode='min',
            save_weights_only=True)

        early_stopping = EarlyStopping(
            monitor='val_c_loss',
            min_delta=0,
            patience=50,
            mode='auto',
            verbose=verbose)

        callbacks = [
            tensorboard_callback,
            save_model,
            early_stopping
        ]

        history[target] = models[target].fit(train[target],
                                             epochs=config.epochs,
                                             steps_per_epoch=train_steps,
                                             validation_data=test[target],
                                             validation_steps=test_steps,
                                             callbacks=callbacks,
                                             use_multiprocessing=True,
                                             verbose=1)

        models[target].load_weights(model_file)

    del train
    del test
    del dataset

    return models, history
