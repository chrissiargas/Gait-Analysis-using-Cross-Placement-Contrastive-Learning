import copy
import pandas as pd
import ruamel.yaml
import os
import time
import gc

from parameters import simCLR_params
from data_preprocessing.building import Builder
from model.contrastive_training import CLR_train

import matplotlib.pyplot as plt

def reset_tensorflow_keras_backend():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    _ = gc.collect()


def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:
            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)


def config_save(paramsFile):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        parameters = yaml.load(fp)

    with open(paramsFile, 'w') as fb:
        yaml.dump(parameters, fb)


def save(path, model, history, hparams=None):
    if not hparams:
        try:
            os.makedirs(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    else:
        try:
            path = os.path.join(path, hparams)
            os.makedirs(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    paramsFile = os.path.join(path, "parameters.yaml")
    config_save(paramsFile)

    fig = plt.figure()
    plt.plot(history.history['c_loss'])
    plt.plot(history.history['val_c_loss'])
    plt.title('Sim-CLR performance')
    plt.ylabel('contrastive loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    historyFile = os.path.join(path, hparams + "_history.png")
    plt.savefig(historyFile)

    plt.close(fig)


REGENERATE = False


def simCLR_experiment():

    data = Builder(regenerate=REGENERATE)
    model, history = CLR_train(data, summary=True, verbose=False)
    del data

    return model, history


def simCLR(archive_path):
    parameters = simCLR_params

    for param_name, param_value in parameters.items():
        config_edit('main_args', param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))

    reset_tensorflow_keras_backend()

    datasets = ['marea']
    split_type = 'loso'
    hold_out = 3

    config_edit('main_args', 'datasets', datasets)
    config_edit('main_args', 'split_type', split_type)
    config_edit('main_args', 'hold_out', hold_out)

    xp_model, xp_history = simCLR_experiment()
    hparams = '_'.join(datasets) + '_' +  split_type + '_' + str(hold_out)
    save(archive, xp_model, xp_history, hparams=hparams)