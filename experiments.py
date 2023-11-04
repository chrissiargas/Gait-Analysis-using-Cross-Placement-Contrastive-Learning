import copy
import pandas as pd
from datasets import multiset
from simCLR import train_evaluate
import ruamel.yaml
import os
import time
import gc
from parameters import simCLR_params


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


def save(path, scores, hparams=None):
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

    scoresFile = os.path.join(path, "scores.csv")
    paramsFile = os.path.join(path, "parameters.yaml")

    scores.to_csv(scoresFile, index=False)
    config_save(paramsFile)


def simCLR_experiment():
    # config_edit('train_args', 'events', [event])

    data = Dataset(regenerate=False)
    stats = train_evaluate(data, summary=False, verbose=1, mVerbose=False)
    del data

    return stats


def simCLR(archive_path):
    parameters = simCLR_params

    for param_name, param_value in parameters.items():
        config_edit('train_args', param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))

    results = pd.DataFrame()
    reset_tensorflow_keras_backend()
    results_per_xp = simCLR_params()
    results = pd.concat([results, pd.DataFrame([results_per_xp])],
                        ignore_index=True)

    save(archive, results)

