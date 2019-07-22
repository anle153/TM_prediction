import os

import numpy as np
import tensorflow as tf
import yaml
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm

from Models.dcrnn_supervisor import DCRNNSupervisor
from common.DataPreprocessing import prepare_train_valid_test_2d

HOME_PATH = os.path.expanduser('~/TM_prediction')

DATA_SETS = ['Abilene', 'Geant']
DATA_NAME = DATA_SETS[0]
ABILENE_DAY_SIZE = 288
GEANT_DAY_SIZE = 288
CONFIG_PATH = os.path.join(HOME_PATH, 'Config')
CONFIG_FILE = 'config_dgclstm.yaml'
RESULTS_PATH = os.path.join(HOME_PATH, 'results')


def get_corr_matrix(data, step):
    corr_matrices = np.ones(shape=(data.shape[0] - step + 1, data.shape[1], data.shape[1]))

    for i in tqdm(range(data.shape[0] - step + 1)):
        corr_matrices[i] = np.corrcoef(data[i:i + step])

    corr_matrix = np.mean(corr_matrices, axis=0)

    return corr_matrix


def generate_data(config):
    data = np.load(config['data']['data_path'])
    data[data <= 0] = 0.1

    if config['data']['data_name'] == 'Abilene':
        day_size = config['abilene_day_size']
    else:
        day_size = config['geant_day_size']

    step = config['step']

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')
    scaler = PowerTransformer()
    scaler.fit(train_data2d)
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    x_train, y_train = create_data(train_data2d_norm)
    x_val, y_val = create_data(valid_data2d_norm)
    x_test, y_test = create_data(test_data2d_norm)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(config['data']['norm_data_path'], "%s.npz" % cat),
            x=_x,
            y=_y,
        )

    adj_mx = get_corr_matrix(train_data2d, step)
    np.save(os.path.join(config['data']['adj_mx_path'], "{}.npz".format('Corr_matrix')),
            adj_mx)


def train_dgc_lstm():
    print('|-- Run model training dgc_lstm.')

    with open(os.path.join(CONFIG_PATH, CONFIG_FILE)) as f:
        config = yaml.load(f)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    adj_mx = np.load(os.path.join(config['data']['adj_mx_path'], "{}.npz".format('Corr_matrix')))

    with tf.Session(config=tf_config) as sess:
        model = DCRNNSupervisor(adj_mx=adj_mx, **config)
        model.train(sess)


def test_dgc_lstm():
    print('|-- Run model testing dgc_lstm.')

    with open(os.path.join(CONFIG_PATH, CONFIG_FILE)) as f:
        config = yaml.load(f)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    adj_mx = np.load(os.path.join(config['data']['adj_mx_path'], "{}.npz".format('Corr_matrix')))

    with tf.Session(config=tf_config) as sess:
        model = DCRNNSupervisor(adj_mx=adj_mx, **config)
        model.load(sess, config['train']['model_filename'])
        outputs = model.evaluate(sess)
        np.savez_compressed(os.path.join(config['test']['results_path'],
                                         config['test']['results_name']),
                            **outputs)

        print('Predictions saved as {}.'.format(os.path.join(config['test']['results_path'],
                                                             config['test']['results_name'])))
