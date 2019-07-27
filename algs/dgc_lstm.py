import os

import numpy as np
import pandas as pd
import tensorflow as tf
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


def create_data(data, seq_len, horizon, input_dim, mon_ratio, eps):
    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf, size=data.shape, p=(mon_ratio, 1.0 - mon_ratio))
    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    x = np.zeros(shape=(data.shape[0] - seq_len - horizon, seq_len, data.shape[1], input_dim))
    y = np.zeros(shape=(data.shape[0] - seq_len - horizon, horizon, data.shape[1], 1))

    for idx in range(_data.shape[0] - seq_len - horizon):
        _x = _data[idx: (idx + seq_len)]
        _label = _labels[idx: (idx + seq_len)]

        x[idx, :, :, 0] = _x
        x[idx, :, :, 1] = _label

        _y = data[idx + seq_len:idx + seq_len + horizon]

        y[idx] = np.expand_dims(_y, axis=2)

    return x, y


def get_corr_matrix(data, seq_len):
    corr_matrices = np.zeros(shape=(data.shape[0] - seq_len, data.shape[1], data.shape[1]))

    for i in tqdm(range(data.shape[0] - seq_len)):
    # for i in tqdm(range(seq_len)):
        data_corr = data[i:i + seq_len]
        df = pd.DataFrame(data_corr, index=range(data_corr.shape[0]),
                               columns=['{}'.format(x + 1) for x in range(data_corr.shape[1])])

        corr_mx = df.corr().values
        corr_matrices[i] = corr_mx

    nan_idx = []
    for i in range(corr_matrices.shape[0]):
        if not np.any(np.isnan(corr_matrices[i])) and not np.any(np.isinf(corr_matrices[i])):
            nan_idx.append(i)

    corr_matrices = corr_matrices[nan_idx]
    corr_matrix = np.mean(corr_matrices, axis=0)

    return corr_matrix


def generate_data(config):
    data = np.load(config['data']['raw_dataset_dir'])
    data[data <= 0] = 0.1

    data = data.astype("float32")

    if config['data']['data_name'] == 'Abilene':
        day_size = config['data']['Abilene_day_size']
    else:
        day_size = config['data']['Geant_day_size']

    seq_len = int(config['model']['seq_len'])
    horizon = int(config['model']['horizon'])
    input_dim = int(config['model']['input_dim'])

    mon_ratio = float(config['mon_ratio'])

    data = data[:int(data.shape[0] * float(config['data']['data_size']))]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    test_data2d = test_data2d[0:-day_size * 3]

    # print('|--- Normalizing the train set.')
    # scaler = PowerTransformer()
    # scaler.fit(train_data2d)
    # train_data2d_norm = scaler.transform(train_data2d)
    # valid_data2d_norm = scaler.transform(valid_data2d)
    # test_data2d_norm = scaler.transform(test_data2d)

    x_train, y_train = create_data(train_data2d, seq_len=seq_len, horizon=horizon, input_dim=input_dim,
                                   mon_ratio=mon_ratio, eps=train_data2d.mean())
    x_val, y_val = create_data(valid_data2d, seq_len=seq_len, horizon=horizon, input_dim=input_dim,
                               mon_ratio=mon_ratio, eps=train_data2d.mean())
    x_test, y_test = create_data(test_data2d, seq_len=seq_len, horizon=horizon, input_dim=input_dim,
                                 mon_ratio=mon_ratio, eps=train_data2d.mean())

    if not os.path.exists(config['data']['norm_data_path']):
        os.makedirs(config['data']['norm_data_path'])

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(config['data']['norm_data_path'], "%s.npz" % cat),
            x=_x,
            y=_y,
        )

    if not os.path.isfile(config['data']['graph_pkl_filename'] + '.npy'):
        adj_mx = get_corr_matrix(train_data2d, seq_len)
        adj_mx = (adj_mx - adj_mx.min()) / (adj_mx.max() - adj_mx.min())
        np.save(config['data']['graph_pkl_filename'],
                adj_mx)


def train_dgc_lstm(config):
    print('|-- Run model training dgc_lstm.')

    if config['data']['generate_data']:
        generate_data(config)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    adj_mx = np.load(config['data']['graph_pkl_filename'] + '.npy')
    adj_mx = adj_mx.astype('float32')
    with tf.Session(config=tf_config) as sess:
        model = DCRNNSupervisor(adj_mx=adj_mx, **config)
        model.train(sess)


def test_dgc_lstm(config):
    print('|-- Run model testing dgc_lstm.')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    adj_mx = np.load(config['data']['graph_pkl_filename'] + '.npy')
    adj_mx = adj_mx.astype('float32')
    with tf.Session(config=tf_config) as sess:
        model = DCRNNSupervisor(adj_mx=adj_mx, **config)
        model.load(sess, config['train']['model_filename'])
        outputs = model.evaluate(sess)
        np.savez_compressed(os.path.join(HOME_PATH, config['test']['results_path']), **outputs)

        print('Predictions saved as {}.'.format(os.path.join(HOME_PATH, config['test']['results_path']) + '.npz'))
