import numpy as np
import pandas as pd
import os
import yaml
from common.DataPreprocessing import prepare_train_valid_test_2d
from tqdm import tqdm


HOME_PATH = os.path.expanduser('~/TM_prediction')
CONFIG_PATH = os.path.join(HOME_PATH, 'Config')
CONFIG_FILE = 'config_dgc_lstm.yaml'


def create_data(data, seq_len, horizon, input_dim, mon_ratio, eps):
    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf, size=data.shape, p=(mon_ratio, 1 - mon_ratio))
    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    x = np.zeros(shape=(data.shape[0] - seq_len - horizon, seq_len, data.shape[1], input_dim))
    y = np.zeros(shape=(data.shape[0] - seq_len - horizon, horizon, data.shape[1], 1))

    for idx in range(_data.shape[0] - seq_len - horizon):
        _x = _data[idx: (idx + seq_len)]
        _label = _labels[idx: (idx + seq_len)]

        x[idx, :, :, 0] = _x
        x[idx, :, :, 1] = _label

        _y = _data[idx + seq_len:idx + seq_len + horizon]

        y[idx] = np.expand_dims(_y, axis=2)

    return x, y


with open(os.path.join(CONFIG_PATH, CONFIG_FILE)) as f:
    config = yaml.load(f)

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

print('|--- Splitting train-test set.')
train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
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

mean_train = x_train.mean()
std_train = x_train.std()

x_train_norm = (x_train - mean_train) / std_train
y_train_norm = (y_train - mean_train) / std_train
x_val_norm = (x_val - mean_train) / std_train
y_val_norm = (y_val - mean_train) / std_train
x_test_norm = (x_test - mean_train) / std_train
y_test_norm = (y_test - mean_train) / std_train

if np.any(np.isinf(x_train_norm)):
    print("INF")
    raise ValueError

if np.any(np.isnan(x_train_norm)):
    print("NAN")
    raise ValueError

if np.any(np.isinf(x_val_norm)):
    print("INF")
    raise ValueError

if np.any(np.isnan(x_val_norm)):
    print("NAN")
    raise ValueError

if np.any(np.isinf(x_test_norm)):
    print("INF")
    raise ValueError

if np.any(np.isnan(x_test_norm)):
    print("NAN")
    raise ValueError

if np.any(np.isinf(y_train_norm)):
    print("INF")
    raise ValueError

if np.any(np.isnan(y_train_norm)):
    print("NAN")
    raise ValueError

if np.any(np.isinf(y_val_norm)):
    print("INF")
    raise ValueError

if np.any(np.isnan(y_val_norm)):
    print("NAN")
    raise ValueError

if np.any(np.isinf(y_test_norm)):
    print("INF")
    raise ValueError

if np.any(np.isnan(y_test_norm)):
    print("NAN")
    raise ValueError


def get_corr_matrix(data, seq_len):
    corr_matrices = np.zeros(shape=(data.shape[0] - seq_len, data.shape[1], data.shape[1]))

    for i in tqdm(range(data.shape[0] - seq_len)):
        data_corr = data[i:i + seq_len]
        data_hm = pd.DataFrame(data_corr, index=range(data_corr.shape[0]),
                               columns=['{}'.format(x + 1) for x in range(data_corr.shape[1])])

        corr_mx = data_hm.corr()
        # print("|---- Corr MX")
        # print(corr_mx.shape)
        # print(corr_mx.max())
        # print(corr_mx.min())
        # print(corr_mx.mean())
        # print(corr_mx.std())

        corr_matrices[i] = data_hm.corr()

    corr_matrix = np.mean(corr_matrices, axis=0)

    return corr_matrix

adj_mx = get_corr_matrix(train_data2d, seq_len)

print("|-- ADJ MX")
print(adj_mx.shape)
print(adj_mx.max())
print(adj_mx.min())
print(adj_mx.mean())
print(adj_mx.std())



