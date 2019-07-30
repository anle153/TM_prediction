import argparse
import os
import sys

import numpy as np
import yaml

from algs.fwbw_lstm import train_fwbw_lstm, test_fwbw_lstm
from common.DataPreprocessing import prepare_train_valid_test_2d


def print_fwbw_lstm_info(config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(config['mode']))
    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- DATA:\t{}'.format(config['data']['data_name']))
    print('|--- GPU:\t{}'.format(config['gpu']))
    print('|--- GENERATE_DATA:\t{}'.format(config['data']['generate_data']))

    print('|--- MON_RATIO:\t{}'.format(config['mon_ratio']))
    print('|--- LOG_DIR:\t{}'.format(config['train']['log_dir']))

    print('----------------------- MODEL -----------------------')

    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- NUM_RNN_LAYERS:\t{}'.format(config['model']['num_rnn_layers']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))

    if config['mode'] == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- LEARNING_RATE:\t{}'.format(config['train']['base_lr']))
        print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
        print('|--- EPSILON:\t{}'.format(config['train']['epsilon']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))
        print('|--- BATCH:\t{}'.format(config['data']['batch_size']))
        print('|--- CONTINUE_TRAIN:\t{}'.format(config['train']['continue_train']))

    if config['mode'] == 'test':
        print('----------------------- TEST -----------------------')
        print('|--- MODEL_FILENAME:\t{}'.format(config['train']['model_filename']))
        print('|--- RUN_TIMES:\t{}'.format(config['test']['run_times']))
        print('|--- FLOW_SELECTION:\t{}'.format(config['test']['flow_selection']))
        print('|--- RESULTS_PATH:\t{}'.format(config['test']['results_path']))
        print('|--- LAMBDA 0:\t{}'.format(config['test']['lambda_0']))
        print('|--- LAMBDA 1:\t{}'.format(config['test']['lambda_1']))
        print('|--- LAMBDA 2:\t{}'.format(config['test']['lambda_2']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


def create_data(data, seq_len, input_dim, mon_ratio, eps):

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    data_x = np.zeros(((data.shape[0] - seq_len - 1) * data.shape[1], seq_len, input_dim))
    data_y_1 = np.zeros(((data.shape[0] - seq_len - 1) * data.shape[1], seq_len, 1))
    data_y_2 = np.zeros(((data.shape[0] - seq_len - 1) * data.shape[1], seq_len))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(1, _data.shape[0] - seq_len):
            _x = _data[idx: (idx + seq_len), flow]
            _label = _labels[idx: (idx + seq_len), flow]

            data_x[i, :, 0] = _x
            data_x[i, :, 1] = _label

            _y_1 = data[(idx + 1):(idx + seq_len + 1), flow]
            _y_2 = data[(idx - 1):(idx + seq_len - 1), flow]

            data_y_1[i] = np.reshape(_y_1, newshape=(seq_len, 1))
            data_y_2[i] = _y_2
            i += 1

    return data_x, data_y_1, data_y_2



def generate_data(config):
    data = np.load(config['data']['raw_dataset_dir'])
    data[data <= 0] = 0.1

    # Convert traffic volume from byte to mega-byte
    data = data / 1000000

    data = data.astype("float32")

    day_size = config['data']['day_size']

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

    x_train, y_train_1, y_train_2 = create_data(train_data2d, seq_len=seq_len, input_dim=input_dim,
                                   mon_ratio=mon_ratio, eps=train_data2d.mean())
    x_val, y_val_1, y_val_2= create_data(valid_data2d, seq_len=seq_len, input_dim=input_dim,
                               mon_ratio=mon_ratio, eps=train_data2d.mean())
    x_test, y_test_1, y_test_2 = create_data(test_data2d, seq_len=seq_len, input_dim=input_dim,
                                 mon_ratio=mon_ratio, eps=train_data2d.mean())

    if not os.path.exists(config['data']['dataset_dir']):
        os.makedirs(config['data']['dataset_dir'])

    for cat in ["train", "val", "test"]:
        _x, _y_1, _y_2 = locals()["x_" + cat], locals()["y_" + cat + '_1'], locals()["y_" + cat+ '_1']
        print(cat, "x: ", _x.shape, "y_1:", _y_1.shape, "y_2:", _y_2.shape)
        np.savez_compressed(
            os.path.join(config['data']['dataset_dir'], "%s.npz" % cat),
            x=_x,
            y_1=_y_1,
            y_2=_y_2,
        )


if __name__ == '__main__':

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    seq_len = str(config['model']['seq_len'])

    if seq_len not in config['data']['dataset_dir'] or seq_len not in config['data'][
        'graph_pkl_filename'] or seq_len not in config['train']['log_dir']:
        raise AttributeError('Check data path!')

    data = np.load(config['data']['raw_dataset_dir'])
    print_fwbw_lstm_info(config)

    if config['data']['generate_data']:
        generate_data(config)

    if config['mode'] == 'train':
        train_fwbw_lstm(config, data)
    else:
        test_fwbw_lstm(config, data)
