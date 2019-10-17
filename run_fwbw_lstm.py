import argparse
import os
import sys

import tensorflow as tf
import yaml

from Models.fwbw_lstm_supervisor import FwbwLstmRegression

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
session = tf.Session(config=config_gpu)


def print_fwbw_lstm_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(mode))
    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- DATA:\t{}'.format(config['data']['data_name']))
    print('|--- GPU:\t{}'.format(config['gpu']))
    print('|--- GENERATE_DATA:\t{}'.format(config['data']['generate_data']))

    print('|--- MON_RATIO:\t{}'.format(config['mon_ratio']))
    print('|--- BASE_DIR:\t{}'.format(config['base_dir']))

    print('----------------------- MODEL -----------------------')

    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- NUM_RNN_LAYERS:\t{}'.format(config['model']['num_rnn_layers']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))

    if mode == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- LEARNING_RATE:\t{}'.format(config['train']['base_lr']))
        print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
        print('|--- EPSILON:\t{}'.format(config['train']['epsilon']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))
        print('|--- BATCH:\t{}'.format(config['data']['batch_size']))
        print('|--- CONTINUE_TRAIN:\t{}'.format(config['train']['continue_train']))

    if mode == 'test':
        print('----------------------- TEST -----------------------')
        print('|--- RUN_TIMES:\t{}'.format(config['test']['run_times']))
        print('|--- FLOW_SELECTION:\t{}'.format(config['test']['flow_selection']))
        print('|--- LAMBDA 0:\t{}'.format(config['test']['lamda_0']))
        print('|--- LAMBDA 1:\t{}'.format(config['test']['lamda_1']))
        print('|--- LAMBDA 2:\t{}'.format(config['test']['lamda_2']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


# def create_data_fwbw_lstm(data, seq_len, input_dim, mon_ratio, eps):
#
#     _tf = np.array([1.0, 0.0])
#     _labels = np.random.choice(_tf,
#                                size=data.shape,
#                                p=(mon_ratio, 1 - mon_ratio))
#     data_x = np.zeros(((data.shape[0] - seq_len - 1) * data.shape[1], seq_len, input_dim))
#     data_y_1 = np.zeros(((data.shape[0] - seq_len - 1) * data.shape[1], seq_len, 1))
#     data_y_2 = np.zeros(((data.shape[0] - seq_len - 1) * data.shape[1], seq_len))
#
#     _data = np.copy(data)
#
#     _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)
#
#     i = 0
#     for flow in range(_data.shape[1]):
#         for idx in range(1, _data.shape[0] - seq_len):
#             _x = _data[idx: (idx + seq_len), flow]
#             _label = _labels[idx: (idx + seq_len), flow]
#
#             data_x[i, :, 0] = _x
#             data_x[i, :, 1] = _label
#
#             _y_1 = data[(idx + 1):(idx + seq_len + 1), flow]
#             _y_2 = data[(idx - 1):(idx + seq_len - 1), flow]
#
#             data_y_1[i] = np.reshape(_y_1, newshape=(seq_len, 1))
#             data_y_2[i] = _y_2
#             i += 1
#
#     return data_x, data_y_1, data_y_2

def build_model(config):
    print('|--- Build models fwbw-lstm.')

    # fwbw-lstm model
    fwbw_net = FwbwLstmRegression(**config)
    fwbw_net.construct_fwbw_lstm()
    # print(fwbw_net.model.summary())
    fwbw_net.plot_models()
    return fwbw_net


def train_fwbw_lstm(config):
    print('|-- Run model training fwbw_lstm.')

    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        fwbw_net = build_model(config)

    fwbw_net.train()

    return


def evaluate_fwbw_lstm(config):
    print('|--- EVALUATE FWBW-LSTM')
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        fwbw_net = build_model(config)

    fwbw_net.load()
    fwbw_net.evaluate()


def test_fwbw_lstm(config):
    print('|--- TEST FWBW-LSTM')
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        fwbw_net = build_model(config)
    fwbw_net.load()
    fwbw_net.test()


if __name__ == '__main__':

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config-file', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    print_fwbw_lstm_info(args.mode, config)

    if args.mode == 'train':
        train_fwbw_lstm(config)
    elif args.mode == 'evaluate':
        evaluate_fwbw_lstm(config)
    else:
        test_fwbw_lstm(config)
