import argparse
import os
import sys

import tensorflow as tf
import yaml

from Models.ConvLSTM_supervised import ConvLSTM

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
session = tf.Session(config=config_gpu)


def print_conv_lstm_info(mode, config):
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
    print('|--- WIDE:\t{}'.format(config['model']['wide']))
    print('|--- HIGH:\t{}'.format(config['model']['high']))
    print('|--- CHANNEL:\t{}'.format(config['model']['channel']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- NUM_RNN_LAYERS:\t{}'.format(config['model']['num_rnn_layers']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))
    print('|--- FILTERS:\t{}'.format(config['model']['filters']))
    print('|--- KERNEL_SIZE:\t{}'.format(config['model']['kernel_size']))
    print('|--- STRIDES:\t{}'.format(config['model']['strides']))

    if mode == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- LEARNING_RATE:\t{}'.format(config['train']['base_lr']))
        print('|--- RNN_DROPOUT:\t{}'.format(config['train']['rnn_dropout']))
        print('|--- CONV_DROPOUT:\t{}'.format(config['train']['conv_dropout']))
        print('|--- EPSILON:\t{}'.format(config['train']['epsilon']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))
        print('|--- BATCH:\t{}'.format(config['data']['batch_size']))
        print('|--- CONTINUE_TRAIN:\t{}'.format(config['train']['continue_train']))

    if mode == 'test':
        print('----------------------- TEST -----------------------')
        print('|--- LOG_DIR:\t{}'.format(config['train']['log_dir']))
        print('|--- RUN_TIMES:\t{}'.format(config['test']['run_times']))
        print('|--- FLOW_SELECTION:\t{}'.format(config['test']['flow_selection']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


def build_model(config):
    print('|--- Build models conv-lstm.')

    conv_lstm_net = ConvLSTM(**config)
    conv_lstm_net.construct_conv_lstm()
    print(conv_lstm_net.model.summary())
    conv_lstm_net.plot_models()
    return conv_lstm_net


def train_conv_lstm(config):
    print('|-- Run model training conv-lstm.')

    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        conv_lstm_net = build_model(config)

    conv_lstm_net.train()

    return


def evaluate_conv_lstm(config):
    print('|--- EVALUATE CONV-LSTM')
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        conv_lstm_net = build_model(config)

    conv_lstm_net.load()
    conv_lstm_net.evaluate()


def test_conv_lstm(config):
    print('|--- TEST CONV-LSTM')
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        conv_lstm_net = build_model(config)
    conv_lstm_net.load()
    conv_lstm_net.test()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config-file', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    print_conv_lstm_info(args.mode, config)

    if args.mode == 'train':
        train_conv_lstm(config)
    elif args.mode == 'evaluate' or args.mode == 'evaluation':
        evaluate_conv_lstm(config)
    elif args.mode == "test":
        test_conv_lstm(config)
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
