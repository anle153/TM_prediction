import argparse
import os
import sys

import tensorflow as tf
import yaml

from Models.fwbw_lstm_s.fwbw_lstm_s_supervisor import FwbwLstmSRegression

print(tf.__version__)
if tf.__version__ != '2.0.0':
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    session = tf.Session(config=config_gpu)


def print_fwbw_lstm_s_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(mode))
    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- DATA:\t{}'.format(config['data']['data_name']))
    print('|--- GPU:\t{}'.format(config['gpu']))
    print('|--- MON_RATIO:\t{}'.format(config['mon_ratio']))
    print('|--- BASE_DIR:\t{}'.format(config['base_dir']))
    print('|--- SCALER:\t{}'.format(config['scaler']))

    print('----------------------- MODEL -----------------------')

    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))

    if mode == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))
        print('|--- BATCH:\t{}'.format(config['data']['batch_size']))

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


def build_model(config, is_training=False):
    print('|--- Build models fwbw-lstm.')

    # fwbw-lstm model
    fwbw_net = FwbwLstmSRegression(is_training=is_training, **config)
    fwbw_net.construct_fwbw_lstm()
    return fwbw_net


def train_fwbw_lstm_s(config):
    print('|-- TRAINING FWBW-LSTM-S.')
    if tf.__version__ == '2.0.0':
        fwbw_net = build_model(config)
    else:
        device = config['gpu']
        with tf.device('/device:GPU:{}'.format(device)):
            fwbw_net = build_model(config, is_training=True)

    try:
        if config['train']['continue_train']:
            fwbw_net.load()
    except:
        print('No saved model found!')
    fwbw_net.train()

    return


def evaluate_fwbw_lstm_s(config):
    print('|--- EVALUATING FWBW-LSTM-S')
    if tf.__version__ == '2.0.0':
        fwbw_net = build_model(config)
    else:
        device = config['gpu']
        with tf.device('/device:GPU:{}'.format(device)):
            fwbw_net = build_model(config)

    fwbw_net.load()
    fwbw_net.evaluate()


def test_fwbw_lstm_s(config):
    print('|--- TESTING FWBW-LSTM-S')
    if tf.__version__ == '2.0.0':
        fwbw_net = build_model(config)
    else:
        device = config['gpu']
        with tf.device('/device:GPU:{}'.format(device)):
            fwbw_net = build_model(config)

    fwbw_net.load()
    fwbw_net.test()


if __name__ == '__main__':

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    print_fwbw_lstm_s_info(args.mode, config)

    if args.mode == 'train':
        train_fwbw_lstm_s(config)
    elif args.mode == 'evaluate':
        evaluate_fwbw_lstm_s(config)
    else:
        test_fwbw_lstm_s(config)
