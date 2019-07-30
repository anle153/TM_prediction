import argparse
import os
import sys

import numpy as np
import yaml

from algs.fwbw_lstm import train_fwbw_lstm, test_fwbw_lstm
from common import Config_fwbw_lstm as Config


def print_fwbw_lstm_info(config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(config['mode']))
    print('|--- ALG:\t{}'.format(Config.ALG))
    print('|--- TAG:\t{}'.format(Config.TAG))
    print('|--- DATA:\t{}'.format(config['data']['data_name']))
    print('|--- GPU:\t{}'.format(config['gpu']))
    print('|--- GENERATE_DATA:\t{}'.format(config['data']['generate_data']))

    print('|--- MON_RATIO:\t{}'.format(config['mon_ratio']))
    print('|--- BATCH:\t{}'.format(config['data']['batch_size']))

    print('----------------------- MODEL -----------------------')

    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- NUM_RNN_LAYERS:\t{}'.format(config['model']['num_rnn_layers']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))
    print('|--- FLOW_SELECTION:\t{}'.format(config['model']['seq_len']))

    print('----------------------- TRAIN -----------------------')

    print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
    print('|--- LEARNING_RATE:\t{}'.format(config['train']['base_lr']))
    print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
    print('|--- EPSILON:\t{}'.format(config['train']['epsilon']))
    print('|--- LOG_DIR:\t{}'.format(config['train']['log_dir']))

    if config['mode'] == 'test':
        print('----------------------- TEST -----------------------')
        print('|--- MODEL_FILENAME:\t{}'.format(config['train']['model_filename']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


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

    data = np.load(config['data']['raw_dataset_dir'])
    print_fwbw_lstm_info(config)

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        train_fwbw_lstm(data)
    else:
        test_fwbw_lstm(data)
