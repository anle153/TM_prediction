import argparse
import os
import sys

import yaml

from algs.dgc_lstm import train_dgc_lstm, test_dgc_lstm, evaluate_dgc_lstm

HOME_PATH = os.path.expanduser('~/TM_prediction')
CONFIG_PATH = os.path.join(HOME_PATH, 'Config')
CONFIG_FILE = 'config_dgc_lstm.yaml'


def print_dgc_lstm_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(mode))
    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- DATA:\t{}'.format(config['data']['data_name']))
    print('|--- GPU:\t{}'.format(config['gpu']))
    print('|--- GENERATE_DATA:\t{}'.format(config['data']['generate_data']))

    print('|--- MON_RATIO:\t{}'.format(config['mon_ratio']))
    print('|--- BASED_DIR:\t{}'.format(config['base_dir']))

    print('|--- ADJ_THRESHOLD:\t{}'.format(config['data']['adj_mx_threshold']))
    if config['train']['log_dir'] is not None:
        print('|--- LOG_DIR:\t{}'.format(config['train']['log_dir']))



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

    else:
        print('----------------------- TEST -----------------------')
        print('|--- MODEL_FILENAME:\t{}'.format(config['train']['model_filename']))
        print('|--- RUN_TIMES:\t{}'.format(config['test']['run_times']))
        print('|--- RESULTS_PATH:\t{}'.format(config['test']['results_path']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


# def get_results(data):
#     print('|--- Test ARIMA')
#     if Config.DATA_NAME == Config.DATA_SETS[0]:
#         day_size = Config.ABILENE_DAY_SIZE
#     else:
#         day_size = Config.GEANT_DAY_SIZE
#
#     data[data <= 0] = 0.1
#
#     train_data2d, test_data2d = prepare_train_test_2d(data=data, day_size=day_size)
#
#     if Config.DATA_NAME == Config.DATA_SETS[0]:
#         print('|--- Remove last 3 days in test_set.')
#         test_data2d = test_data2d[0:-day_size * 3]
#
#     # Data normalization
#     scaler = data_scalling(train_data2d)
#
#     test_data_normalized2d = scaler.transform(test_data2d)
#
#     _, _, y_true = prepare_test_set_last_5days(test_data2d, test_data_normalized2d)
#
#     results_path = Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
#                                                                Config.ALG, Config.TAG, Config.SCALER)
#     results_processing(y_true, Config.ARIMA_TESTING_TIME, results_path)


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

    print_dgc_lstm_info(args.mode, config)
    if args.mode == 'train':
        train_dgc_lstm(config)
    elif args.mode == 'evaluate' or args.mode == 'evaluation':
        evaluate_dgc_lstm(config)
    else:
        test_dgc_lstm(config)
    # get_results(data)
