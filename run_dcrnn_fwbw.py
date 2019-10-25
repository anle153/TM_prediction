import argparse
import os
import sys

import tensorflow as tf
import yaml

from Models.dcrnn_fwbw.dcrnn_fwbw_supervisor import DCRNNSupervisor


def print_dcrnn_fwbw_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(mode))
    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- DATA:\t{}'.format(config['data']['data_name']))
    print('|--- GPU:\t{}'.format(config['gpu']))
    print('|--- GENERATE_DATA:\t{}'.format(config['data']['generate_data']))

    print('|--- MON_RATIO:\t{}'.format(config['mon_ratio']))
    print('|--- BASED_DIR:\t{}'.format(config['base_dir']))

    print('|--- ADJ_METHOD:\t{}'.format(config['data']['adj_method']))
    print('|--- ADJ_POS_THRES:\t{}'.format(config['data']['pos_thres']))
    print('|--- ADJ_NEG_THRES:\t{}'.format(config['data']['neg_thres']))

    print('----------------------- MODEL -----------------------')
    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- NUM_RNN_LAYERS:\t{}'.format(config['model']['num_rnn_layers']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}\n'.format(config['model']['rnn_units']))

    if mode == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- LEARNING_RATE:\t{}'.format(config['train']['base_lr']))
        print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
        print('|--- EPSILON:\t{}'.format(config['train']['epsilon']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))
        print('|--- BATCH:\t{}'.format(config['data']['batch_size']))
        print('|--- CONTINUE_TRAIN:\t{}\n'.format(config['train']['continue_train']))

    else:
        print('----------------------- TEST -----------------------')
        print('|--- MODEL_FILENAME:\t{}'.format(config['train']['model_filename']))
        print('|--- RUN_TIMES:\t{}\n'.format(config['test']['run_times']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


def train_dcrnn_fwbw(config):
    print('|-- Run model training dgc_lstm.')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        model = DCRNNSupervisor(**config)
        model.train(sess)


def test_dcrnn_fwbw(config_fw, config_bw):
    print('|-- Run model testing dgc_lstm.')

    model = DCRNNSupervisor(**config_fw)
    model.test(config_fw, config_bw)


def evaluate_dcrnn_fwbw(config_fw, config_bw):
    print('|-- Run model testing dgc_lstm.')
    model = DCRNNSupervisor(**config_fw)
    model.evaluate(config_fw, config_bw)


if __name__ == '__main__':

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--config-bw', type=str,
                        help='Config file for pretrained backward model.')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()

    with open(args.config) as f:
        config_fw = yaml.load(f)

    print_dcrnn_fwbw_info(args.mode, config_fw)
    if args.mode == 'train':
        train_dcrnn_fwbw(config_fw)
    elif args.mode == 'evaluate' or args.mode == 'evaluation':
        with open(args.config_bw) as b_bw:
            config_bw = yaml.load(b_bw)
        evaluate_dcrnn_fwbw(config_fw, config_bw)
    else:
        with open(args.config_bw) as b_bw:
            config_bw = yaml.load(b_bw)
        test_dcrnn_fwbw(config_fw, config_bw)
