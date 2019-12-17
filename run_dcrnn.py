import argparse
import os
import sys

import tensorflow as tf
import yaml

from Models.dcrnn.dcrnn_supervisor import DCRNNSupervisor


def print_dcrnn_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(mode))
    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- DATA:\t{}'.format(config['data']['data_name']))
    print('|--- GPU:\t{}'.format(config['gpu']))
    print('|--- GENERATE_DATA:\t{}'.format(config['data']['generate_data']))

    print('|--- MON_RATIO:\t{}'.format(config['mon_ratio']))
    print('|--- BASED_DIR:\t{}'.format(config['base_dir']))
    print('|--- SCALER:\t{}'.format(config['scaler']))

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

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


def train_dcrnn(config, gpu):
    print('|-- Run model training dcrnn.')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    with tf.device('/device:GPU:{}'.format(gpu)):
        with tf.Session(config=tf_config) as sess:
            model = DCRNNSupervisor(is_training=True, **config)
            sess.run(tf.global_variables_initializer())
            try:
                if config['train']['continue_train']:
                    model.load(sess, config['train']['model_filename'])
            except KeyError:
                print('No saved model found!')
            model.train(sess)


def test_dcrnn(config, gpu):
    print('|-- Run model testing dcrnn.')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    with tf.device('/device:GPU:{}'.format(gpu)):
        with tf.Session(config=tf_config) as sess:
            model = DCRNNSupervisor(is_training=False, **config)
            sess.run(tf.global_variables_initializer())
            model.load(sess, config['train']['model_filename'])
            model.test(sess)
            # np.savez_compressed(os.path.join(HOME_PATH, config['test']['results_path']), **outputs)
            #
            # print('Predictions saved as {}.'.format(os.path.join(HOME_PATH, config['test']['results_path']) + '.npz'))


def evaluate_dcrnn(config):
    print('|-- Run model testing dcrnn.')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = DCRNNSupervisor(is_training=False, **config)
        model.load(sess, config['train']['model_filename'])
        outputs = model.evaluate(sess)


if __name__ == '__main__':

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU device.')

    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    print_dcrnn_info(args.mode, config)
    if args.mode == 'train':
        train_dcrnn(config, args.gpu)
    elif args.mode == 'evaluate' or args.mode == 'evaluation':
        evaluate_dcrnn(config)
    else:
        test_dcrnn(config, args.gpu)
    # get_results(data)
