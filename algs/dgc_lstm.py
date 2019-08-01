import os

import tensorflow as tf

from Models.dcrnn_supervisor import DCRNNSupervisor

HOME_PATH = os.path.expanduser('~/TM_prediction')

DATA_SETS = ['Abilene', 'Geant']
DATA_NAME = DATA_SETS[0]
ABILENE_DAY_SIZE = 288
GEANT_DAY_SIZE = 288
CONFIG_PATH = os.path.join(HOME_PATH, 'Config')
CONFIG_FILE = 'config_dgclstm.yaml'
RESULTS_PATH = os.path.join(HOME_PATH, 'results')


def train_dgc_lstm(config):
    print('|-- Run model training dgc_lstm.')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        model = DCRNNSupervisor(**config)
        model.train(sess)


def test_dgc_lstm(config):
    print('|-- Run model testing dgc_lstm.')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = DCRNNSupervisor(**config)
        model.load(sess, config['train']['model_filename'])
        outputs = model.test(sess)
        # np.savez_compressed(os.path.join(HOME_PATH, config['test']['results_path']), **outputs)
        #
        # print('Predictions saved as {}.'.format(os.path.join(HOME_PATH, config['test']['results_path']) + '.npz'))


def evaluate_dgc_lstm(config):
    print('|-- Run model testing dgc_lstm.')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = DCRNNSupervisor(**config)
        model.load(sess, config['train']['model_filename'])
        outputs = model.evaluate(sess)
