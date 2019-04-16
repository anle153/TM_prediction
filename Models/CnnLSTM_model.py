import fnmatch
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from Utils.Configurations import *
from keras.layers import *
from keras.layers import LSTM, Dense, Dropout, Activation, TimeDistributed
from keras.models import Sequential
from keras.models import model_from_json


class CnnLSTM():

    def __init__(self, n_timsteps, height, weight, depth, saving_path):
        """

        :param n_timsteps:
        :param n_features:
        :param cnn_input_shape: the shape of the traffic matrix [k x k x d] (d = 2: since we have the measured_matrix)
        """
        self.n_timsteps = n_timsteps
        self.height = height
        self.weight = weight
        self.depth = depth
        self.saving_path = saving_path
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        cnn_input_shape = (self.height, self.weight, self.depth)

        with tf.variable_scope('cnnlstm'):
            self.input = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imagesTM = tf.reshape(self.input, shape=[-1, self.height, self.weight, self.depth])

            self.conv1 = slim.conv2d(inputs=self.imagesTM, num_outputs=16,
                                     kernel_size=[3, 3], stride=[1, 1], padding='SAME')

            self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=32,
                                     kernel_size=[3, 3], stride=[1, 1], padding="SAME")

            self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64,
                                     kernel_size=[5, 5], stride=[2, 2], padding="SAME")

            hidden = slim.fully_connected(slim.flatten(self.conv3), 256, activation_fn=tf.nn.relu)

            # recurrent neural network

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        self.cnn = Sequential()
        self.cnn.add(ZeroPadding2D((1, 1), input_shape=cnn_input_shape))
        self.cnn.add(Conv2D(filters=32,
                            kernel_size=[3, 3],
                            strides=[1, 1],
                            padding='same',
                            activation='relu',
                            name='conv'))
        self.cnn.add(Conv2D(filters=64,
                            kernel_size=[5, 5],
                            strides=[2, 2],
                            padding='same',
                            activation='relu',
                            name='conv2'))
        self.cnn.add(Activation('relu'))
        self.cnn.add(MaxPooling2D(pool_size=[2, 2],
                                  strides=[2, 2]))
        self.cnn.add(Flatten())
        self.cnn.add(Dense(64, activation='relu'))
        self.cnn.add(Dropout(0.5))
        self.cnn.add(Dense(64, activation='relu'))

        self.model = Sequential()
        self.model.add(TimeDistributed(self.cnn,
                                       input_shape=(self.n_timsteps, self.height, self.weight, self.depth)))
        self.model.add(LSTM(LSTM_HIDDEN_DIM, input_shape=(self.n_timsteps, 32), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(144)))

    def load_model_from_disk(self, model_json_file='model.json', model_weight_file='model.h5'):
        """
        Load RNN model from disk
        :param path_to_file:
        :param model_json_file: model is stored in json format
        :param model_weight_file: model weight
        :return:
        """
        assert os.path.isfile(self.saving_path + model_json_file) & os.path.isfile(self.saving_path + model_weight_file)

        json_file = open(self.saving_path + model_json_file, 'r')
        model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(model_json)
        self.model.load_weights(self.saving_path + model_weight_file)
        self.saving_path = self.saving_path

        print('----> [CNN_LSTM-load_model_from_disk]--- Models has been loaded from %s' % (
                    self.saving_path + model_json_file))
        return True

    def load_model_from_check_point(self, _from_epoch=0, weights_file_type='h5'):

        if weights_file_type == 'h5':
            if os.path.exists(self.saving_path):

                list_weights_files = fnmatch.filter(os.listdir(self.saving_path), '*.h5')

                if len(list_weights_files) == 0:
                    print(
                        '----> [CNN_LSTM-load_model_from_check_point] --- Found no weights file at %s---' % self.saving_path)
                    return -1

                list_weights_files = sorted(list_weights_files, key=lambda x: int(x.split('-')[1]))
                weights_file_name = ''
                model_file_name = ''
                epoch = -1
                if _from_epoch:
                    for _weights_file_name in list_weights_files:
                        epoch = int(_weights_file_name.split('-')[1])
                        if _from_epoch == epoch:
                            weights_file_name = _weights_file_name
                            model_file_name = 'model-' + str(epoch) + '-.json'
                            break
                else:
                    # Get the last check point
                    weights_file_name = list_weights_files[-1]
                    epoch = int(weights_file_name.split('-')[1])
                    model_file_name = 'model-' + str(epoch) + '-.json'

                if self.load_model_from_disk(model_weight_file=weights_file_name, model_json_file=model_file_name):
                    return epoch
                else:
                    return -1
            else:
                print('----> [CNN_LSTM-load_model_from_check_point] --- Models saving path dose not exist')
                return -1
        else:
            if os.path.exists(self.saving_path):
                list_files = fnmatch.filter(os.listdir(self.saving_path), '*.hdf5')

                if len(list_files) == 0:
                    print(
                        '----> [CNN_LSTM-load_model_from_check_point] --- Found no weights file at %s---' % self.saving_path)
                    return -1

                list_files = sorted(list_files, key=lambda x: int(x.split('-')[1]))

                weights_file_name = ''
                epoch = -1
                if _from_epoch:
                    for _weights_file_name in list_files:
                        epoch = int(_weights_file_name.split('-')[1])
                        if _from_epoch == epoch:
                            weights_file_name = _weights_file_name
                            break
                else:
                    # Get the last check point
                    weights_file_name = list_files[-1]
                    epoch = int(weights_file_name.split('-')[1])

                if self.load_weights_model(path=self.saving_path, weight_file=weights_file_name):
                    return epoch
                else:
                    return -1
            else:
                print('----> [CNN_LSTM-load_model_from_check_point] --- Models saving path dose not exist')
                return -1
