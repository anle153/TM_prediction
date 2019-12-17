from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from Models.gat_lstm.gat_lstm_cell import GATLSTMCell
from keras.layers import LSTM


# from tensorflow.contrib import legacy_seq2seq


class GATLSTMModel(object):
    def __init__(self, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mse = None
        self._train_op = None

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))

        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seq_len = seq_len
        s
        cell = GATLSTMCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                           filter_type=filter_type)
        cell_with_projection = GATLSTMCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step,
                                           num_nodes=num_nodes,
                                           num_proj=output_dim, filter_type=filter_type)

        encoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)

        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            # inputs = tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim))
            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            labels.insert(0, GO_SYMBOL)

            outputs, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)

        # Project the output to output_dim.
        self._outputs = tf.reshape(outputs[-1], (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._merged = tf.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mse(self):
        return self._mse

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs

    def _build_placeholder(self):
        self._lstm_inputs = tf.placeholder(tf.float32,
                                           shape=(self.batch_size, self.seq_len, self.num_nodes, self.input_dim),
                                           name='lstm_inputs')

        self._gat_input = tf.placeholder(tf.float32,
                                         shape=(self.batch_size, self.num_nodes, self.num_nodes, self.gat_input_dim),
                                         name='gat_input')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(self.batch_size, horizon, num_nodes, 1), name='labels')

        lstm_net = LSTM(units=self.lstm_units, return_sequences=False)
