from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from Models.gat_lstm.gat_lstm_cell import attn_head


# from tensorflow.contrib import legacy_seq2seq


class GATLSTMModel(object):
    def __init__(self, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mse = None
        self._train_op = None

        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.batch_size = batch_size
        self.n_heads = model_kwargs.get('n_heads')
        self.hid_units = model_kwargs.get('hid_units')
        self.activation = tf.nn.elu
        self.residual = model_kwargs.get('residual')

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
        with tf.name_scope('input'):
            self._inputs = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_nodes, self.input_dim))
            self.bias_in = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_nodes, self.num_nodes))
            self._labels = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.num_nodes, self.output_dim))
            self.msk_in = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.num_nodes))
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            self.is_train = tf.placeholder(dtype=tf.bool, shape=())

    def _build_model(self):
        attns = []
        for _ in range(self.n_heads[0]):
            attns.append(attn_head(self._inputs, bias_mat=self.msk_in,
                                   out_sz=self.hid_units, activation=self.activation,
                                   in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False))

        h_1 = tf.concat(attns, axis=-1)  # the outputs for first layer are con catenated

        # attention for other layers
        for i in range(1, len(self.hid_units)):
            # h_1 then is used as input for other layers
            h_old = h_1
            attns = []
            for _ in range(self.n_heads[i]):
                attns.append(attn_head(h_1, bias_mat=self.msk_in,
                                       out_sz=self.hid_units[i], activation=self.activation,
                                       in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual))
            h_1 = tf.concat(attns, axis=-1)

        # Calculate output by applying averaging attention layer on the final layer of the network
        # n_heads[-1]: the number of head attention applied for the last layer
        out = []
        for i in range(self.n_heads[-1]):
            out.append(attn_head(h_1, bias_mat=self.msk_in,
                                 out_sz=self.output_dim, activation=lambda x: x,
                                 in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False))
        self._outputs = tf.add_n(out) / self.n_heads[-1]  # Averaging to obtain the output
        self._merged = tf.summary.merge_all()
