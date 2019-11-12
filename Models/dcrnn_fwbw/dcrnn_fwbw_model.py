from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.layers import Dense, Dropout, TimeDistributed
from tensorflow.contrib import legacy_seq2seq

from Models.dcrnn.dcrnn_cell import DCGRUCell


class DCRNNModel(object):
    def __init__(self, is_training, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mse = None
        self._train_op = None

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))

        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')

        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels_fw = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, 1), name='labels_fw')
        # self._labels_bw = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, 1), name='labels_bw')
        # self._enc_labels_fw = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, 1),
        #                                      name='enc_labels_fw')
        self._enc_labels_bw = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, 1),
                                             name='enc_labels_bw')

        # GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * input_dim))
        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * output_dim))

        cell = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        cell_with_projection = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         num_proj=output_dim, filter_type=filter_type)

        encoding_cells_fw = [cell] * num_rnn_layers
        decoding_cells_fw = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        encoding_cells_fw = tf.contrib.rnn.MultiRNNCell(encoding_cells_fw, state_is_tuple=True)
        decoding_cells_fw = tf.contrib.rnn.MultiRNNCell(decoding_cells_fw, state_is_tuple=True)

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            inputs_fw = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            labels_fw = tf.unstack(
                tf.reshape(self._labels_fw[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            labels_fw.insert(0, GO_SYMBOL)

            def _loop_function_fw(prev_fw, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        result_fw = tf.cond(tf.less(c, threshold), lambda: labels_fw[i], lambda: prev_fw)
                    else:
                        result_fw = labels_fw[i]
                else:
                    # Return the prediction of the model in testing.
                    result_fw = prev_fw
                return result_fw

            # enc_outputs_fw, enc_state_fw = tf.contrib.rnn.static_rnn(encoding_cells_fw, inputs_fw, dtype=tf.float32)
            _, enc_state_fw = tf.contrib.rnn.static_rnn(encoding_cells_fw, inputs_fw, dtype=tf.float32)

            # encoder_layers = RNN(encoding_cells, return_state=True, return_sequences=True)
            # _, enc_state = encoder_layers(inputs)
            outputs_fw, final_state_fw = legacy_seq2seq.rnn_decoder(labels_fw, enc_state_fw, decoding_cells_fw,
                                                                    loop_function=_loop_function_fw)

        # Project the output to output_dim.
        outputs_fw = tf.stack(outputs_fw[:-1], axis=1)
        self._outputs_fw = tf.reshape(outputs_fw, (batch_size, horizon, num_nodes, output_dim), name='outputs_fw')

        # construct backward network
        encoding_cells_bw = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        encoding_cells_bw = tf.contrib.rnn.MultiRNNCell(encoding_cells_bw, state_is_tuple=True)

        with tf.variable_scope('DCRNN_SEQ_BW'):
            inputs_bw = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)

            enc_outputs_bw, enc_state_bw = tf.contrib.rnn.static_rnn(encoding_cells_bw, inputs_bw, dtype=tf.float32)

        enc_outputs_bw = tf.stack(enc_outputs_bw, axis=1)

        enc_outputs_bw = tf.reshape(enc_outputs_bw, (batch_size, seq_len, num_nodes, output_dim))

        enc_outputs_bw = tf.concat([enc_outputs_bw, self._inputs], axis=3)
        enc_outputs_bw = tf.reshape(enc_outputs_bw, (batch_size, seq_len, num_nodes * (output_dim + input_dim)))

        enc_outputs_bw = Dropout(0.5,
                                 batch_input_shape=(batch_size, seq_len, num_nodes * (output_dim + input_dim)))(
            enc_outputs_bw)
        enc_outputs_bw = TimeDistributed(Dense(512),
                                         batch_input_shape=(batch_size, seq_len, num_nodes * (output_dim + input_dim)))(
            enc_outputs_bw)
        enc_outputs_bw = Dropout(0.5, batch_input_shape=(batch_size, seq_len, 512))(enc_outputs_bw)
        enc_outputs_bw = TimeDistributed(Dense(num_nodes), batch_input_shape=(batch_size, seq_len, 512))(enc_outputs_bw)
        self._enc_outputs_bw = tf.reshape(enc_outputs_bw, (batch_size, seq_len, num_nodes, output_dim),
                                          name='enc_outputs_bw')

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
    def labels_fw(self):
        return self._labels_fw

    # @property
    # def labels_bw(self):
    #     return self._labels_bw
    #
    # @property
    # def enc_labels_fw(self):
    #     return self._enc_labels_fw

    @property
    def enc_labels_bw(self):
        return self._enc_labels_bw

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
    def outputs_fw(self):
        return self._outputs_fw

    # @property
    # def enc_outputs_fw(self):
    #     return self._enc_outputs_fw

    # @property
    # def outputs_bw(self):
    #     return self._outputs_bw

    @property
    def enc_outputs_bw(self):
        return self._enc_outputs_bw
