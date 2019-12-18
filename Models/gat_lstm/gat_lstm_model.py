from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from lib.metrics import masked_mae_tf, masked_mse_tf


def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    """

    :param seq: inputs
    :param out_sz: output's size
    :param bias_mat: bias matrix
    :param activation: activation_fn
    :param in_drop: input_drop
    :param coef_drop:
    :param residual:
    :return:
    """
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        print('seq_fts shape: ', seq_fts.shape)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        print('f_1 shape: ', f_1.shape)
        print('f_2 shape: ', tf.transpose(f_2, [0, 2, 1]).shape)

        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        print('logits shape: ', logits.shape)

        # alpha_ij
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        print('coefs shape: ', coefs.shape)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


class GATLSTMModel(object):
    def __init__(self, batch_size, scaler, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mse = None
        self._train_op = None

        self.num_nodes = int(model_kwargs.get('num_nodes', 144))
        self.input_dim = int(model_kwargs.get('input_dim', 3))
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.batch_size = batch_size
        self.n_heads = model_kwargs.get('n_heads')
        self.hid_units = model_kwargs.get('hid_units')
        self.activation = tf.nn.elu
        self.residual = model_kwargs.get('residual')
        self.max_grad_norm = model_kwargs.get('max_grad_norm')
        self.classif_loss = model_kwargs.get('classif_loss')
        self.learning_rate = model_kwargs.get('learning_rate')
        self.optimizer = model_kwargs.get('optimizer')

        self._build_placeholder()
        self._build_model()
        self._build_steps()
        self._build_optim()

        show_all_variables()

    def _build_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_nodes, self.input_dim))
        self.adj_mx = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_nodes, self.num_nodes))
        self.labels = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_nodes, self.output_dim))
        self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())

        self.model_step = tf.Variable(
            0, name='model_step', trainable=False)

    def _build_model(self, reuse=None):
        with tf.variable_scope("gatlstm_model", reuse=reuse) as sc:

            attns = []
            for _ in range(self.n_heads[0]):
                attns.append(attn_head(self.inputs, bias_mat=self.adj_mx,
                                       out_sz=self.hid_units[0], activation=self.activation,
                                       in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False))

            h_1 = tf.concat(attns, axis=-1)  # the outputs for first layer are con catenated

            # attention for other layers
            for i in range(1, len(self.hid_units)):
                # h_1 then is used as input for other layers
                h_old = h_1
                attns = []
                for _ in range(self.n_heads[i]):
                    attns.append(attn_head(h_1, bias_mat=self.adj_mx,
                                           out_sz=self.hid_units[i], activation=self.activation,
                                           in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual))
                h_1 = tf.concat(attns, axis=-1)

            # Calculate output by applying averaging attention layer on the final layer of the network
            # n_heads[-1]: the number of head attention applied for the last layer
            out = []
            for i in range(self.n_heads[-1]):
                out.append(attn_head(h_1, bias_mat=self.adj_mx,
                                     out_sz=self.output_dim, activation=lambda x: x,
                                     in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False))
            self.outputs = tf.add_n(out) / self.n_heads[-1]  # Averaging to obtain the output
            print('output shape: ', self.outputs.get_shape())
            self.merged = tf.summary.merge_all()

            self.model_vars = tf.contrib.framework.get_variables(
                sc, collection=tf.GraphKeys.TRAINABLE_VARIABLES)

        self._build_loss()

    def _build_loss(self):

        if self.classif_loss == 'mae':
            loss_batchmean = masked_mae_tf(self.outputs, self.labels)
        elif self.classif_loss == 'mse':
            loss_batchmean = masked_mse_tf(self.outputs, self.labels)
        else:
            raise ValueError(
                "Unsupported loss type {}".format(
                    self.classif_loss))
        with tf.name_scope("losses"):
            self.loss = loss_batchmean

    def _build_steps(self):
        def run(sess, feed_dict, fetch,
                summary_op, summary_writer, output_op=None, output_img=None):
            if summary_writer is not None:
                fetch['summary'] = summary_op
            if output_op is not None:
                fetch['output'] = output_op

            result = sess.run(fetch, feed_dict=feed_dict)
            if "summary" in result.keys() and "step" in result.keys():
                summary_writer.add_summary(result['summary'], result['step'])
                summary_writer.flush()
            return result

        def train(sess, feed_dict, summary_writer=None,
                  with_output=False):
            fetch = {'loss': self.loss,
                     'optim': self.model_optim,  # ?
                     }
            return run(sess, feed_dict, fetch,
                       self.merged, summary_writer,
                       output_op=self.outputs if with_output else None, )

        def test(sess, feed_dict, summary_writer=None, with_output=False):
            fetch = {}
            return run(sess, feed_dict, fetch,
                       self.merged, summary_writer,
                       output_op=self.outputs if with_output else None, )

        self.train = train
        self.test = test

    def _build_optim(self):
        def minimize(loss, step, var_list, learning_rate, optimizer):
            if optimizer == "sgd":
                optim = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer == "adam":
                optim = tf.train.AdamOptimizer(learning_rate)
            elif optimizer == "rmsprop":
                optim = tf.train.RMSPropOptimizer(learning_rate)
            else:
                raise Exception("[!] Unkown optimizer: {}".format(
                    optimizer))
            ## Gradient clipping ##
            if self.max_grad_norm is not None:
                grads_and_vars = optim.compute_gradients(
                    loss, var_list=var_list)
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None and var in var_list:
                        grad = tf.clip_by_norm(grad, self.max_grad_norm)
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}".format(
                                var.name))
                        new_grads_and_vars.append((grad, var))
                return optim.apply_gradients(new_grads_and_vars, global_step=step)
            else:
                grads_and_vars = optim.compute_gradients(
                    loss, var_list=var_list)
                return optim.apply_gradients(grads_and_vars,
                                             global_step=step)

        # optim #
        self.model_optim = minimize(
            self.loss,
            self.model_step,
            self.model_vars,
            self.learning_rate,
            self.optimizer)
