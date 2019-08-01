from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from Models.dcrnn_model import DCRNNModel
from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import masked_mse_loss


class DCRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)

        self._mon_ratio = float(self._kwargs.get('mon_ratio'))

        # Model's args
        self._seq_len = int(self._model_kwargs.get('seq_len'))
        self._horizon = int(self._model_kwargs.get('horizon'))
        self._input_dim = int(self._model_kwargs.get('input_dim'))
        self._nodes = int(self._model_kwargs.get('num_nodes'))

        # Test's args
        self._flow_selection = self._test_kwargs.get('flow_selection')
        self._test_size = self._test_kwargs.get('test_size')

        # Data preparation
        self._data = utils.load_dataset_dcrnn(seq_len=self._model_kwargs.get('seq_len'),
                                              horizon=self._model_kwargs.get('horizon'),
                                              input_dim=self._model_kwargs.get('input_dim'),
                                              mon_ratio=self._mon_ratio,
                                              test_size=self._test_size,
                                              **self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                self._train_model = DCRNNModel(is_training=True, scaler=scaler,
                                               batch_size=self._data_kwargs['batch_size'],
                                               adj_mx=self._data['adj_mx'], **self._model_kwargs)

        with tf.name_scope('Evaluation'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._eval_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=self._data_kwargs['eval_batch_size'],
                                              adj_mx=self._data['adj_mx'], **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._test_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=1,
                                              adj_mx=self._data['adj_mx'], **self._model_kwargs)

        # Learning rate.
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Configure optimizer
        optimizer_name = self._train_kwargs.get('optimizer', 'adam').lower()
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr, )
        elif optimizer_name == 'amsgrad':
            optimizer = AMSGrad(self._lr, epsilon=epsilon)

        # Calculate loss
        output_dim = self._model_kwargs.get('output_dim')
        preds = self._train_model.outputs
        labels = self._train_model.labels[..., :output_dim]

        null_val = 0.
        self._loss_fn = masked_mse_loss(scaler, null_val)
        # self._loss_fn = masked_mae_loss(scaler, null_val)
        self._train_loss = self._loss_fn(preds=preds, labels=labels)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)
        self._epoch = 0
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def run_epoch_generator(self, sess, model, data_generator, return_output=False, training=False, writer=None):
        losses = []
        mses = []
        outputs = []
        output_dim = self._model_kwargs.get('output_dim')
        preds = model.outputs
        labels = model.labels[..., :output_dim]
        loss = self._loss_fn(preds=preds, labels=labels)
        fetches = {
            'loss': loss,
            'mse': loss,
            'global_step': tf.train.get_or_create_global_step()
        }
        if training:
            fetches.update({
                'train_op': self._train_op
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        for _, (x, y) in enumerate(data_generator):
            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            mses.append(vals['mse'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])

        results = {
            'loss': np.mean(losses),
            'mse': np.mean(mses)
        }
        if return_output:
            results['outputs'] = outputs
        return results

    def _prepare_input_online_prediction(self, ground_truth, data, m_indicator):

        x = np.zeros(shape=(1, self._seq_len, self._nodes, self._input_dim))
        y = np.zeros(shape=(1, self._horizon, self._nodes, 1))

        x[0, :, :, 0] = data[-self._seq_len:]
        x[0, :, :, 1] = m_indicator[-self._seq_len:]

        y[0, :, :, 0] = ground_truth[-self._horizon:]

        return x, y

    @staticmethod
    def calculate_consecutive_loss(labels):
        """

        :param labels: shape(#time-steps, #n_flows)
        :return: consecutive_losses: shape(#n_flows)
        """

        consecutive_losses = []
        for flow_id in range(labels.shape[1]):
            flows_labels = labels[:, flow_id]
            if flows_labels[-1] == 1:
                consecutive_losses.append(1)
            else:
                measured_idx = np.argwhere(flows_labels == 1)
                if measured_idx.size == 0:
                    consecutive_losses.append(labels.shape[0])
                else:
                    consecutive_losses.append(labels.shape[0] - measured_idx[-1][0])

        consecutive_losses = np.asarray(consecutive_losses)
        return consecutive_losses

    def set_measured_flow_fairness(self, labels):
        """

        :param rnn_input: shape(#n_flows, #time-steps)
        :param labels: shape(n_flows, #time-steps)
        :return:
        """

        cl = self.calculate_consecutive_loss(labels).astype(float)

        w = 1 / cl

        sampling = np.zeros(shape=self._nodes)
        m = int(self._mon_ratio * self._nodes)

        w = w.flatten()
        sorted_idx_w = np.argsort(w)
        sampling[sorted_idx_w[:m]] = 1

        return sampling

    def run_tm_prediction(self, sess, model, data, writer=None):

        init_data = data[0]
        test_data = data[1]

        # Initialize traffic matrix data
        tm_pred = np.zeros(shape=(init_data.shape[0] + test_data.shape[0] - self._horizon, test_data.shape[1]))
        tm_pred[0:init_data.shape[0]] = init_data

        # Initialize predicted_traffic matrice
        predicted_tm = np.zeros(shape=(test_data.shape[0] - self._horizon, self._horizon, test_data.shape[1]))

        # Initialize measurement matrix
        m_indicator = np.zeros(shape=(init_data.shape[0] + test_data.shape[0] - self._horizon, test_data.shape[1]))
        m_indicator[0:init_data.shape[0]] = np.ones(shape=init_data.shape)

        losses = []
        mses = []
        outputs = []
        output_dim = self._model_kwargs.get('output_dim')
        preds = model.outputs
        labels = model.labels[..., :output_dim]
        loss = self._loss_fn(preds=preds, labels=labels)
        fetches = {
            'loss': loss,
            'mse': loss,
            'global_step': tf.train.get_or_create_global_step()
        }

        fetches.update({
            'outputs': model.outputs
        })

        for ts in tqdm(range(test_data.shape[0] - self._horizon)):

            x, y = self._prepare_input_online_prediction(ground_truth=test_data[ts + 1:ts + 1 + self._horizon],
                                                         data=tm_pred[ts:ts + self._seq_len],
                                                         m_indicator=m_indicator[ts:ts + self._seq_len])

            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            mses.append(vals['mse'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])

            pred = vals['outputs']
            predicted_tm[ts] = pred[0, :, :, 0]
            pred = pred[0, 0, :, 0]

            if self._flow_selection == 'Random':
                sampling = np.random.choice([1.0, 0.0], size=(test_data.shape[1]),
                                            p=[self._mon_ratio, 1 - self._mon_ratio])
            else:
                sampling = self.set_measured_flow_fairness(labels=m_indicator[ts: ts + self._seq_len])

            m_indicator[ts + self._seq_len] = sampling
            # invert of sampling: for choosing value from the original data
            inv_sampling = 1.0 - sampling
            pred_input = pred * inv_sampling

            ground_true = test_data[ts]

            measured_input = ground_true * sampling

            # Merge value from pred_input and measured_input
            new_input = pred_input + measured_input

            # Concatenating new_input into current rnn_input
            tm_pred[ts + self._seq_len] = new_input

            outputs.append(vals['outputs'])

        results = {'loss': np.mean(losses), 'mse': np.mean(mses), 'outputs': outputs, 'predicted_tm': predicted_tm}
        return results

    def get_lr(self, sess):
        return sess.run(self._lr).item()

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def test(self, sess, **kwargs):
        kwargs.update(self._test_kwargs)
        return self._test(sess, **kwargs)

    def _train(self, sess, base_lr, epoch, steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = train_kwargs.get('model_filename')
        continue_train = train_kwargs.get('continue_train')
        if continue_train is True and model_filename is not None:
            saver.restore(sess, model_filename)
            self._epoch = epoch + 1
        else:
            sess.run(tf.global_variables_initializer())
        self._logger.info('Start training ...')

        while self._epoch <= epochs:
            # Learning rate schedule.
            new_lr = max(min_learning_rate, base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, self._train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     training=True,
                                                     writer=self._writer)
            train_loss, train_mse = train_results['loss'], train_results['mse']
            # if train_loss > 1e5:
            #     self._logger.warning('Gradient explosion detected. Ending...')
            #     break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = self.run_epoch_generator(sess, self._test_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   training=False)
            val_loss, val_mse = val_results['loss'].item(), val_results['mse'].item()

            utils.add_simple_summary(self._writer,
                                     ['loss/train_loss', 'metric/train_mse', 'loss/val_loss', 'metric/val_mse'],
                                     [train_loss, train_mse, val_loss, val_mse], global_step=global_step)
            end_time = time.time()
            message = 'Epoch [{}/{}] ({}) train_mse: {:.4f}, val_mse: {:.4f} lr:{:.6f} {:.1f}s'.format(
                self._epoch, epochs, global_step, train_mse, val_mse, new_lr, (end_time - start_time))
            self._logger.info(message)
            if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
                self.evaluate(sess)
            if val_loss <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_loss)
                self._logger.info(
                    'Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warning('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_mse)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    def prepare_test_set(self):

        day_size = self._data_kwargs.get('day_size')

        idx = self._data['test_data'].shape[0] - day_size * self._test_size - 10

        test_data_normalize = np.copy(self._data['test_data_norm'][idx:idx + day_size * self._test_size])
        init_data_normalize = np.copy(self._data['test_data_norm'][idx - self._seq_len: idx])
        test_data = self._data['test_data'][idx:idx + day_size * self._test_size]

        return test_data_normalize, init_data_normalize, test_data

    def _test(self, sess, **kwargs):

        global_step = sess.run(tf.train.get_or_create_global_step())
        mape, r2_score, rmse = [], [], []
        mape_ims, r2_score_ims, rmse_ims = [], [], []

        for i in range(self._test_kwargs.get('run_times')):
            print('|--- Running time: {}'.format(i))
            test_data_normalize, init_data_normalize, test_data = self.prepare_test_set()

            test_results = self.run_tm_prediction(sess,
                                                  model=self._test_model,
                                                  data=(init_data_normalize, test_data_normalize)
                                                  )

            # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
            test_loss, y_preds, predicted_tm = test_results['loss'], test_results['outputs'], test_results[
                'predicted_tm']
            utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)

            y_preds = np.concatenate(y_preds, axis=0)
            scaler = self._data['scaler']
            predictions = []
            y_truths = []
            for horizon_i in range(self._data['y_test'].shape[1]):
                y_truth = scaler.inverse_transform(self._data['y_test'][:, horizon_i, :, 0])
                y_truths.append(y_truth)

                y_pred = scaler.inverse_transform(y_preds[:y_truth.shape[0], horizon_i, :, 0])
                predictions.append(y_pred)

                mse = metrics.masked_mse_np(y_pred, y_truth, null_val=0)
                mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
                rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
                self._logger.info(
                    "Horizon {:02d}, MSE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                        horizon_i + 1, mse, mape, rmse
                    )
                )
                utils.add_simple_summary(self._writer,
                                         ['%s_%d' % (item, horizon_i + 1) for item in
                                          ['metric/rmse', 'metric/mape', 'metric/mse']],
                                         [rmse, mape, mse],
                                         global_step=global_step)
            outputs = {
                'predictions': predictions,
                'groundtruth': y_truths
            }
        return

    def evaluate(self, sess, **kwargs):
        global_step = sess.run(tf.train.get_or_create_global_step())
        test_results = self.run_epoch_generator(sess, self._eval_model,
                                                self._data['eval_loader'].get_iterator(),
                                                return_output=True,
                                                training=False)

        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = test_results['loss'], test_results['outputs']
        utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)

        y_preds = np.concatenate(y_preds, axis=0)
        scaler = self._data['scaler']
        predictions = []
        y_truths = []
        for horizon_i in range(self._data['y_eval'].shape[1]):
            y_truth = scaler.inverse_transform(self._data['y_eval'][:, horizon_i, :, 0])
            y_truths.append(y_truth)

            y_pred = scaler.inverse_transform(y_preds[:y_truth.shape[0], horizon_i, :, 0])
            predictions.append(y_pred)

            mse = metrics.masked_mse_np(y_pred, y_truth, null_val=0)
            mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
            rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
            self._logger.info(
                "Horizon {:02d}, MSE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                    horizon_i + 1, mse, mape, rmse
                )
            )
            utils.add_simple_summary(self._writer,
                                     ['%s_%d' % (item, horizon_i + 1) for item in
                                      ['metric/rmse', 'metric/mape', 'metric/mse']],
                                     [rmse, mape, mse],
                                     global_step=global_step)
        outputs = {
            'predictions': predictions,
            'groundtruth': y_truths
        }
        return outputs

    def load(self, sess, model_filename):
        """
        Restore from saved model.
        :param sess:
        :param model_filename:
        :return:
        """
        self._saver.restore(sess, model_filename)

    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        global_step = sess.run(tf.train.get_or_create_global_step()).item()
        prefix = os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self._saver.save(sess, prefix, global_step=global_step,
                                                             write_meta_graph=False)
        config_filename = 'config_{}.yaml'.format(self._epoch)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']
