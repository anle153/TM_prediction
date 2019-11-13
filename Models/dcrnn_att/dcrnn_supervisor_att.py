from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tqdm import tqdm

from Models.AbstractModel import AbstractModel
from Models.dcrnn_att.dcrnn_model_att import DCRNNModel
from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import masked_mse_loss


class DCRNNSupervisor(AbstractModel):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._data = utils.load_dataset_dcrnn_att(seq_len=self._model_kwargs.get('seq_len'),
                                                  horizon=self._model_kwargs.get('horizon'),
                                                  input_dim=self._model_kwargs.get('input_dim'),
                                                  mon_ratio=self._mon_ratio,
                                                  scaler_type=self._kwargs.get('scaler'),
                                                  **self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN_ATT', reuse=False):
                self._train_model = DCRNNModel(is_training=True, scaler=scaler,
                                               batch_size=self._data_kwargs['batch_size'],
                                               adj_mx=self._data['adj_mx'], **self._model_kwargs)

        with tf.name_scope('Val'):
            with tf.variable_scope('DCRNN_ATT', reuse=True):
                self._val_model = DCRNNModel(is_training=False, scaler=scaler,
                                             batch_size=self._data_kwargs['val_batch_size'],
                                             adj_mx=self._data['adj_mx'], **self._model_kwargs)

        with tf.name_scope('Eval'):
            with tf.variable_scope('DCRNN_ATT', reuse=True):
                self._eval_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=self._data_kwargs['eval_batch_size'],
                                              adj_mx=self._data['adj_mx'], **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN_ATT', reuse=True):
                self._test_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=self._data_kwargs['test_batch_size'],
                                              adj_mx=self._data['adj_mx'], **self._model_kwargs)

        # Learning rate.
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Configure optimizer
        optimizer_name = self._train_kwargs.get('optimizer', 'adam').lower()
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon, )
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

    def _prepare_input(self, ground_truth, data, m_indicator):

        x = np.zeros(shape=(self._seq_len, self._nodes, self._input_dim), dtype='float32')
        y = np.zeros(shape=(self._horizon, self._nodes), dtype='float32')

        x[:, :, 0] = data
        x[:, :, 1] = m_indicator

        y[:] = ground_truth
        y = np.expand_dims(y, axis=2)

        return np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)

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

        :param labels:
        :return:
        """

        cl = self.calculate_consecutive_loss(labels).astype(float)

        w = 1 / cl

        sampling = np.zeros(shape=self._nodes, dtype='float32')
        m = int(self._mon_ratio * self._nodes)

        w = w.flatten()
        sorted_idx_w = np.argsort(w)
        sampling[sorted_idx_w[:m]] = 1

        return sampling

    def _run_tm_prediction(self, sess, model, runId, writer=None):

        test_data_norm = self._data['test_data_norm']

        # Initialize traffic matrix data
        tm_pred = np.zeros(shape=(test_data_norm.shape[0] - self._horizon, self._nodes), dtype='float32')
        tm_pred[0:self._seq_len] = test_data_norm[:self._seq_len]

        # Initialize measurement matrix
        if self._flow_selection == 'Random':
            m_indicator = np.load(os.path.join(self._base_dir + '/random_m_indicator/m_indicator{}.npy'.format(runId)))
            m_indicator = np.concatenate([np.ones(shape=(self._seq_len, self._nodes)), m_indicator], axis=0)
        else:
            m_indicator = np.zeros(shape=(test_data_norm.shape[0] - self._horizon, self._nodes),
                                   dtype='float32')
            m_indicator[0:self._seq_len] = np.ones(shape=(self._seq_len, self._nodes))

        losses = []
        mses = []
        y_preds = []
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

        y_truths = []

        for ts in tqdm(range(test_data_norm.shape[0] - self._horizon - self._seq_len)):

            x, y = self._prepare_input(
                ground_truth=test_data_norm[ts + self._seq_len:ts + self._seq_len + self._horizon],
                data=tm_pred[ts:ts + self._seq_len],
                m_indicator=m_indicator[ts:ts + self._seq_len]
            )

            y_truths.append(y)

            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)
            y_preds.append(vals['outputs'])

            losses.append(vals['loss'])
            mses.append(vals['mse'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])

            pred = vals['outputs'][0, 0, :, 0]

            if self._flow_selection == 'Random':
                sampling = m_indicator[ts + self._seq_len]
            else:
                sampling = self.set_measured_flow_fairness(labels=m_indicator[ts: ts + self._seq_len])
                m_indicator[ts + self._seq_len] = sampling

            # invert of sampling: for choosing value from the original data

            ground_true = test_data_norm[ts + self._seq_len]

            # Merge value from pred_input and measured_input
            new_input = pred * (1.0 - sampling) + ground_true * sampling

            # Concatenating new_input into current rnn_input
            tm_pred[ts + self._seq_len] = new_input

        results = {'loss': np.mean(losses),
                   'mse': np.mean(mses),
                   'y_preds': y_preds,
                   'tm_pred': tm_pred[self._seq_len:],
                   'm_indicator': m_indicator[self._seq_len:],
                   'y_truths': y_truths
                   }
        return results

    def get_lr(self, sess):
        return sess.run(self._lr).item()

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    def _train(self, sess, base_lr, epoch, steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        training_history = pd.DataFrame()
        losses, val_losses = [], []

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
            self._logger.info('Training epoch: {}/{}'.format(self._epoch, epochs))
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
            val_results = self.run_epoch_generator(sess, self._val_model,
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
            losses.append(train_loss)
            val_losses.append(val_loss)
            sys.stdout.flush()

        training_history['epoch'] = np.arange(self._epoch)
        training_history['loss'] = losses
        training_history['val_loss'] = val_losses
        training_history.to_csv(self._log_dir + 'training_history.csv', index=False)

        return np.min(history)

    def _prepare_test_set(self):

        y_test = np.zeros(shape=(self._data['test_data_norm'].shape[0] - self._seq_len - self._horizon + 1,
                                 self._horizon,
                                 self._nodes,
                                 1), dtype='float32')
        for t in range(self._data['test_data_norm'].shape[0] - self._seq_len - self._horizon + 1):
            y_test[t] = np.expand_dims(self._data['test_data_norm']
                                       [t + self._seq_len:t + self._seq_len + self._horizon],
                                       axis=2)

        return y_test

    def _test(self, sess):
        n_metrics = 4
        # Metrics: MSE, MAE, RMSE, MAPE, ER
        metrics_summary = np.zeros(shape=(self._run_times + 3, self._horizon * n_metrics + 1))

        for i in range(self._run_times):
            self._logger.info('|--- Run time: {}'.format(i))
            # y_test = self._prepare_test_set()

            test_results = self._run_tm_prediction(sess, model=self._test_model, runId=i)

            self._calculate_metrics(prediction_results=test_results, metrics_summary=metrics_summary,
                                    scaler=self._data['scaler'],
                                    runId=i, data_norm=self._data['test_data_norm'])

        self._summarize_results(metrics_summary=metrics_summary, n_metrics=n_metrics)

        return

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def test(self, sess, **kwargs):
        kwargs.update(self._test_kwargs)
        return self._test(sess)

    def evaluate(self, sess):
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

            y_pred = scaler.inverse_transform(y_preds[:, horizon_i, :, 0])
            predictions.append(y_pred)

            mse = metrics.masked_mse_np(preds=y_pred, labels=y_truth, null_val=0)
            mae = metrics.masked_mae_np(preds=y_pred, labels=y_truth, null_val=0)
            mape = metrics.masked_mape_np(preds=y_pred, labels=y_truth, null_val=0)
            rmse = metrics.masked_rmse_np(preds=y_pred, labels=y_truth, null_val=0)
            self._logger.info(
                "Horizon {:02d}, MSE: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}".format(
                    horizon_i + 1, mse, mae, rmse, mape
                )
            )
            utils.add_simple_summary(self._writer,
                                     ['%s_%d' % (item, horizon_i + 1) for item in
                                      ['metric/rmse', 'metric/mae', 'metric/mse']],
                                     [rmse, mae, mse],
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
