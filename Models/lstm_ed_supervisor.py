import os
import sys
import time

import keras.callbacks as keras_callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import plot_model
from tqdm import tqdm

from Models.encoder_decoder_model import EncoderDecoderLSTM
from common.error_utils import error_ratio
from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import masked_mse_loss


class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class lstm_ed():

    def __init__(self, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')
        self._model_kwargs = kwargs.get('model')

        self._alg_name = self._kwargs.get('alg')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)

        # Data's args
        self._day_size = self._data_kwargs.get('day_size')

        # Model's Args
        self._rnn_units = self._model_kwargs.get('rnn_units')
        self._seq_len = self._model_kwargs.get('seq_len')
        self._horizon = self._model_kwargs.get('horizon')
        self._input_dim = self._model_kwargs.get('input_dim')
        self._input_shape = (self._seq_len, self._input_dim)
        self._output_dim = self._model_kwargs.get('output_dim')
        self._nodes = self._model_kwargs.get('num_nodes')
        self._n_rnn_layers = self._model_kwargs.get('n_rnn_layers')

        # Train's args
        self._drop_out = self._train_kwargs.get('dropout')
        self._epochs = self._train_kwargs.get('epochs')
        self._batch_size = self._data_kwargs.get('batch_size')

        # Test's args
        self._run_times = self._test_kwargs.get('run_times')
        self._flow_selection = self._test_kwargs.get('flow_selection')
        self._test_size = self._test_kwargs.get('test_size')
        self._results_path = self._test_kwargs.get('results_path')

        self._mon_ratio = self._kwargs.get('mon_ratio')

        # Load data
        self._data = utils.load_dataset_lstm(seq_len=self._seq_len, horizon=self._horizon, input_dim=self._input_dim,
                                             mon_ratio=self._mon_ratio, test_size=self._test_size,
                                             **self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        scaler = self._data['scaler']

        # Model
        with tf.name_scope('Train'):
            with tf.variable_scope('LSTM', reuse=False):
                self._train_model = EncoderDecoderLSTM(is_training=True, scaler=self._data['scaler'],
                                                       batch_size=self._batch_size, **self._model_kwargs)
        with tf.name_scope('Val'):
            with tf.variable_scope('LSTM', reuse=True):
                self._val_model = EncoderDecoderLSTM(is_training=True, scaler=self._data['scaler'],
                                                     batch_size=self._data_kwargs['val_batch_size'],
                                                     **self._model_kwargs)
        with tf.name_scope('Eval'):
            with tf.variable_scope('LSTM', reuse=True):
                self._eval_model = EncoderDecoderLSTM(is_training=True, scaler=self._data['scaler'],
                                                      batch_size=self._data_kwargs['eval_batch_size'],
                                                      **self._model_kwargs)
        with tf.name_scope('Test'):
            with tf.variable_scope('LSTM', reuse=True):
                self._test_model = EncoderDecoderLSTM(is_training=True, scaler=self._data['scaler'],
                                                      batch_size=self._data_kwargs['test_batch_size'],
                                                      **self._model_kwargs)

        # Learning rate
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
            run_id = 'lstm_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def plot_models(self):
        plot_model(model=self.model, to_file=self._log_dir + '/model.png', show_shapes=True)

    def _plot_training_history(self, model_history):
        import matplotlib.pyplot as plt

        plt.plot(model_history.history['loss'], label='loss')
        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[val_loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

    def _prepare_input(self, data, m_indicator):

        dataX = np.zeros(shape=(data.shape[1], self._seq_len, 2), dtype='float32')
        for flow_id in range(data.shape[1]):
            x = data[-self._seq_len:, flow_id]
            label = m_indicator[-self._seq_len:, flow_id]

            sample = np.array([x, label]).T
            dataX[flow_id] = sample

        return dataX

    def _set_measured_flow_fairness(self, m_indicator):
        """

        :param rnn_input: shape(#n_flows, #time-steps)
        :param m_indicator: shape(n_flows, #time-steps)
        :return:
        """

        cl = self._calculate_consecutive_loss(m_indicator).astype(float)

        w = 1 / cl

        sampling = np.zeros(shape=self._nodes, dtype='float32')
        m = int(self._mon_ratio * self._nodes)

        w = w.flatten()
        sorted_idx_w = np.argsort(w)
        sampling[sorted_idx_w[:m]] = 1

        return sampling

    @staticmethod
    def _calculate_consecutive_loss(m_indicator):

        consecutive_losses = []
        for flow_id in range(m_indicator.shape[1]):
            flows_labels = m_indicator[:, flow_id]
            if flows_labels[-1] == 1:
                consecutive_losses.append(1)
            else:
                measured_idx = np.argwhere(flows_labels == 1)
                if measured_idx.size == 0:
                    consecutive_losses.append(m_indicator.shape[0])
                else:
                    consecutive_losses.append(m_indicator.shape[0] - measured_idx[-1][0])

        consecutive_losses = np.asarray(consecutive_losses)
        return consecutive_losses

    def _ims_tm_prediction(self, init_data, init_labels):
        multi_steps_tm = np.zeros(shape=(init_data.shape[0] + self._horizon, init_data.shape[1]),
                                  dtype='float32')
        multi_steps_tm[0:self._seq_len] = init_data

        m_indicator = np.zeros(shape=(init_labels.shape[0] + self._horizon, init_labels.shape[1]),
                               dtype='float32')
        m_indicator[0:self._seq_len] = init_labels

        for ts_ahead in range(self._horizon):
            rnn_input = self._prepare_input(data=multi_steps_tm[ts_ahead:ts_ahead + self._seq_len],
                                            m_indicator=m_indicator[ts_ahead:ts_ahead + self._seq_len])
            predictX = self.model.predict(rnn_input)
            multi_steps_tm[ts_ahead + self._seq_len] = np.squeeze(predictX, axis=1)

        return multi_steps_tm[-self._horizon:]

    def _run_tm_prediction(self):

        test_data_norm = self._data['test_data_norm']

        tf_a = np.array([1.0, 0.0])
        m_indicator = np.zeros(shape=(test_data_norm.shape[0] - self._horizon, self._nodes),
                               dtype='float32')

        tm_pred = np.zeros(shape=(test_data_norm.shape[0] - self._horizon, self._nodes),
                           dtype='float32')

        tm_pred[0:self._seq_len] = test_data_norm[0:self._seq_len]
        m_indicator[0:self._seq_len] = np.ones(shape=(self._seq_len, self._nodes))

        y_preds = []
        y_truths = []

        # Predict the TM from time slot look_back
        for ts in tqdm(range(test_data_norm.shape[0] - self._horizon - self._seq_len)):
            # This block is used for iterated multi-step traffic matrices prediction

            predicted_tm = self._ims_tm_prediction(init_data=tm_pred[ts:ts + self._seq_len],
                                                   init_labels=m_indicator[ts:ts + self._seq_len])

            # Get the TM prediction of next time slot

            y_preds.append(np.expand_dims(predicted_tm, axis=0))
            pred = predicted_tm[0]

            # Using part of current prediction as input to the next estimation
            # Randomly choose the flows which is measured (using the correct data from test_set)

            # boolean array(1 x n_flows):for choosing value from predicted data
            if self._flow_selection == 'Random':
                sampling = np.random.choice(tf_a, size=self._nodes,
                                            p=[self._mon_ratio, 1 - self._mon_ratio])
            else:
                sampling = self._set_measured_flow_fairness(m_indicator=m_indicator[ts: ts + self._seq_len])

            m_indicator[ts + self._seq_len] = sampling
            # invert of sampling: for choosing value from the original data
            inv_sampling = 1.0 - sampling
            pred_input = pred * inv_sampling

            ground_true = test_data_norm[ts + self._seq_len]
            y_truths.append(
                np.expand_dims(test_data_norm[ts + self._seq_len:ts + self._seq_len + self._horizon], axis=0))

            measured_input = ground_true * sampling

            # Merge value from pred_input and measured_input
            new_input = pred_input + measured_input
            # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

            # Concatenating new_input into current rnn_input
            tm_pred[ts + self._seq_len] = new_input

        outputs = {
            'tm_pred': tm_pred[self._seq_len:],
            'm_indicator': m_indicator[self._seq_len:],
            'y_preds': y_preds,
            'y_truths': y_truths
        }

        return outputs

    def _save_results(self, g_truth, pred_tm, m_indicator, tag):
        np.save(self._log_dir + '/g_truth{}'.format(tag), g_truth)
        np.save(self._log_dir + '/pred_tm_{}'.format(tag), pred_tm)
        np.save(self._log_dir + '/m_indicator{}'.format(tag), m_indicator)

    def _test(self):
        scaler = self._data['scaler']
        results_summary = pd.DataFrame(index=range(self._run_times))
        results_summary['No.'] = range(self._run_times)

        n_metrics = 4
        # Metrics: MSE, MAE, RMSE, MAPE, ER
        metrics_summary = np.zeros(shape=(self._run_times, self._horizon * n_metrics + 1))

        for i in range(self._run_times):
            print('|--- Running time: {}/{}'.format(i, self._run_times))

            outputs = self._run_tm_prediction()

            tm_pred, m_indicator, y_preds = outputs['tm_pred'], outputs['m_indicator'], outputs['y_preds']

            y_preds = np.concatenate(y_preds, axis=0)
            predictions = []
            y_truths = outputs['y_truths']
            y_truths = np.concatenate(y_truths, axis=0)

            for horizon_i in range(self._horizon):
                y_truth = scaler.inverse_transform(y_truths[:, horizon_i, :])

                y_pred = scaler.inverse_transform(y_preds[:, horizon_i, :])
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
                metrics_summary[i, horizon_i * n_metrics + 0] = mse
                metrics_summary[i, horizon_i * n_metrics + 1] = mae
                metrics_summary[i, horizon_i * n_metrics + 2] = rmse
                metrics_summary[i, horizon_i * n_metrics + 3] = mape

            tm_pred = scaler.inverse_transform(tm_pred)
            g_truth = scaler.inverse_transform(self._data['test_data_norm'][self._seq_len:-self._horizon])
            er = error_ratio(y_pred=tm_pred,
                             y_true=g_truth,
                             measured_matrix=m_indicator)
            metrics_summary[i, -1] = er

            self._save_results(g_truth=g_truth, pred_tm=tm_pred, m_indicator=m_indicator, tag=str(i))

            print('ER: {}'.format(er))

        for horizon_i in range(self._horizon):
            results_summary['mse_{}'.format(horizon_i)] = metrics_summary[:, horizon_i * n_metrics + 0]
            results_summary['mae_{}'.format(horizon_i)] = metrics_summary[:, horizon_i * n_metrics + 1]
            results_summary['rmse_{}'.format(horizon_i)] = metrics_summary[:, horizon_i * n_metrics + 2]
            results_summary['mape_{}'.format(horizon_i)] = metrics_summary[:, horizon_i * n_metrics + 3]

        results_summary['er'] = metrics_summary[:, -1]
        results_summary.to_csv(self._log_dir + 'results_summary.csv', index=False)

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

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

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

            sys.stdout.flush()
        return np.min(history)

    def evaluate(self):
        scaler = self._data['scaler']

        y_pred = self.eval_model.predict(self._data['x_eval'])
        y_pred = scaler.inverse_transform(y_pred)
        y_truth = scaler.inverse_transform(self._data['y_eval'])

        mse = metrics.masked_mse_np(preds=y_pred, labels=y_truth, null_val=0)
        mae = metrics.masked_mae_np(preds=y_pred, labels=y_truth, null_val=0)
        mape = metrics.masked_mape_np(preds=y_pred, labels=y_truth, null_val=0)
        rmse = metrics.masked_rmse_np(preds=y_pred, labels=y_truth, null_val=0)
        self._logger.info(
            "Horizon {:02d}, MSE: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}".format(
                1, mse, mae, rmse, mape
            )
        )

    def test(self):
        return self._test()
