import os
import time

import keras.callbacks as keras_callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import plot_model

from lib import metrics
from lib import utils


def _get_log_dir_lstm_based(kwargs):
    alg = kwargs.get('alg')
    batch_size = kwargs['data'].get('batch_size')
    rnn_units = kwargs['model'].get('rnn_units')
    horizon = kwargs['model'].get('horizon')

    mon_ratio = kwargs['mon_ratio']

    scaler = kwargs['scaler']

    run_id = '%s_%d_%g_%d_%d_%s/' % (alg, rnn_units, mon_ratio, horizon, batch_size, scaler)
    base_dir = kwargs.get('base_dir')
    log_dir = os.path.join(base_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def _get_log_dir_dcrnn_based(kwargs):
    alg = kwargs.get('alg')
    batch_size = kwargs['data'].get('batch_size')
    learning_rate = kwargs['train'].get('base_lr')
    max_diffusion_step = kwargs['model'].get('max_diffusion_step')
    num_rnn_layers = kwargs['model'].get('num_rnn_layers')
    rnn_units = kwargs['model'].get('rnn_units')
    structure = '-'.join(
        ['%d' % rnn_units for _ in range(num_rnn_layers)])
    horizon = kwargs['model'].get('horizon')
    seq_len = kwargs['model'].get('seq_len')
    filter_type = kwargs['model'].get('filter_type')
    filter_type_abbr = 'L'
    if filter_type == 'random_walk':
        filter_type_abbr = 'R'
    elif filter_type == 'dual_random_walk':
        filter_type_abbr = 'DR'

    mon_ratio = kwargs['mon_ratio']
    scaler = kwargs['scaler']

    # ADJ_METHOD = ['CORR1', 'CORR2', 'OD', 'EU_PPA', 'DTW', 'DTW_PPA', 'SAX', 'KNN', 'SD']
    adj_method = kwargs['data'].get('adj_method')
    adj_pos_thres = kwargs['data'].get('pos_thres')
    adj_neg_thres = -kwargs['data'].get('neg_thres')

    if adj_method != 'OD':
        run_id = '%s_%s_%g_%d_%s_%g_%g_%d_%d_%s_%g_%d_%s/' % (
            alg, filter_type_abbr, mon_ratio, max_diffusion_step, adj_method, adj_pos_thres, adj_neg_thres,
            horizon, seq_len, structure, learning_rate, batch_size, scaler)
    else:
        run_id = '%s_%s_%g_%d_%s_%d_%d_%s_%g_%d_%s/' % (
            alg, filter_type_abbr, mon_ratio, max_diffusion_step, adj_method,
            horizon, seq_len, structure, learning_rate, batch_size, scaler)
    base_dir = kwargs.get('base_dir')
    log_dir = os.path.join(base_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


class AbstractModel(object):

    def __init__(self, **kwargs):
        self._kwargs = kwargs

        self._alg = kwargs.get('alg')
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')
        self._model_kwargs = kwargs.get('model')
        self._base_dir = kwargs.get('base_dir')

        self._epochs = self._train_kwargs.get('epochs')

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
        self._rnn_units = int(self._model_kwargs.get('rnn_units'))
        self._drop_out = float(self._train_kwargs.get('dropout'))

        # Test's args
        self._flow_selection = self._test_kwargs.get('flow_selection')
        self._run_times = self._test_kwargs.get('run_times')
        # Data preparation
        self._day_size = self._data_kwargs.get('day_size')

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            alg = kwargs.get('alg')
            if 'dcrnn' in alg or 'dclstm' in alg:
                return _get_log_dir_dcrnn_based(kwargs)
            elif 'lstm' in alg:
                return _get_log_dir_lstm_based(kwargs)
        else:
            return log_dir

    def plot_training_history(self, model_history, tag=None):
        import matplotlib.pyplot as plt

        plt.plot(model_history.history['loss'], label='loss')
        plt.plot(model_history.history['val_loss'], label='val_loss')
        if tag is not None:
            saved_loss = '[loss_{}]{}.png'.format(tag, self._alg)
        else:
            saved_loss = '[loss]{}.png'.format(self._alg)

        plt.savefig(self._log_dir + saved_loss)
        plt.legend()
        plt.close()

        if tag is not None:
            saved_val_loss = '[val_loss_{}]{}.png'.format(tag, self._alg)
        else:
            saved_val_loss = '[val_loss]{}.png'.format(self._alg)

        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + saved_val_loss)
        plt.legend()
        plt.close()

    def _save_results(self, g_truth, pred_tm, m_indicator, tag):
        np.save(self._log_dir + '/g_truth{}'.format(tag), g_truth)
        np.save(self._log_dir + '/pred_tm_{}'.format(tag), pred_tm)
        if self._flow_selection != 'Random':
            np.save(self._log_dir + '/m_indicator{}'.format(tag), m_indicator)

    def plot_models(self, model, tag=None):
        if tag is None:
            model_name = '/model.png'
        else:
            model_name = '/{}_model.png'.format(tag)
        plot_model(model=model, to_file=self._log_dir + model_name, show_shapes=True)

    def _calculate_metrics(self, prediction_results, metrics_summary, scaler, runId, data_norm, n_metrics=4):
        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        y_preds = prediction_results['y_preds']
        y_preds = np.concatenate(y_preds, axis=0)

        y_truths = prediction_results['y_truths']
        y_truths = np.concatenate(y_truths, axis=0)
        predictions = []

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
            metrics_summary[runId, horizon_i * n_metrics + 0] = mse
            metrics_summary[runId, horizon_i * n_metrics + 1] = mae
            metrics_summary[runId, horizon_i * n_metrics + 2] = rmse
            metrics_summary[runId, horizon_i * n_metrics + 3] = mape

        tm_pred = scaler.inverse_transform(prediction_results['tm_pred'])
        g_truth = scaler.inverse_transform(data_norm[self._seq_len:-self._horizon])
        m_indicator = prediction_results['m_indicator']
        er = metrics.error_ratio(y_pred=tm_pred,
                                 y_true=g_truth,
                                 measured_matrix=m_indicator)
        metrics_summary[runId, -1] = er
        self._logger.info('ER: {}'.format(er))
        self._save_results(g_truth=g_truth, pred_tm=tm_pred, m_indicator=m_indicator, tag=str(runId))
        return metrics_summary

    def _summarize_results(self, metrics_summary, n_metrics):
        results_summary = pd.DataFrame(index=range(self._run_times + 3))
        results_summary['No.'] = range(self._run_times + 3)

        avg = np.mean(metrics_summary[:self._run_times], axis=0)
        std = np.std(metrics_summary[:self._run_times], axis=0)
        conf = metrics.calculate_confident_interval(metrics_summary[:self._run_times])
        metrics_summary[-3, :] = avg
        metrics_summary[-2, :] = std
        metrics_summary[-1, :] = conf
        self._logger.info('AVG: {}'.format(metrics_summary[-3, :]))

        for horizon_i in range(self._horizon):
            results_summary['mse_{}'.format(horizon_i)] = metrics_summary[:, horizon_i * n_metrics + 0]
            results_summary['mae_{}'.format(horizon_i)] = metrics_summary[:, horizon_i * n_metrics + 1]
            results_summary['rmse_{}'.format(horizon_i)] = metrics_summary[:, horizon_i * n_metrics + 2]
            results_summary['mape_{}'.format(horizon_i)] = metrics_summary[:, horizon_i * n_metrics + 3]

        results_summary['er'] = metrics_summary[:, -1]
        results_summary.to_csv(self._log_dir + 'results_summary.csv', index=False)

    def _prepare_input_dcrnn(self, data, m_indicator):
        x = np.zeros(shape=(self._seq_len, self._nodes, self._input_dim), dtype='float32')
        x[:, :, 0] = data
        x[:, :, 1] = m_indicator
        return np.expand_dims(x, axis=0)

    def _prepare_input_lstm(self, data, m_indicator):

        dataX = np.zeros(shape=(data.shape[1], self._seq_len, self._input_dim), dtype='float32')
        for flow_id in range(data.shape[1]):
            x = data[:, flow_id]
            label = m_indicator[:, flow_id]

            dataX[flow_id, :, 0] = x
            dataX[flow_id, :, 1] = label

        return dataX

    def _init_data_test(self, test_data_norm, runId):
        tm_pred = np.zeros(shape=(test_data_norm.shape[0] - self._horizon, self._nodes), dtype='float32')
        tm_pred[0:self._seq_len] = test_data_norm[:self._seq_len]

        # Initialize measurement matrix
        if self._flow_selection == 'Random':
            save_m_indicator = os.path.join(self._base_dir + '/random_m_indicator_{}_{}_{}/'.format(
                self._seq_len, self._horizon, self._mon_ratio))
            if not os.path.isfile(os.path.join(save_m_indicator + '/m_indicator{}.npy'.format(runId))):
                m_indicator = np.random.choice([1.0, 0.0],
                                               size=(test_data_norm.shape[0] - self._horizon -
                                                     self._seq_len, test_data_norm.shape[1]),
                                               p=(self._mon_ratio, 1.0 - self._mon_ratio))
                if not os.path.isdir(save_m_indicator):
                    os.makedirs(save_m_indicator)
                np.save(save_m_indicator + '/m_indicator{}.npy'.format(runId), m_indicator)
            else:
                m_indicator = np.load(os.path.join(save_m_indicator + '/m_indicator{}.npy'.format(runId)))

            m_indicator = np.concatenate([np.ones(shape=(self._seq_len, self._nodes)), m_indicator], axis=0)
        else:
            m_indicator = np.zeros(shape=(test_data_norm.shape[0] - self._horizon, self._nodes),
                                   dtype='float32')
            m_indicator[0:self._seq_len] = np.ones(shape=(self._seq_len, self._nodes))

        return tm_pred, m_indicator

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

    def _set_measured_flow_fairness(self, m_indicator):
        """

        :param m_indicator: shape(#seq_len, #nflows)
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

    def _calculate_flows_weights(self, fw_losses, m_indicator, lamda):
        """

        :param fw_losses: shape(#n_flows)
        :param m_indicator: shape(#seq_len, #nflows)
        :return: w: flow_weight shape(#n_flows)
        """

        cl = self._calculate_consecutive_loss(m_indicator)

        w = 1 / (fw_losses * lamda[0] +
                 cl * lamda[1])

        return w

    def _set_measured_flow(self, rnn_input, pred_forward, m_indicator, lamda):
        """

        :param rnn_input: shape(#seq_len, #nflows)
        :param pred_forward: shape(#seq_len, #nflows)
        :param m_indicator: shape(#seq_len, #nflows)
        :return:
        """

        n_flows = rnn_input.shape[0]

        fw_losses = []
        for flow_id in range(m_indicator.shape[1]):
            idx_fw = m_indicator[1:, flow_id]

            # fw_losses.append(error_ratio(y_true=rnn_input[1:, flow_id][idx_fw == 1.0],
            #                              y_pred=pred_forward[:-1, flow_id][idx_fw == 1.0],
            #                              measured_matrix=np.zeros(idx_fw[idx_fw == 1.0].shape)))
            fw_losses.append(metrics.masked_mae_np(preds=pred_forward[:-1, flow_id][idx_fw == 1.0],
                                                   labels=rnn_input[1:, flow_id][idx_fw == 1.0]))

        fw_losses = np.array(fw_losses)
        fw_losses[fw_losses == 0.] = np.max(fw_losses)

        w = self._calculate_flows_weights(fw_losses=fw_losses,
                                          m_indicator=m_indicator, lamda=lamda)

        sampling = np.zeros(shape=n_flows)
        m = int(self._mon_ratio * n_flows)

        w = w.flatten()
        sorted_idx_w = np.argsort(w)
        sampling[sorted_idx_w[:m]] = 1

        return sampling

    def _monitored_flows_slection(self, time_slot, m_indicator, tm_pred=None, fw_outputs=None, lamda=None):
        if self._flow_selection == 'Random':
            sampling = m_indicator[time_slot + self._seq_len]
        elif self._flow_selection == 'Fairness':
            sampling = self._set_measured_flow_fairness(m_indicator=m_indicator[time_slot: time_slot + self._seq_len])
            m_indicator[time_slot + self._seq_len] = sampling
        else:
            sampling = self._set_measured_flow(rnn_input=tm_pred[time_slot: time_slot + self._seq_len],
                                               pred_forward=fw_outputs,
                                               m_indicator=m_indicator[time_slot: time_slot + self._seq_len].T,
                                               lamda=lamda)
            m_indicator[time_slot + self._seq_len] = sampling

        return sampling

    def _data_correction_v3(self, rnn_input, pred_backward, labels, r):
        # Shape = (#n_flows, #time-steps)
        _rnn_input = np.copy(rnn_input.T)
        _labels = np.copy(labels.T)

        beta = np.zeros(_rnn_input.shape)

        corrected_range = int(self._seq_len / r)

        for i in range(_rnn_input.shape[1] - corrected_range):
            mu = np.sum(_labels[:, i + 1:i + corrected_range + 1], axis=1) / corrected_range

            h = np.arange(1, corrected_range + 1)

            rho = (1 / (np.log(corrected_range) + 1)) * np.sum(
                _labels[:, i + 1:i + corrected_range + 1] / h, axis=1)

            beta[:, i] = mu * rho

        considered_backward = pred_backward[:, 1:]
        considered_rnn_input = _rnn_input[:, 0:-1]

        beta[beta > 0.5] = 0.5

        alpha = 1.0 - beta

        alpha = alpha[:, 0:-1]
        beta = beta[:, 0:-1]
        # gamma = gamma[:, 1:-1]

        # corrected_data = considered_rnn_input * alpha + considered_rnn_input * beta + considered_backward * gamma
        corrected_data = considered_rnn_input * alpha + considered_backward * beta

        return corrected_data.T

    def _calculate_pred_err(self, pred, tm, m_indicator, beta=0.1):
        """

        :param pred: shape (ts: ts + seq_len - 1, nflow)
        :param tm: shape (ts: ts + seq_len - 1, nflow)
        :param m_indicator: shape (ts: ts + seq_len - 1, nflow)
        :return:
        """

        _m = 1.0 - m_indicator
        er = metrics.error_ratio(y_true=tm, y_pred=pred, measured_matrix=_m)

        return er + beta * np.sqrt(1 / np.sum(m_indicator))

    def save_model_history(self, times, model_history):
        loss = np.array(model_history.history['loss'])
        val_loss = np.array(model_history.history['val_loss'])
        dump_model_history = pd.DataFrame(index=range(loss.size),
                                          columns=['epoch', 'loss', 'val_loss', 'train_time'])

        dump_model_history['epoch'] = range(loss.size)
        dump_model_history['loss'] = loss
        dump_model_history['val_loss'] = val_loss

        if times is not None:
            dump_model_history['train_time'] = times

        dump_model_history.to_csv(self._log_dir + 'training_history.csv', index=False)

class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
