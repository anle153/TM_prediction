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
            if 'dcrnn' in alg:
                return _get_log_dir_dcrnn_based(kwargs)
            elif 'lstm' in alg:
                return _get_log_dir_lstm_based(kwargs)
        else:
            return log_dir

    def plot_training_history(self, model_history):
        import matplotlib.pyplot as plt

        plt.plot(model_history.history['loss'], label='loss')
        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[loss]{}.png'.format(self._alg))
        plt.legend()
        plt.close()

        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[val_loss]{}.png'.format(self._alg))
        plt.legend()
        plt.close()

    def _save_results(self, g_truth, pred_tm, m_indicator, tag):
        np.save(self._log_dir + '/g_truth{}'.format(tag), g_truth)
        np.save(self._log_dir + '/pred_tm_{}'.format(tag), pred_tm)
        if self._flow_selection != 'Random':
            np.save(self._log_dir + '/m_indicator{}'.format(tag), m_indicator)

    def plot_models(self, model):
        plot_model(model=model, to_file=self._log_dir + '/model.png', show_shapes=True)

    def _calculate_metrics(self, prediction_results, metrics_summary, scaler, runId, data_norm, n_metrics=4):
        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = prediction_results['loss'], prediction_results['y_preds']

        y_preds = prediction_results['y_preds']
        y_preds = np.concatenate(y_preds, axis=0)

        y_truths = prediction_results['y_truths']
        y_truths = np.concatenate(y_truths, axis=0)
        predictions = []

        for horizon_i in range(self._horizon):
            y_truth = scaler.inverse_transform(y_truths[:, horizon_i, :, 0])

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

        self._save_results(g_truth=g_truth, pred_tm=tm_pred, m_indicator=m_indicator, tag=str(runId))
        return

    def _summarize_results(self, metrics_summary, n_metrics):
        results_summary = pd.DataFrame(index=range(self._run_times + 3))
        results_summary['No.'] = range(self._run_times + 3)

        avg = np.mean(metrics_summary, axis=0)
        std = np.std(metrics_summary, axis=0)
        conf = metrics.calculate_confident_interval(metrics_summary)
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


class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
