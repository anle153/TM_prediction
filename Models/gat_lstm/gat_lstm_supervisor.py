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

from Models.gat_lstm.gat_lstm_model import GATLSTMModel
from lib import utils


def _get_log_dir_graph_based(kwargs):
    alg = kwargs.get('alg')
    batch_size = kwargs['data'].get('batch_size')
    learning_rate = kwargs['train'].get('base_lr')
    mon_ratio = kwargs['mon_ratio']
    scaler = kwargs['scaler']

    # ADJ_METHOD = ['CORR1', 'CORR2', 'OD', 'EU_PPA', 'DTW', 'DTW_PPA', 'SAX', 'KNN', 'SD']
    adj_method = kwargs['data'].get('adj_method')

    run_id = '%s_%g_%s_%g_%g_%s/' % (alg, mon_ratio, adj_method, learning_rate, batch_size, scaler)
    base_dir = kwargs.get('base_dir')
    log_dir = os.path.join(base_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_log_dir(kwargs):
    log_dir = kwargs['train'].get('log_dir')
    if log_dir is None:
        return _get_log_dir_graph_based(kwargs)
    else:
        return log_dir


class GATLSTMSupervisor():
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, is_training=False, **kwargs):
        self._kwargs = kwargs

        self._alg = kwargs.get('alg')
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')
        self._model_kwargs = kwargs.get('model')
        self._base_dir = kwargs.get('base_dir')

        self._epochs = self._train_kwargs.get('epochs')

        # logging.
        self._log_dir = get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)

        self._mon_ratio = float(self._kwargs.get('mon_ratio'))

        # Model's args
        self._input_dim = int(self._model_kwargs.get('input_dim'))
        self._nodes = int(self._model_kwargs.get('num_nodes'))
        self._drop_out = float(self._train_kwargs.get('dropout'))
        self.batch_size = int(self._data_kwargs['batch_size'])
        # Test's args
        self._flow_selection = self._test_kwargs.get('flow_selection')
        self._run_times = self._test_kwargs.get('run_times')
        # Data preparation
        self._day_size = self._data_kwargs.get('day_size')

        self._data = utils.load_dataset_gatlstm(num_nodes=self._model_kwargs.get('num_nodes'),
                                                input_dim=self._model_kwargs.get('input_dim'),
                                                mon_ratio=self._mon_ratio,
                                                scaler_type=self._kwargs.get('scaler'),
                                                is_training=is_training,
                                                **self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        if is_training:
            self.model = GATLSTMModel(scaler=scaler,
                                      batch_size=self.batch_size,
                                      **self._model_kwargs)
        else:
            self.model = GATLSTMModel(scaler=scaler,
                                      batch_size=1,
                                      **self._model_kwargs)

        # Learning rate.
        max_to_keep = self._train_kwargs.get('max_to_keep', 100)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        self.summary_writer = tf.summary.FileWriter(self._log_dir)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    def run_epoch_generator(self, sess, model, data_generator, return_output=False, training=False, writer=None):
        losses = []

        adj_mx = self._data['adj_mx']
        adj_mx = np.expand_dims(adj_mx, 0)
        adj_mx = np.tile(adj_mx, [self.batch_size, 1, 1])

        for _, (x, y) in enumerate(data_generator):
            feed_dict = {
                model.inputs: x,
                model.labels: y,
                model.adj_mx: adj_mx,
                model.attn_drop: 0.6,
                model.ffd_drop: 0.6
            }
            res = model.train(sess, feed_dict, self.model_summary_writer,
                              with_output=True)
            losses.append(res['loss'])

        results = {
            'loss': np.mean(losses)
        }
        return results

    def _prepare_input(self, ground_truth, data, m_indicator):

        x = np.zeros(shape=(self._seq_len, self._nodes, self._input_dim), dtype='float32')
        y = np.zeros(shape=(self._horizon, self._nodes), dtype='float32')

        x[:, :, 0] = data
        x[:, :, 1] = m_indicator

        y[:] = ground_truth
        y = np.expand_dims(y, axis=2)

        return np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)

    def _run_tm_prediction(self, sess, model, runId, writer=None):

        test_data_norm = self._data['test_data_norm']

        # Initialize traffic matrix data
        tm_pred, m_indicator = self._init_data_test(test_data_norm, runId)

        y_preds = []
        y_truths = []

        for ts in tqdm(range(test_data_norm.shape[0] - self._horizon - self._seq_len)):

            x = self._prepare_input_dcrnn(
                data=tm_pred[ts:ts + self._seq_len],
                m_indicator=m_indicator[ts:ts + self._seq_len]
            )

            y_truths.append(
                np.expand_dims(test_data_norm[ts + self._seq_len:ts + self._seq_len + self._horizon].copy(), axis=0))

            feed_dict = {
                model.inputs: x,
            }

            res = model.test(sess, feed_dict, with_output=True)

            y_preds.append(np.squeeze(res['output'], axis=-1))

            pred = res['output'][0, 0, :, 0]

            sampling = self._monitored_flows_slection(time_slot=ts, m_indicator=m_indicator)

            # invert of sampling: for choosing value from the original data

            ground_true = test_data_norm[ts + self._seq_len]

            # Merge value from pred_input and measured_input
            new_input = pred * (1.0 - sampling) + ground_true * sampling

            # Concatenating new_input into current rnn_input
            tm_pred[ts + self._seq_len] = new_input

        results = {'y_preds': y_preds,
                   'tm_pred': tm_pred[self._seq_len:],
                   'm_indicator': m_indicator[self._seq_len:],
                   'y_truths': y_truths
                   }
        return results

    def train(self, sess, save_model=1, patience=50):
        history = []
        training_history = pd.DataFrame()
        losses, val_losses = [], []

        min_val_loss = float('inf')
        wait = 0

        self._epoch = int(self._train_kwargs.get('epoch'))

        if self._epoch > 0:
            pretrained_model = self._train_kwargs.get('model_filename')
            self._logger.info("[*] Saved result exists! loading...")
            self.saver.restore(
                sess,
                pretrained_model
            )
            self._logger.info("[*] Loaded previously trained weights")
            self.b_pretrain_loaded = True
        else:
            self._logger.info("[*] No previous result")
            self.b_pretrain_loaded = False

        self._logger.info("[*] Training starts...")
        self.model_summary_writer = None

        ##Training

        while self._epoch <= self._epochs:
            start_time = time.time()

            self._logger.info('Training epoch: {}/{}'.format(self._epoch, self._epochs))

            # run training
            train_data_generator = self._data['train_loader'].get_iterator()
            val_data_generator = self._data['val_loader'].get_iterator()

            res = self.run_epoch_generator(sess, model=self.model,
                                           data_generator=train_data_generator)
            train_loss = res['loss']

            # run validating
            val_res = self.run_epoch_generator(sess, model=self.model,
                                               data_generator=val_data_generator)
            val_loss = val_res['loss']
            end_time = time.time()
            message = 'Epoch [{}/{}] train_loss: {:f}, val_loss: {:f} {:.1f}s'.format(
                self._epoch, self._epochs, train_loss, val_loss, (end_time - start_time))
            self._logger.info(message)

            # early stopping
            if val_loss <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_loss)
                self._logger.info(
                    'Val loss decrease from %f to %f, saving to %s' % (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warning('Early stopping at epoch: %d' % self._epoch)
                    break

            self._epoch += 1

            history.append(val_loss)
            # Increases epoch.
            losses.append(train_loss)
            val_losses.append(val_loss)
            sys.stdout.flush()

        training_history['epoch'] = np.arange(self._epoch)
        training_history['loss'] = losses
        training_history['val_loss'] = val_losses
        training_history.to_csv(self._log_dir + 'training_history.csv', index=False)

        return np.min(history)

    def test(self, sess):

        pretrained_model = self._train_kwargs.get('model_filename')
        self._logger.info("[*] Saved result exists! loading...")
        self.saver.restore(
            sess,
            pretrained_model
        )
        self._logger.info("[*] Loaded previously trained weights")
        self.b_pretrain_loaded = True

        n_metrics = 4
        # Metrics: MSE, MAE, RMSE, MAPE, ER
        metrics_summary = np.zeros(shape=(self._run_times + 3, self._horizon * n_metrics + 1))

        for i in range(self._run_times):
            self._logger.info('|--- Run time: {}'.format(i))
            # y_test = self._prepare_test_set()

            test_results = self._run_tm_prediction(sess, model=self.model, runId=i)

            metrics_summary = self._calculate_metrics(prediction_results=test_results, metrics_summary=metrics_summary,
                                                      scaler=self._data['scaler'],
                                                      runId=i, data_norm=self._data['test_data_norm'])

        self._summarize_results(metrics_summary=metrics_summary, n_metrics=n_metrics)

        return


    def evaluate(self, sess):
        pass

    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        prefix = os.path.join(self._log_dir, 'models-{}-{}'.format(self._epoch, val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self.saver.save(sess, prefix)

        config_filename = 'config_{}.yaml'.format(self._epoch)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']
