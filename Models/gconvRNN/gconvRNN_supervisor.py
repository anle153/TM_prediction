from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import scipy
import tensorflow as tf
import yaml
from tqdm import tqdm

import Models.gconvRNN.graph as graph
from Models.AbstractModel import AbstractModel
from Models.gconvRNN.gconvrnn_model import Model
from lib import utils


class GCONVRNN(AbstractModel):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, is_training=False, **kwargs):
        super(GCONVRNN, self).__init__(**kwargs)

        self._train_batch_size = int(kwargs['data'].get('batch_size'))
        self._test_batch_size = int(kwargs['data'].get('test_batch_size'))

        self._epoch = int(kwargs['train'].get('epoch'))

        self._logstep = int(kwargs['train'].get('logstep'))

        self._checkpoint_secs = int(kwargs['train'].get('checkpoint_secs'))

        self._data = utils.load_dataset_dcrnn(seq_len=self._model_kwargs.get('seq_len'),
                                              horizon=self._model_kwargs.get('horizon'),
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

        W = self._data['adj_mx']
        laplacian = W / W.max()
        laplacian = scipy.sparse.csr_matrix(laplacian, dtype=np.float32)
        lmax = graph.lmax(laplacian)

        if is_training:
            self._train_model = Model(is_training=True, laplacian=laplacian,
                                      lmax=lmax, batch_size=self._train_batch_size, reuse=True, **self._model_kwargs)
        else:
            self._test_model = Model(is_training=False, laplacian=laplacian,
                                     lmax=lmax, batch_size=self._test_batch_size, reuse=False, **self._model_kwargs)

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        self.summary_writer = tf.summary.FileWriter(self._log_dir)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    def run_epoch_generator(self, sess, model, data_generator):
        losses = []

        for _, (x, y) in enumerate(data_generator):
            feed_dict = {
                model.rnn_input: x,
                model.rnn_output: y
            }
            res = model.train(sess, feed_dict, self.model_summary_writer,
                              with_output=True)
            # self.model_summary_writer = self._get_summary_writer(sess)
            losses.append(res['loss'])

        results = {
            'loss': np.mean(losses)
        }
        return results

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
                model.rnn_input: x,
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
        min_val_loss = float('inf')
        wait = 0

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

            res = self.run_epoch_generator(sess, model=self._train_model,
                                           data_generator=train_data_generator)
            train_loss = res['loss']

            # run validating
            val_res = self.run_epoch_generator(sess, model=self._train_model,
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

            test_results = self._run_tm_prediction(sess, model=self._test_model, runId=i)

            metrics_summary = self._calculate_metrics(prediction_results=test_results, metrics_summary=metrics_summary,
                                                      scaler=self._data['scaler'],
                                                      runId=i, data_norm=self._data['test_data_norm'])

        self._summarize_results(metrics_summary=metrics_summary, n_metrics=n_metrics)

    def _get_summary_writer(self, sess):
        global_step = sess.run(tf.train.get_or_create_global_step())
        if global_step % self._logstep == 0:
            return self.summary_writer
        else:
            return None

    def load(self, sess, model_filename):
        """
        Restore from saved model.
        :param sess:
        :param model_filename:
        :return:
        """
        self.saver.restore(sess, model_filename)

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
