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
from lib import utils, metrics


class GCONVRNN(AbstractModel):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, is_training=False, **kwargs):
        super(GCONVRNN, self).__init__(**kwargs)

        self._logstep = int(kwargs['train'].get('logstep'))

        self._checkpoint_secs = int(kwargs['train'].get('checkpoint_secs'))

        self._data = utils.load_dataset_gconvrnn(seq_len=self._model_kwargs.get('seq_len'),
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

        W = self._data['adj']
        laplacian = W / W.max()
        laplacian = scipy.sparse.csr_matrix(laplacian, dtype=np.float32)
        lmax = graph.lmax(laplacian)

        with tf.name_scope('Train'):
            with tf.variable_scope('GCONVRNN', reuse=False):
                self._train_model = Model(is_training=True, laplacian=laplacian, lmax=lmax, **self._model_kwargs)

        with tf.name_scope('Val'):
            with tf.variable_scope('GCONVRNN', reuse=True):
                self._val_model = Model(is_training=False, laplacian=laplacian, lmax=lmax, **self._model_kwargs)

        with tf.name_scope('Eval'):
            with tf.variable_scope('GCONVRNN', reuse=True):
                self._eval_model = Model(is_training=False, laplacian=laplacian, lmax=lmax, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('GCONVRNN', reuse=True):
                self._test_model = Model(is_training=False, laplacian=laplacian, lmax=lmax, **self._model_kwargs)

        self.saver = tf.train.Saver()
        self.model_saver = tf.train.Saver(self._train_model.model_vars)
        self.summary_writer = tf.summary.FileWriter(self._log_dir)

        sv = tf.train.Supervisor(logdir=self._log_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_summaries_secs=300,
                                 save_model_secs=self._checkpoint_secs,
                                 global_step=self._train_model.model_step)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1.0,
            allow_growth=True)  # seems to be not working
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    def run_epoch_generator(self, model, data_generator):
        losses = []

        for _, (x, y) in enumerate(data_generator):
            feed_dict = {
                model.rnn_input: x,
                model.rnn_output: y
            }
            res = model.train(self.sess, feed_dict, self.model_summary_writer,
                              with_output=True)
            self.model_summary_writer = self._get_summary_writer(res)
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
        fetches = {
            'global_step': tf.train.get_or_create_global_step()
        }

        fetches.update({
            'outputs': model.outputs
        })

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

            vals = sess.run(fetches, feed_dict=feed_dict)
            y_preds.append(np.squeeze(vals['outputs'], axis=-1))

            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])

            pred = vals['outputs'][0, 0, :, 0]

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

    def train(self, save_model=1, patience=50):
        min_val_loss = float('inf')
        wait = 0

        print("[*] Checking if previous run exists in {}"
              "".format(self._log_dir))
        latest_checkpoint = tf.train.latest_checkpoint(self._log_dir)
        if tf.train.latest_checkpoint(self._log_dir) is not None:
            print("[*] Saved result exists! loading...")
            self.saver.restore(
                self.sess,
                latest_checkpoint
            )
            print("[*] Loaded previously trained weights")
            self.b_pretrain_loaded = True
        else:
            print("[*] No previous result")
            self.b_pretrain_loaded = False

        print("[*] Training starts...")
        self.model_summary_writer = None

        ##Training
        _epoch = 0
        while _epoch <= self._epochs:
            start_time = time.time()

            self._logger.info('Training epoch: {}/{}'.format(_epoch, self._epochs))

            train_data_generator = self._data['train_loader'].get_iterator()
            val_data_generator = self._data['val_loader'].get_iterator()

            losses = []

            res = self.run_epoch_generator(model=self._train_model,
                                           data_generator=train_data_generator)
            train_loss = res['loss']

            val_res = self.run_epoch_generator(model=self._val_model,
                                               data_generator=val_data_generator)
            val_loss = val_res['loss']
            end_time = time.time()
            message = 'Epoch [{}/{}] train_loss: {:f}, val_loss: {:f} {:.1f}s'.format(
                _epoch, self._epochs, train_loss, val_loss, (end_time - start_time))
            self._logger.info(message)

            if val_loss <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(val_loss)
                self._logger.info(
                    'Val loss decrease from %f to %f, saving to %s' % (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warning('Early stopping at epoch: %d' % _epoch)
                    break

    def _get_summary_writer(self, result):
        if result['step'] % self._logstep == 0:
            return self.summary_writer
        else:
            return None

    def test(self, sess):

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

        return

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

    def save(self, val_loss):
        config = dict(self._kwargs)
        global_step = sess.run(tf.train.get_or_create_global_step()).item()
        prefix = os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self.saver.save(self.sess, self._log_dir)

        config_filename = 'config_{}.yaml'.format(self._epoch)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']
