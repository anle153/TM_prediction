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
from Models.dcrnn_fwbw.dcrnn_fwbw_model import DCRNNModel
from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import masked_mse_loss


class DCRNNSupervisor(AbstractModel):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, **kwargs):
        super(DCRNNSupervisor, self).__init__(**kwargs)

        self._r = int(self._model_kwargs.get('r'))
        self._lamda = []
        self._lamda.append(self._test_kwargs.get('lamda_0'))
        self._lamda.append(self._test_kwargs.get('lamda_1'))
        self._lamda.append(self._test_kwargs.get('lamda_2'))

        # Data preparation
        self._day_size = self._data_kwargs.get('day_size')
        self._data = utils.load_dataset_dcrnn_fwbw(seq_len=self._model_kwargs.get('seq_len'),
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
            with tf.variable_scope('DCRNN', reuse=False):
                self.train_model = DCRNNModel(is_training=True, scaler=scaler,
                                              batch_size=self._data_kwargs['batch_size'],
                                              adj_mx=self._data['adj_mx'], **self._model_kwargs)

        with tf.name_scope('Val'):
            with tf.variable_scope('DCRNN', reuse=True):
                self.val_model = DCRNNModel(is_training=False, scaler=scaler,
                                            batch_size=self._data_kwargs['val_batch_size'],
                                            adj_mx=self._data['adj_mx'], **self._model_kwargs)

        with tf.name_scope('Eval'):
            with tf.variable_scope('DCRNN', reuse=True):
                self.eval_model = DCRNNModel(is_training=False, scaler=scaler,
                                             batch_size=self._data_kwargs['eval_batch_size'],
                                             adj_mx=self._data['adj_mx'], **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                self.test_model = DCRNNModel(is_training=False, scaler=scaler,
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
        # fw decoder
        preds_fw = self.train_model.outputs_fw
        labels_fw = self.train_model.labels_fw[..., :output_dim]

        # bw encoder
        enc_preds_bw = self.train_model.enc_outputs_bw
        enc_labels_bw = self.train_model.enc_labels_bw[..., :output_dim]

        null_val = 0.
        self._loss_fn = masked_mse_loss(scaler, null_val)
        self._train_loss_dec = self._loss_fn(preds=preds_fw, labels=labels_fw)

        # backward loss
        self._train_loss_enc_bw = self._loss_fn(preds=enc_preds_bw, labels=enc_labels_bw)

        self._train_loss = self._train_loss_dec + self._train_loss_enc_bw

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
        dec_losses_fw = []
        enc_losses_bw = []
        outputs = []
        output_dim = self._model_kwargs.get('output_dim')

        preds_fw = model.outputs_fw
        labels_fw = model.labels_fw[..., :output_dim]
        loss = self._loss_fn(preds=preds_fw, labels=labels_fw)

        enc_preds_bw = model.enc_outputs_bw
        enc_labels_bw = model.enc_labels_bw[..., :output_dim]
        enc_loss_bw = self._loss_fn(preds=enc_preds_bw, labels=enc_labels_bw)

        fetches = {
            'loss': loss,
            'enc_loss_bw': enc_loss_bw,
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
                'outputs': model.outputs_fw
            })

        for _, (_inputs, _dec_labels_fw, _enc_labels_bw) in enumerate(data_generator):
            feed_dict = {
                model.inputs: _inputs,
                model.labels_fw: _dec_labels_fw,
                model.enc_labels_bw: _enc_labels_bw,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'] + vals['enc_loss_bw'])
            dec_losses_fw.append(vals['loss'])
            enc_losses_bw.append(vals['enc_loss_bw'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])

        results = {
            'loss': np.mean(losses),
            'enc_loss_bw': np.mean(enc_losses_bw),
            'dec_loss_fw': np.mean(dec_losses_fw),
        }
        if return_output:
            results['outputs'] = outputs
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
            # Learning rate schedule.
            new_lr = max(min_learning_rate, base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, self.train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     training=True,
                                                     writer=self._writer)
            train_loss, train_enc_loss_bw = train_results['loss'], train_results['enc_loss_bw']
            train_dec_loss_fw = train_results['dec_loss_fw']
            # if train_loss > 1e5:
            #     self._logger.warning('Gradient explosion detected. Ending...')
            #     break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = self.run_epoch_generator(sess, self.val_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   training=False)
            val_loss, val_enc_loss_bw = val_results['loss'].item(), val_results['enc_loss_bw'].item()
            val_dec_loss_fw = val_results['dec_loss_fw'].item()

            utils.add_simple_summary(self._writer,
                                     ['loss/train_loss', 'loss/val_loss'],
                                     [train_loss, val_loss], global_step=global_step)
            end_time = time.time()
            message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f} || train_dec_fw: {:.4f}, val_dec_fw: {:.4f}, train_enc_bw: {:.4f}, val_enc_bw: {:.4f} - lr:{:.6f} {:.1f}s'.format(
                self._epoch, epochs, train_loss, val_loss, train_dec_loss_fw, val_dec_loss_fw, train_enc_loss_bw,
                val_enc_loss_bw, new_lr, (end_time - start_time))
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

            # Increases epoch.
            self._epoch += 1
            losses.append(train_loss)
            val_losses.append(val_loss)
            sys.stdout.flush()

        training_history['epoch'] = np.arange(self._epoch)
        training_history['loss'] = losses
        training_history['val_loss'] = val_losses
        training_history.to_csv(self._log_dir + 'training_history.csv', index=False)

        return

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def evaluate(self, sess, **kwargs):
        global_step = sess.run(tf.train.get_or_create_global_step())
        test_results = self.run_epoch_generator(sess, self.eval_model,
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
        for horizon_i in range(self._data['dec_labels_fw_eval'].shape[1]):
            y_truth = scaler.inverse_transform(self._data['dec_labels_fw_eval'][:, horizon_i, :, 0])
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

    def test(self, sess):
        scaler = self._data['scaler']

        results_summary = pd.DataFrame(index=range(self._run_times + 3))
        results_summary['No.'] = range(self._run_times + 3)

        n_metrics = 4
        # Metrics: MSE, MAE, RMSE, MAPE, ER
        metrics_summary = np.zeros(shape=(self._run_times + 3, self._horizon * n_metrics + 1))

        for i in range(self._run_times):
            # y_test = self._prepare_test_set()
            test_results = self._run_tm_prediction(sess, runId=i)

            self._calculate_metrics(prediction_results=test_results, metrics_summary=metrics_summary,
                                    scaler=self._data['scaler'],
                                    runId=i, data_norm=self._data['test_data_norm'])

        self._summarize_results(metrics_summary=metrics_summary, n_metrics=n_metrics)

    def _run_tm_prediction(self, sess, runId):

        test_data_norm = self._data['test_data_norm']
        tm_pred, m_indicator = self._init_data_test(test_data_norm, runId)

        y_truths = []
        y_preds = []
        fetches = {
            'global_step': tf.train.get_or_create_global_step()
        }
        fetches.update({
            'outputs_fw': self.test_model.outputs_fw,
            'enc_outputs_bw': self.test_model.enc_outputs_bw
        })

        for ts in tqdm(range(test_data_norm.shape[0] - self._horizon - self._seq_len)):

            # inputs, dec_labels, enc_labels, dec_labels_bw, enc_labels_bw
            x = self._prepare_input_dcrnn(
                data=tm_pred[ts:ts + self._seq_len],
                m_indicator=m_indicator[ts:ts + self._seq_len]
            )

            feed_dict = {
                self.test_model.inputs: x,
            }
            vals = sess.run(fetches, feed_dict=feed_dict)

            # encoder_outputs_bw (1, seq_len, num_node, 1), decoder_outputs_fw (1, horizon, num_node, 1)
            # decoder_outputs_fw (ts + seq_len, ts + seq_len + h)
            encoder_outputs_bw, decoder_outputs_fw = vals['enc_outputs_bw'], vals['outputs_fw']

            decoder_outputs_fw = np.squeeze(decoder_outputs_fw, axis=-1)

            encoder_outputs_bw = np.squeeze(encoder_outputs_bw, axis=0)
            encoder_outputs_bw = np.squeeze(encoder_outputs_bw,
                                            axis=-1)  # encoder_outputs_bw (ts - 1, ts + seq_len - 1)
            # encoder_outputs_bw = encoder_outputs_bw.T

            # corrected_data = self._data_correction_v3(rnn_input=tm_pred[ts: ts + self._seq_len],
            #                                           pred_backward=encoder_outputs_bw,
            #                                           labels=m_indicator[ts: ts + self._seq_len])
            # measured_data = tm_pred[ts:ts + self._seq_len - 1] * m_indicator[ts:ts + self._seq_len - 1]
            # pred_data = corrected_data * (1.0 - m_indicator[ts:ts + self._seq_len - 1])
            # tm_pred[ts:ts + self._seq_len - 1] = measured_data + pred_data

            _corr_data = encoder_outputs_bw * (1.0 - m_indicator[ts:ts + self._seq_len])
            _measured_data = tm_pred[ts:ts + self._seq_len] * m_indicator[ts:ts + self._seq_len]
            tm_pred[ts:ts + self._seq_len] = _measured_data + _corr_data

            y_preds.append(decoder_outputs_fw)
            y_truths.append(
                np.expand_dims(test_data_norm[ts + self._seq_len:ts + self._seq_len + self._horizon].copy(), axis=0))

            decoder_outputs_fw = np.squeeze(decoder_outputs_fw, axis=0)
            pred = decoder_outputs_fw[0]

            # Using part of current prediction as input to the next estimation
            # Randomly choose the flows which is measured (using the correct data from test_set)

            # boolean array(1 x n_flows):for choosing value from predicted data
            sampling = self._monitored_flows_slection(time_slot=ts, tm_pred=tm_pred, m_indicator=m_indicator,
                                                      fw_outputs=decoder_outputs_fw, lamda=self._lamda)

            ground_truth = test_data_norm[ts + self._seq_len].copy()
            # Concatenating new_input into current rnn_input
            tm_pred[ts + self._seq_len] = pred * (1.0 - sampling) + ground_truth * sampling

        outputs = {
            'tm_pred': tm_pred[self._seq_len:],
            'm_indicator': m_indicator[self._seq_len:],
            'y_preds': y_preds,
            'y_truths': y_truths
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
