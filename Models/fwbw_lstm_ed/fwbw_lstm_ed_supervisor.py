import os
import time

import keras.callbacks as keras_callbacks
import numpy as np
import pandas as pd
import yaml
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Input
from keras.models import Model
from keras.utils import plot_model
from tqdm import tqdm

from lib import metrics
from lib import utils


class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class FwbwLstmED():

    def __init__(self, is_training=True, **kwargs):

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
        self._logger.info(kwargs)

        self._mon_ratio = self._kwargs.get('mon_ratio')
        _scaler_type = self._kwargs.get('scaler')

        # Data's args
        self._day_size = self._data_kwargs.get('day_size')

        # Model's Args
        self._hidden = self._model_kwargs.get('rnn_units')
        self._seq_len = self._model_kwargs.get('seq_len')
        self._horizon = self._model_kwargs.get('horizon')
        self._input_dim = self._model_kwargs.get('input_dim')
        self._input_shape = (self._seq_len, self._input_dim)
        self._output_dim = self._model_kwargs.get('output_dim')
        self._nodes = self._model_kwargs.get('num_nodes')
        self._r = self._model_kwargs.get('r')

        # Train's args
        self._drop_out = self._train_kwargs.get('dropout')
        self._epochs = self._train_kwargs.get('epochs')
        self._batch_size = self._data_kwargs.get('batch_size')

        # Test's args
        self._run_times = self._test_kwargs.get('run_times')
        self._flow_selection = self._test_kwargs.get('flow_selection')
        self._lamda = []
        self._lamda.append(self._test_kwargs.get('lamda_0'))
        self._lamda.append(self._test_kwargs.get('lamda_1'))
        self._lamda.append(self._test_kwargs.get('lamda_2'))


        # Load data
        self._data = utils.load_dataset_fwbw_lstm_ed(seq_len=self._seq_len, horizon=self._horizon,
                                                     input_dim=self._input_dim,
                                                     mon_ratio=self._mon_ratio,
                                                     scaler_type=_scaler_type,
                                                     **self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Model
        self.model = self.construct_fwbw_lstm_ed(is_training=is_training)

        self.callbacks_list = []

        self._checkpoints = ModelCheckpoint(
            self._log_dir + "best_model.hdf5",
            monitor='val_loss', verbose=1,
            save_best_only=True,
            mode='auto', period=1)
        self.callbacks_list = [self._checkpoints]

        self._earlystop = EarlyStopping(monitor='val_loss', patience=self._train_kwargs.get('patience'),
                                        verbose=1, mode='auto')
        self.callbacks_list.append(self._earlystop)

        self._time_callback = TimeHistory()
        self.callbacks_list.append(self._time_callback)

    def save_model_history(self, model_history):
        loss = np.array(model_history.history['loss'])
        val_loss = np.array(model_history.history['val_loss'])
        dump_model_history = pd.DataFrame(index=range(loss.size),
                                          columns=['epoch', 'loss', 'val_loss', 'train_time'])

        dump_model_history['epoch'] = range(loss.size)
        dump_model_history['loss'] = loss
        dump_model_history['val_loss'] = val_loss

        if self._time_callback.times is not None:
            dump_model_history['train_time'] = self._time_callback.times

        dump_model_history.to_csv(self._log_dir + 'training_history.csv', index=False)

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            rnn_units = kwargs['model'].get('rnn_units')
            horizon = kwargs['model'].get('horizon')
            mon_r = kwargs['mon_ratio']
            scaler = kwargs['scaler']
            run_id = 'fwbw_lstm_ed_%d_%g_%d_%d_%s/' % (rnn_units, mon_r, horizon, batch_size, scaler)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def construct_fwbw_lstm_ed(self, is_training=True):
        encoder_inputs = Input(shape=(None, self._input_dim))

        # encoder fw
        encoder = LSTM(self._hidden, return_state=True, name='encoder-fw')
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # encoder bw
        encoder_bw = LSTM(self._hidden, return_sequences=True, go_backwards=True)
        encoder_outputs_bw = encoder_bw(encoder_inputs)

        encoder_outputs_bw = TimeDistributed(Flatten())(encoder_outputs_bw)
        encoder_outputs_bw = TimeDistributed(Dense(128, activation='relu'))(encoder_outputs_bw)
        encoder_outputs_bw = Dropout(self._drop_out)(encoder_outputs_bw)
        encoder_outputs_bw = TimeDistributed(Dense(64, activation='relu'))(encoder_outputs_bw)
        encoder_outputs_bw = Dropout(self._drop_out)(encoder_outputs_bw)
        encoder_outputs_bw = TimeDistributed(Dense(1, activation='relu'), name='encoder_bw')(encoder_outputs_bw)
        #
        # seq_len = encoder_inputs.get_shape()[1].value
        #
        # bw_input_tensor_flatten = Reshape((-1, seq_len * self._input_dim, 1))(encoder_inputs)
        # _input_bw = Concatenate(axis=1)([bw_input_tensor_flatten, bw_output])
        #
        # _input_bw = Flatten()(_input_bw)
        # _input_bw = Dense(256, )(_input_bw)
        # _input_bw = Dense(128, )(_input_bw)
        # en_outputs_bw = Dense(self._seq_len, name='bw_outputs')(_input_bw)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, 1))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self._hidden, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)

        decoder_dense = Dense(1, activation='relu', name='decoder')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], [encoder_outputs_bw, decoder_outputs], name='fwbw-lstm-ed')

        if is_training:
            model.compile(optimizer='adam', loss='mse')
            return model
        else:
            self._logger.info("|--- Load model from: {}".format(self._log_dir))
            model.load_weights(self._log_dir + 'best_model.hdf5')
            model.compile(optimizer='adam', loss='mse')

            # Construct E_D model for predicting
            self.encoder_model = Model(encoder_inputs, encoder_states)
            self.encoder_model_bw = Model(encoder_inputs, encoder_outputs_bw)

            decoder_state_input_h = Input(shape=(self._hidden,))
            decoder_state_input_c = Input(shape=(self._hidden,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_outputs, state_h, state_c = decoder_lstm(
                decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
            decoder_outputs = decoder_dense(decoder_outputs)
            self.decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)

            plot_model(model=self.encoder_model, to_file=self._log_dir + '/encoder.png', show_shapes=True)
            plot_model(model=self.encoder_model_bw, to_file=self._log_dir + '/encoder_bw.png', show_shapes=True)
            plot_model(model=self.decoder_model, to_file=self._log_dir + '/decoder.png', show_shapes=True)

            return model

    def plot_models(self):
        plot_model(model=self.model, to_file=self._log_dir + '/model.png', show_shapes=True)

    def plot_training_history(self, model_history):
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

        dataX = np.zeros(shape=(self._nodes, self._seq_len, self._input_dim), dtype='float32')
        for flow_id in range(self._nodes):
            x = data[:, flow_id]
            label = m_indicator[:, flow_id]

            dataX[flow_id, :, 0] = x
            dataX[flow_id, :, 1] = label

        return dataX

    def _predict(self, inputs):

        states_value = self.encoder_model.predict(inputs)
        bw_outputs = self.encoder_model_bw.predict(inputs)  # shape (nodes, seq_len, 1)

        target_seq = np.zeros((self._nodes, 1, 1))
        target_seq[:, 0, 0] = [0] * self._nodes

        multi_steps_tm = np.zeros(shape=(self._horizon + 1, self._nodes),
                                  dtype='float32')

        for ts_ahead in range(self._horizon + 1):
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            output_tokens = output_tokens[:, -1, 0]

            multi_steps_tm[ts_ahead] = output_tokens

            target_seq = np.zeros((self._nodes, 1, 1))
            target_seq[:, 0, 0] = output_tokens

            # Update states
            states_value = [h, c]

        return multi_steps_tm[-self._horizon:], np.squeeze(bw_outputs, axis=-1)  # shape (nodes, seq_len)

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

    def _calculate_flows_weights(self, fw_losses, m_indicator):
        """

        :param fw_losses: shape(#n_flows)
        :param m_indicator: shape(#seq_len, #nflows)
        :return: w: flow_weight shape(#n_flows)
        """

        cl = self._calculate_consecutive_loss(m_indicator)

        w = 1 / (fw_losses * self._lamda[0] +
                 cl * self._lamda[1])

        return w

    def _set_measured_flow(self, rnn_input, pred_forward, m_indicator):
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
                                          m_indicator=m_indicator)

        sampling = np.zeros(shape=n_flows)
        m = int(self._mon_ratio * n_flows)

        w = w.flatten()
        sorted_idx_w = np.argsort(w)
        sampling[sorted_idx_w[:m]] = 1

        return sampling

    def _data_correction_v3(self, rnn_input, pred_backward, labels):
        # Shape = (#n_flows, #time-steps)
        _rnn_input = np.copy(rnn_input.T)
        _labels = np.copy(labels.T)

        beta = np.zeros(_rnn_input.shape)

        corrected_range = int(self._seq_len / self._r)

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

        pred_bw = []
        gt_bw = []

        # Predict the TM from time slot look_back
        for ts in tqdm(range(test_data_norm.shape[0] - self._horizon - self._seq_len)):
            # This block is used for iterated multi-step traffic matrices prediction

            inputs = self._prepare_input(data=tm_pred[ts:ts + self._seq_len],
                                         m_indicator=m_indicator[ts:ts + self._seq_len])

            # fw_outputs (horizon, num_flows); bw_outputs (num_flows, seq_len)
            fw_outputs, bw_outputs = self._predict(inputs)

            if ts > 0:
                pred_bw.append(bw_outputs.T)
                gt_bw.append(test_data_norm[ts - 1:ts + self._seq_len - 1])

            # Get the TM prediction of next time slot
            # corrected_data = self._data_correction_v3(rnn_input=tm_pred[ts: ts + self._seq_len],
            #                                           pred_backward=bw_outputs,
            #                                           labels=m_indicator[ts: ts + self._seq_len])
            # measured_data = tm_pred[ts:ts + self._seq_len - 1] * m_indicator[ts:ts + self._seq_len - 1]
            # pred_data = corrected_data * (1.0 - m_indicator[ts:ts + self._seq_len - 1])
            # tm_pred[ts:ts + self._seq_len - 1] = measured_data + pred_data

            y_preds.append(np.expand_dims(fw_outputs, axis=0))
            pred = fw_outputs[0]

            # Using part of current prediction as input to the next estimation
            # Randomly choose the flows which is measured (using the correct data from test_set)

            # boolean array(1 x n_flows):for choosing value from predicted data
            if self._flow_selection == 'Random':
                sampling = np.random.choice(tf_a, size=self._nodes,
                                            p=[self._mon_ratio, 1 - self._mon_ratio])
            elif self._flow_selection == 'Fairness':
                sampling = self._set_measured_flow_fairness(m_indicator=m_indicator[ts: ts + self._seq_len])
            else:
                sampling = self._set_measured_flow(rnn_input=tm_pred[ts: ts + self._seq_len],
                                                   pred_forward=fw_outputs,
                                                   m_indicator=m_indicator[ts: ts + self._seq_len].T)

            m_indicator[ts + self._seq_len] = sampling
            # invert of sampling: for choosing value from the original data
            pred_input = pred * (1.0 - sampling)

            ground_truth = test_data_norm[ts + self._seq_len]
            y_truths.append(
                np.expand_dims(test_data_norm[ts + self._seq_len:ts + self._seq_len + self._horizon], axis=0))

            measured_input = ground_truth * sampling

            # Merge value from pred_input and measured_input
            new_input = pred_input + measured_input
            # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

            # Concatenating new_input into current rnn_input
            tm_pred[ts + self._seq_len] = new_input

        pred_bw = np.stack(pred_bw)
        gt_bw = np.stack(gt_bw)

        outputs = {
            'tm_pred': tm_pred[self._seq_len:],
            'm_indicator': m_indicator[self._seq_len:],
            'y_preds': y_preds,
            'y_truths': y_truths,
            'pred_bw': pred_bw.reshape((pred_bw.shape[0] * pred_bw.shape[1], pred_bw.shape[2])),
            'gt_bw': gt_bw.reshape((gt_bw.shape[0] * gt_bw.shape[1], gt_bw.shape[2]))
        }
        return outputs

    def train(self):
        training_fw_history = self.model.fit(x=[self._data['inputs_train'], self._data['dec_inputs_train']],
                                             y=[self._data['enc_labels_bw_train'], self._data['dec_labels_train']],
                                             batch_size=self._batch_size,
                                             epochs=self._epochs,
                                             callbacks=self.callbacks_list,
                                             validation_data=([self._data['inputs_val'],
                                                               self._data['dec_inputs_val']],
                                                              [self._data['enc_labels_bw_val'],
                                                               self._data['dec_labels_val']]),
                                             shuffle=True,
                                             verbose=2)
        if training_fw_history is not None:
            self.plot_training_history(training_fw_history)
            self.save_model_history(training_fw_history)
            config = dict(self._kwargs)
            config_filename = 'config_fwbw_lstm_ed.yaml'
            config['train']['log_dir'] = self._log_dir
            with open(os.path.join(self._log_dir, config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    def evaluate(self):
        pass

    def test(self):
        scaler = self._data['scaler']
        results_summary = pd.DataFrame(index=range(self._run_times + 3))
        results_summary['No.'] = range(self._run_times + 3)

        n_metrics = 4
        # Metrics: MSE, MAE, RMSE, MAPE, ER
        metrics_summary = np.zeros(shape=(self._run_times + 3, self._horizon * n_metrics + 1))
        for i in range(self._run_times):
            print('|--- Running time: {}/{}'.format(i, self._run_times))

            outputs = self._run_tm_prediction()

            tm_pred, m_indicator, y_preds = outputs['tm_pred'], outputs['m_indicator'], outputs['y_preds']

            y_preds = np.concatenate(y_preds, axis=0)
            predictions = []
            y_truths = outputs['y_truths']
            y_truths = np.concatenate(y_truths, axis=0)

            pred_bw, gt_bw = outputs['pred_bw'], outputs['gt_bw']
            pred_bw = scaler.inverse_transform(pred_bw)
            gt_bw = scaler.inverse_transform(gt_bw)
            self._logger.info('RMSE BW: {}'.format(metrics.masked_rmse_np(pred_bw.flatten(), gt_bw.flatten())))

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

            er = metrics.error_ratio(y_pred=tm_pred,
                                     y_true=g_truth,
                                     measured_matrix=m_indicator)
            metrics_summary[i, -1] = er

            self._save_results(g_truth=g_truth, pred_tm=tm_pred, m_indicator=m_indicator, tag=str(i))
            print('ER: {}'.format(er))

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

    def _save_results(self, g_truth, pred_tm, m_indicator, tag):
        np.save(self._log_dir + '/g_truth{}'.format(tag), g_truth)
        np.save(self._log_dir + '/pred_tm_{}'.format(tag), pred_tm)
        np.save(self._log_dir + '/m_indicator{}'.format(tag), m_indicator)

    def load(self):
        self.model.load_weights(self._log_dir + 'best_model.hdf5')
