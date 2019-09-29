import os
import time

import keras.callbacks as keras_callbacks
import numpy as np
import pandas as pd
import yaml
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Input
from keras.models import Model

from Models.lstm_supervisor import lstm
from common.error_utils import error_ratio
from lib import utils, metrics


class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class EncoderDecoder(lstm):

    def __init__(self, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self._kwargs = kwargs

        # Load data
        self._data = utils.load_dataset_lstm_ed(seq_len=self._seq_len, horizon=self._horizon, input_dim=self._input_dim,
                                                mon_ratio=self._mon_ratio, test_size=self._test_size,
                                                **self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Model
        encoder_inputs = Input(shape=self._input_shape)
        encoder = LSTM(self._rnn_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self._horizon))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self._rnn_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self._horizon, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

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

        self.model = None

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

    def train(self):
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse, mae'])

        training_history = self.model.fit(x=[self._data['encode_input_train'], self._data['decode_input_train']],
                                          y=self._data['decode_target_train'],
                                          batch_size=self._batch_size,
                                          epochs=self._epochs,
                                          callbacks=self.callbacks_list,
                                          validation_data=([self._data['encode_input_val'],
                                                            self._data['decode_input_val']],
                                                           self._data['decode_target_val']),
                                          shuffle=True,
                                          verbose=2)
        if training_history is not None:
            self._plot_training_history(training_history)
            self._save_model_history(training_history)
            config = dict(self._kwargs)
            config_filename = 'config_lstm.yaml'
            config['train']['log_dir'] = self._log_dir
            with open(os.path.join(self._log_dir, config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        # evaluate
        scaler = self._data['scaler']
        x_eval = self._data['x_eval']
        y_truth = self._data['y_eval']

        y_pred = self.model.predict(x_eval)
        y_pred = scaler.inverse_transform(y_pred)
        y_truth = scaler.inverse_transform(y_truth)

        mse = metrics.masked_mse_np(preds=y_pred, labels=y_truth, null_val=0)
        mape = metrics.masked_mape_np(preds=y_pred, labels=y_truth, null_val=0)
        rmse = metrics.masked_rmse_np(preds=y_pred, labels=y_truth, null_val=0)
        self._logger.info(
            "Horizon {:02d}, MSE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                1, mse, mape, rmse
            )
        )

    def evaluate(self):
        scaler = self._data['scaler']

        y_pred = self.model.predict(self._data['x_eval'])
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

    def _save_model_history(self, model_history):
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

    def test(self):
        return self._test()

    def load(self):
        self.model.load_weights(self._log_dir + 'best_model.hdf5')
