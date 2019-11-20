import os

import numpy as np
import yaml
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
from tqdm import tqdm

from Models.AbstractModel import AbstractModel, TimeHistory
from lib import utils


class lstm(AbstractModel):

    def __init__(self, is_training=False, **kwargs):
        super(lstm, self).__init__(**kwargs)

        self._rnn_units = self._model_kwargs.get('rnn_units')
        self._input_shape = (self._seq_len, self._input_dim)
        self._output_dim = self._model_kwargs.get('output_dim')
        self._nodes = self._model_kwargs.get('num_nodes')

        # Train's args
        self._batch_size = self._data_kwargs.get('batch_size')

        # Load data
        self._data = utils.load_dataset_lstm(seq_len=self._seq_len, horizon=self._horizon,
                                             input_dim=self._input_dim,
                                             mon_ratio=self._mon_ratio,
                                             scaler_type=self._kwargs.get('scaler'),
                                             is_training=is_training,
                                             **self._data_kwargs)

        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Model
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

    def seq2seq_model_construction(self):
        self.model = Sequential()
        self.model.add(LSTM(self._rnn_units, input_shape=self._input_shape, return_sequences=True))
        self.model.add(Dropout(self._drop_out))
        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Dense(64))
        self.model.add(Dense(32))
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def _ims_tm_prediction(self, init_data, init_labels):
        multi_steps_tm = np.zeros(shape=(init_data.shape[0] + self._horizon, self._nodes), dtype='float32')
        multi_steps_tm[0:self._seq_len] = init_data

        m_indicator = np.zeros(shape=(init_labels.shape[0] + self._horizon, self._nodes), dtype='float32')
        m_indicator[0:self._seq_len] = init_labels

        for ts_ahead in range(self._horizon):
            rnn_input = self._prepare_input_lstm(data=multi_steps_tm[ts_ahead:ts_ahead + self._seq_len],
                                                 m_indicator=m_indicator[ts_ahead:ts_ahead+self._seq_len])
            predictX = self.model.predict(rnn_input)
            multi_steps_tm[ts_ahead + self._seq_len] = np.squeeze(predictX, axis=1)

        return multi_steps_tm[-self._horizon:]

    def _run_tm_prediction(self, runId):

        test_data_norm = self._data['test_data_norm']

        tm_pred, m_indicator = self._init_data_test(test_data_norm, runId)

        y_preds = []
        y_truths = []

        # Predict the TM from time slot look_back
        for ts in tqdm(range(test_data_norm.shape[0] - self._horizon - self._seq_len)):

            predicted_tm = self._ims_tm_prediction(init_data=tm_pred[ts:ts + self._seq_len],
                                                   init_labels=m_indicator[ts:ts + self._seq_len])

            y_preds.append(np.expand_dims(predicted_tm, axis=0))
            y_truths.append(
                np.expand_dims(test_data_norm[ts + self._seq_len:ts + self._seq_len + self._horizon], axis=0))

            pred = predicted_tm[0]

            sampling = self._monitored_flows_slection(time_slot=ts, m_indicator=m_indicator)
            pred_input = pred * (1.0 - sampling)
            ground_true = test_data_norm[ts + self._seq_len].copy()
            new_input = pred_input + ground_true * sampling
            tm_pred[ts + self._seq_len] = new_input

        outputs = {
            'tm_pred': tm_pred[self._seq_len:],
            'm_indicator': m_indicator[self._seq_len:],
            'y_preds': y_preds,
            'y_truths': y_truths
        }

        return outputs

    def test(self):
        n_metrics = 4
        # Metrics: MSE, MAE, RMSE, MAPE, ER
        metrics_summary = np.zeros(shape=(self._run_times + 3, self._horizon * n_metrics + 1))

        for i in range(self._run_times):
            self._logger.info('|--- Running time: {}/{}'.format(i, self._run_times))

            test_results = self._run_tm_prediction(runId=i)

            metrics_summary = self._calculate_metrics(prediction_results=test_results, metrics_summary=metrics_summary,
                                                      scaler=self._data['scaler'],
                                                      runId=i, data_norm=self._data['test_data_norm'])

        self._summarize_results(metrics_summary=metrics_summary, n_metrics=n_metrics)

    def train(self):
        training_history = self.model.fit(x=self._data['x_train'],
                                          y=self._data['y_train'],
                                          batch_size=self._batch_size,
                                          epochs=self._epochs,
                                          callbacks=self.callbacks_list,
                                          validation_data=(self._data['x_val'], self._data['y_val']),
                                          shuffle=True,
                                          verbose=2)
        if training_history is not None:
            self.plot_training_history(training_history)
            self.save_model_history(times=self._time_callback.times, model_history=training_history)
            config = dict(self._kwargs)
            config_filename = 'config_lstm.yaml'
            config['train']['log_dir'] = self._log_dir
            with open(os.path.join(self._log_dir, config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    def evaluate(self):
        pass

    def load(self):
        self.model.load_weights(self._log_dir + 'best_model.hdf5')
