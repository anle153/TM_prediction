import os

import numpy as np
import yaml
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Input, Concatenate, Reshape
from keras.models import Model
from tqdm import tqdm

from Models.AbstractModel import AbstractModel, TimeHistory
from lib import utils


class FwbwLstmRegression(AbstractModel):

    def __init__(self, is_training=False, **kwargs):
        super(FwbwLstmRegression, self).__init__(**kwargs)

        self._base_dir = kwargs.get('base_dir')

        # Model's Args
        self._output_dim = self._model_kwargs.get('output_dim')
        self._r = self._model_kwargs.get('r')

        # Train's args
        self._drop_out = self._train_kwargs.get('dropout')
        self._batch_size = self._data_kwargs.get('batch_size')

        # Test's args
        self._lamda = []
        self._lamda.append(self._test_kwargs.get('lamda_0'))
        self._lamda.append(self._test_kwargs.get('lamda_1'))
        self._lamda.append(self._test_kwargs.get('lamda_2'))

        # Load data
        self._data = utils.load_dataset_fwbw_lstm(is_training=is_training, seq_len=self._seq_len, horizon=self._horizon,
                                                  input_dim=self._input_dim,
                                                  mon_ratio=self._mon_ratio,
                                                  scaler_type=self._kwargs.get('scaler'),
                                                  **self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Model
        self.model = None

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

    def construct_fwbw_lstm(self):
        # Input
        input_tensor = Input(shape=(self._seq_len, self._input_dim), name='input')

        # Forward Network
        fw_lstm_layer = LSTM(self._rnn_units, input_shape=(self._seq_len, self._input_dim), return_sequences=True)(
            input_tensor)
        fw_outputs = Dropout(self._drop_out)(fw_lstm_layer)
        fw_outputs = Flatten()(fw_outputs)
        fw_outputs = Dense(128, )(fw_outputs)
        fw_outputs = Dense(64, )(fw_outputs)
        fw_outputs = Dense(32, )(fw_outputs)
        fw_outputs = Dense(1, name='fw_outputs')(fw_outputs)

        # Backward Network
        bw_lstm_layer = LSTM(self._rnn_units, input_shape=(self._seq_len, self._input_dim),
                             return_sequences=True, go_backwards=True)(input_tensor)
        bw_drop_out = Dropout(self._drop_out)(bw_lstm_layer)
        bw_flat_layer = TimeDistributed(Flatten())(bw_drop_out)
        bw_dense_1 = TimeDistributed(Dense(128, ))(bw_flat_layer)
        bw_dense_2 = TimeDistributed(Dense(64, ))(bw_dense_1)
        bw_dense_3 = TimeDistributed(Dense(32, ))(bw_dense_2)
        bw_output = TimeDistributed(Dense(1, ))(bw_dense_3)

        bw_output = Reshape(target_shape=(self._seq_len, 1))(bw_output)

        # bw_input_tensor_flatten = Reshape((self._seq_len * self._input_dim, 1))(input_tensor)
        _input_bw = Concatenate(axis=-1)([input_tensor, bw_output])

        _input_bw = Flatten()(_input_bw)
        _input_bw = Dense(256, )(_input_bw)
        _input_bw = Dropout(0.5)(_input_bw)
        _input_bw = Dense(128, )(_input_bw)
        _input_bw = Dropout(0.5)(_input_bw)
        bw_outputs = Dense(self._seq_len, name='bw_outputs')(_input_bw)

        self.model = Model(inputs=input_tensor, outputs=[fw_outputs, bw_outputs], name='fwbw-lstm')

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'], loss_weights=[1, 0.1])

    def _ims_tm_prediction(self, init_data, init_labels):
        multi_steps_tm = np.zeros(shape=(init_data.shape[0] + self._horizon, init_data.shape[1]),
                                  dtype='float32')
        multi_steps_tm[0:self._seq_len] = init_data

        m_indicator = np.zeros(shape=(init_labels.shape[0] + self._horizon, init_labels.shape[1]),
                               dtype='float32')
        m_indicator[0:self._seq_len] = init_labels

        bw_outputs = None

        for ts_ahead in range(self._horizon):
            rnn_input = self._prepare_input_lstm(data=multi_steps_tm[ts_ahead:ts_ahead + self._seq_len],
                                                 m_indicator=m_indicator[ts_ahead:ts_ahead + self._seq_len])
            predictX, predictX_2 = self.model.predict(rnn_input)
            multi_steps_tm[ts_ahead + self._seq_len] = np.squeeze(predictX, axis=1)

            if ts_ahead == 0:
                bw_outputs = predictX_2.copy()

        return multi_steps_tm[-self._horizon:], bw_outputs

    def _run_tm_prediction(self, runId):
        test_data_norm = self._data['test_data_norm']
        tm_pred, m_indicator = self._init_data_test(test_data_norm, runId)
        y_preds = []
        y_truths = []

        _last_err = 1.0
        # Predict the TM from time slot look_back
        for ts in tqdm(range(test_data_norm.shape[0] - self._horizon - self._seq_len)):
            # This block is used for iterated multi-step traffic matrices prediction

            # fw_outputs (horizon, num_flows); bw_outputs (num_flows, seq_len)
            fw_outputs, bw_outputs = self._ims_tm_prediction(init_data=tm_pred[ts:ts + self._seq_len],
                                                             init_labels=m_indicator[ts:ts + self._seq_len])

            # Get the TM prediction of next time slot
            # corrected_data = self._data_correction_v3(rnn_input=tm_pred[ts: ts + self._seq_len],
            #                                           pred_backward=bw_outputs,
            #                                           labels=m_indicator[ts: ts + self._seq_len], r=self._r)
            # measured_data = tm_pred[ts:ts + self._seq_len - 1] * m_indicator[ts:ts + self._seq_len - 1]
            # pred_data = corrected_data * (1.0 - m_indicator[ts:ts + self._seq_len - 1])
            # tm_pred[ts:ts + self._seq_len - 1] = measured_data + pred_data

            # test bw correction
            bw_outputs = bw_outputs.T
            _corr_data = bw_outputs[1:]

            _pred_err = self._calculate_pred_err(pred=_corr_data.copy(), tm=tm_pred[ts:ts + self._seq_len - 1].copy(),
                                                 m_indicator=m_indicator[ts:ts + self._seq_len - 1].copy())
            if _pred_err < _last_err:
                _measured_data = tm_pred[ts:ts + self._seq_len - 1] * m_indicator[ts:ts + self._seq_len - 1]
                _corr_data = _corr_data * (1.0 - m_indicator[ts:ts + self._seq_len - 1])
                tm_pred[ts:ts + self._seq_len - 1] = _measured_data + _corr_data

            _last_err = _pred_err

            y_preds.append(np.expand_dims(fw_outputs, axis=0))
            pred = fw_outputs[0]

            # Using part of current prediction as input to the next estimation
            # Randomly choose the flows which is measured (using the correct data from test_set)

            # boolean array(1 x n_flows):for choosing value from predicted data
            sampling = self._monitored_flows_slection(time_slot=ts, tm_pred=tm_pred, m_indicator=m_indicator,
                                                      fw_outputs=fw_outputs, lamda=self._lamda)

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

        outputs = {
            'tm_pred': tm_pred[self._seq_len:],
            'm_indicator': m_indicator[self._seq_len:],
            'y_preds': y_preds,
            'y_truths': y_truths
        }
        return outputs

    def train(self):
        training_fw_history = self.model.fit(x=self._data['x_train'],
                                             y=[self._data['y_train_1'], self._data['y_train_2']],
                                             batch_size=self._batch_size,
                                             epochs=self._epochs,
                                             callbacks=self.callbacks_list,
                                             validation_data=(self._data['x_val'],
                                                              [self._data['y_val_1'], self._data['y_val_2']]),
                                             shuffle=True,
                                             verbose=2)
        if training_fw_history is not None:
            self.plot_training_history(training_fw_history)
            self.save_model_history(times=self._time_callback.times, model_history=training_fw_history)
            config = dict(self._kwargs)
            config_filename = 'config_fwbw_lstm.yaml'
            config['train']['log_dir'] = self._log_dir
            with open(os.path.join(self._log_dir, config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    def evaluate(self):
        pass

    def test(self):
        n_metrics = 4
        # Metrics: MSE, MAE, RMSE, MAPE, ER
        metrics_summary = np.zeros(shape=(self._run_times + 3, self._horizon * n_metrics + 1))
        for i in range(self._run_times):
            print('|--- Running time: {}/{}'.format(i, self._run_times))

            test_results = self._run_tm_prediction(runId=i)

            metrics_summary = self._calculate_metrics(prediction_results=test_results, metrics_summary=metrics_summary,
                                                      scaler=self._data['scaler'],
                                                      runId=i, data_norm=self._data['test_data_norm'])

        self._summarize_results(metrics_summary=metrics_summary, n_metrics=n_metrics)

    def load(self):
        self.model.load_weights(self._log_dir + 'best_model.hdf5')
