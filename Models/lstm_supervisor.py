import os
import time

import keras.callbacks as keras_callbacks
import numpy as np
import pandas as pd
import yaml
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Input, Concatenate, Flatten, Reshape, \
    Add
from keras.models import Sequential, Model
from keras.utils import plot_model
from tqdm import tqdm

from common.error_utils import error_ratio
from lib import utils, metrics


class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class lstm():

    def __init__(self, **kwargs):

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

        # Data's args
        self._day_size = self._data_kwargs.get('day_size')

        # Model's Args
        self._model_type = self._model_kwargs.get('model')
        self._rnn_units = self._model_kwargs.get('rnn_units')
        self._seq_len = self._model_kwargs.get('seq_len')
        self._horizon = self._model_kwargs.get('horizon')
        self._input_dim = self._model_kwargs.get('input_dim')
        self._input_shape = (self._seq_len, self._input_dim)
        self._output_dim = self._model_kwargs.get('output_dim')
        self._nodes = self._model_kwargs.get('num_nodes')
        self._n_rnn_layers = self._model_kwargs.get('n_rnn_layers')

        # Train's args
        self._drop_out = self._train_kwargs.get('dropout')
        self._epochs = self._train_kwargs.get('epochs')
        self._batch_size = self._data_kwargs.get('batch_size')

        # Test's args
        self._run_times = self._test_kwargs.get('run_times')
        self._flow_selection = self._test_kwargs.get('flow_selection')
        self._test_size = self._test_kwargs.get('test_size')
        self._results_path = self._test_kwargs.get('results_path')

        self._mon_ratio = self._kwargs.get('mon_ratio')

        # Load data
        if self._model_type == 'lstm':
            self._data = utils.load_dataset_lstm(seq_len=self._seq_len, horizon=self._horizon,
                                                 input_dim=self._input_dim,
                                                 mon_ratio=self._mon_ratio, test_size=self._test_size,
                                                 **self._data_kwargs)
        elif self._model_type == 'ed' or self._model_type == 'encoder_decoder':
            self._data = utils.load_dataset_lstm_ed(seq_len=self._seq_len, horizon=self._horizon,
                                                    input_dim=self._input_dim,
                                                    mon_ratio=self._mon_ratio, test_size=self._test_size,
                                                    **self._data_kwargs)
        else:
            raise RuntimeError("Model must be lstm or encoder_decoder")

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


    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            n_rnn_layers = kwargs['model'].get('n_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(n_rnn_layers)])
            horizon = kwargs['model'].get('horizon')

            mon_ratio = kwargs['mon_ratio']

            run_id = 'lstm_%g_%d_%s_%g_%d_%s/' % (
                mon_ratio, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def normal_model_contruction(self):
        self.model = Sequential()
        self.model.add(LSTM(self._rnn_units, input_shape=self._input_shape))
        self.model.add(Dropout(self._drop_out))
        self.model.add(Dense(1))

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

    def encoder_decoder_tf(self):

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

    def res_lstm_construction(self):

        input_tensor = Input(shape=self._input_shape, name='input')

        # res lstm network
        lstm_layer = LSTM(self._rnn_units, input_shape=self._input_shape, return_sequences=True)(input_tensor)
        drop_out = Dropout(self._drop_out)(lstm_layer)
        flat_layer = TimeDistributed(Flatten())(drop_out)
        dense_1 = TimeDistributed(Dense(64, ))(flat_layer)
        dense_2 = TimeDistributed(Dense(32, ))(dense_1)
        output = TimeDistributed(Dense(1, ))(dense_2)

        input_tensor_flatten = Reshape((self._input_shape[0] * self._input_shape[1], 1))(input_tensor)
        _input = Concatenate(axis=1)([input_tensor_flatten, output])

        _input = Flatten()(_input)
        _input = Dense(64, )(_input)
        _input = Dense(32, )(_input)
        outputs = Dense(1, name='outputs')(_input)

        self.model = Model(inputs=input_tensor, outputs=outputs, name='res-lstm')
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def res_lstm_2_construction(self):

        input_tensor = Input(shape=self._input_shape, name='input')
        input_tensor_2 = Input(shape=(self._seq_len, 1), name='input_2')
        # res lstm network
        lstm_layer = LSTM(self._rnn_units, input_shape=self._input_shape, return_sequences=True)(input_tensor)
        drop_out = Dropout(self._drop_out)(lstm_layer)
        flat_layer = TimeDistributed(Flatten())(drop_out)
        dense_1 = TimeDistributed(Dense(64, ))(flat_layer)
        dense_2 = TimeDistributed(Dense(32, ))(dense_1)
        output = TimeDistributed(Dense(1, ))(dense_2)

        # input_tensor_flatten = Reshape((self.input_shape[0] * self.input_shape[1], 1))(input_tensor)
        _input = Add()([input_tensor_2, output])

        _input = Flatten()(_input)
        _input = Dense(64, )(_input)
        _input = Dense(32, )(_input)
        outputs = Dense(1, name='outputs')(_input)

        self.model = Model(inputs=[input_tensor, input_tensor_2], outputs=outputs, name='res-lstm')
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def seq2seq_deep_model_construction(self, n_layers):
        self.model = Sequential()
        for layer in range(n_layers):

            if layer != (n_layers - 1):
                self.model.add(LSTM(self._rnn_units, input_shape=self._input_shape, return_sequences=True))
            else:
                self.model.add(LSTM(self._rnn_units, input_shape=self._input_shape, return_sequences=True))
                self.model.add(TimeDistributed(Dense(64)))
                self.model.add(TimeDistributed(Dense(32)))
                self.model.add(TimeDistributed(Dense(1)))
            if layer != 0:
                self.model.add(Dropout(self._drop_out))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def deep_rnn_io_model_construction(self, n_layers=3):
        self.model = Sequential()
        for layer in range(n_layers):

            if layer != (n_layers - 1):
                self.model.add(LSTM(self._rnn_units, input_shape=self._input_shape, return_sequences=True))
            else:
                self.model.add(LSTM(self._rnn_units, input_shape=self._input_shape, return_sequences=False))
                self.model.add(Dense(1))

            if layer != 0:
                self.model.add(Dropout(self._drop_out))

    def bidirectional_model_construction(self, input_shape, drop_out=0.3):
        self.model = Sequential()
        self.model.add(
            Bidirectional(LSTM(self._rnn_units, return_sequences=True), input_shape=input_shape))
        self.model.add(Dropout(drop_out))
        self.model.add(TimeDistributed(Dense(1)))

    def plot_models(self):
        plot_model(model=self.model, to_file=self._log_dir + '/model.png', show_shapes=True)

    def _plot_training_history(self, model_history):
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

        dataX = np.zeros(shape=(data.shape[1], self._seq_len, 2), dtype='float32')
        for flow_id in range(data.shape[1]):
            x = data[-self._seq_len:, flow_id]
            label = m_indicator[-self._seq_len:, flow_id]

            sample = np.array([x, label]).T
            dataX[flow_id] = sample

        return dataX

    def _set_measured_flow_fairness(self, m_indicator):
        """

        :param rnn_input: shape(#n_flows, #time-steps)
        :param m_indicator: shape(n_flows, #time-steps)
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

    def _ims_tm_prediction(self, init_data, init_labels):
        multi_steps_tm = np.zeros(shape=(init_data.shape[0] + self._horizon, init_data.shape[1]),
                                  dtype='float32')
        multi_steps_tm[0:self._seq_len] = init_data

        m_indicator = np.zeros(shape=(init_labels.shape[0] + self._horizon, init_labels.shape[1]),
                               dtype='float32')
        m_indicator[0:self._seq_len] = init_labels

        for ts_ahead in range(self._horizon):
            rnn_input = self._prepare_input(data=multi_steps_tm[ts_ahead:ts_ahead + self._seq_len],
                                            m_indicator=m_indicator[ts_ahead:ts_ahead+self._seq_len])
            predictX = self.model.predict(rnn_input)
            multi_steps_tm[ts_ahead + self._seq_len] = np.squeeze(predictX, axis=1)

        return multi_steps_tm[-self._horizon:]

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

        # Predict the TM from time slot look_back
        for ts in tqdm(range(test_data_norm.shape[0] - self._horizon - self._seq_len)):
            # This block is used for iterated multi-step traffic matrices prediction

            predicted_tm = self._ims_tm_prediction(init_data=tm_pred[ts:ts + self._seq_len],
                                                   init_labels=m_indicator[ts:ts + self._seq_len])

            # Get the TM prediction of next time slot

            y_preds.append(np.expand_dims(predicted_tm, axis=0))
            pred = predicted_tm[0]

            # Using part of current prediction as input to the next estimation
            # Randomly choose the flows which is measured (using the correct data from test_set)

            # boolean array(1 x n_flows):for choosing value from predicted data
            if self._flow_selection == 'Random':
                sampling = np.random.choice(tf_a, size=self._nodes,
                                            p=[self._mon_ratio, 1 - self._mon_ratio])
            else:
                sampling = self._set_measured_flow_fairness(m_indicator=m_indicator[ts: ts + self._seq_len])

            m_indicator[ts + self._seq_len] = sampling
            # invert of sampling: for choosing value from the original data
            inv_sampling = 1.0 - sampling
            pred_input = pred * inv_sampling

            ground_true = test_data_norm[ts + self._seq_len]
            y_truths.append(
                np.expand_dims(test_data_norm[ts + self._seq_len:ts + self._seq_len + self._horizon], axis=0))

            measured_input = ground_true * sampling

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
        training_history = self.model.fit(x=self._data['x_train'],
                                          y=self._data['y_train'],
                                          batch_size=self._batch_size,
                                          epochs=self._epochs,
                                          callbacks=self.callbacks_list,
                                          validation_data=(self._data['x_val'], self._data['y_val']),
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
