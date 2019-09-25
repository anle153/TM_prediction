import os
import time

import keras.callbacks as keras_callbacks
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Input, Concatenate, Reshape, Add
from keras.models import Model
from keras.utils import plot_model
from tqdm import tqdm

from common.error_utils import error_ratio
from lib import metrics
from lib import utils


class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class FwbwLstmRegression():

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
        self._test_size = self._test_kwargs.get('test_size')
        self._results_path = self._test_kwargs.get('results_path')
        self._lamda = []
        self._lamda.append(self._test_kwargs.get('lamda_0'))
        self._lamda.append(self._test_kwargs.get('lamda_1'))
        self._lamda.append(self._test_kwargs.get('lamda_2'))

        self._mon_ratio = self._kwargs.get('mon_ratio')

        # Load data
        self._data = utils.load_dataset_fwbw_lstm(seq_len=self._seq_len, horizon=self._horizon,
                                                  input_dim=self._input_dim,
                                                  mon_ratio=self._mon_ratio, test_size=self._test_size,
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
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'fwbw_lstm_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def construct_fwbw_lstm_2(self):
        input_tensor = Input(shape=self._input_shape, name='input')

        fw_lstm_layer = LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True)(input_tensor)
        fw_drop_out = Dropout(self._drop_out)(fw_lstm_layer)

        fw_flat_layer = TimeDistributed(Flatten())(fw_drop_out)
        fw_dense_1 = TimeDistributed(Dense(64, ))(fw_flat_layer)
        fw_dense_2 = TimeDistributed(Dense(32, ))(fw_dense_1)
        fw_output = TimeDistributed(Dense(1, ))(fw_dense_2)

        fw_input_tensor_flatten = Reshape((self._input_shape[0] * self._input_shape[1], 1))(input_tensor)
        _input_fw = Concatenate(axis=1)([fw_input_tensor_flatten, fw_output])

        _input_fw = Flatten()(_input_fw)
        _input_fw = Dense(256, )(_input_fw)
        _input_fw = Dense(128, )(_input_fw)
        fw_outputs = Dense(self._seq_len, name='fw_outputs')(_input_fw)

        bw_lstm_layer = LSTM(self._hidden, input_shape=self._input_shape,
                             return_sequences=True, go_backwards=True)(input_tensor)

        bw_drop_out = Dropout(self._drop_out)(bw_lstm_layer)

        bw_flat_layer = TimeDistributed(Flatten())(bw_drop_out)
        bw_dense_1 = TimeDistributed(Dense(64, ))(bw_flat_layer)
        bw_dense_2 = TimeDistributed(Dense(32, ))(bw_dense_1)
        bw_outputs = TimeDistributed(Dense(1, ))(bw_dense_2)

        input_tensor_flatten = Reshape((self._input_shape[0] * self._input_shape[1], 1))(input_tensor)
        _input = Concatenate(axis=1)([input_tensor_flatten, bw_outputs])

        _input = Flatten()(_input)
        x = Dense(256, )(_input)
        x = Dense(128, )(x)
        corr_data = Dense(self._seq_len - 2, name='corr_data')(x)

        self.model = Model(inputs=input_tensor, outputs=[fw_outputs, corr_data], name='fwbw-lstm')

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def construct_fwbw_lstm(self):
        # Input
        input_tensor = Input(shape=self._input_shape, name='input')

        # Forward Network
        fw_lstm_layer = LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True)(input_tensor)
        fw_drop_out = Dropout(self._drop_out)(fw_lstm_layer)
        fw_flat_layer = TimeDistributed(Flatten())(fw_drop_out)
        fw_dense_1 = TimeDistributed(Dense(64, ))(fw_flat_layer)
        fw_dense_2 = TimeDistributed(Dense(32, ))(fw_dense_1)
        fw_outputs = TimeDistributed(Dense(1, ), name='fw_outputs')(fw_dense_2)

        # Backward Network
        bw_lstm_layer = LSTM(self._hidden, input_shape=self._input_shape,
                             return_sequences=True, go_backwards=True)(input_tensor)
        bw_drop_out = Dropout(self._drop_out)(bw_lstm_layer)
        bw_flat_layer = TimeDistributed(Flatten())(bw_drop_out)
        bw_dense_1 = TimeDistributed(Dense(64, ))(bw_flat_layer)
        bw_dense_2 = TimeDistributed(Dense(32, ))(bw_dense_1)
        bw_output = TimeDistributed(Dense(1, ))(bw_dense_2)

        bw_input_tensor_flatten = Reshape((self._input_shape[0] * self._input_shape[1], 1))(input_tensor)
        _input_bw = Concatenate(axis=1)([bw_input_tensor_flatten, bw_output])

        _input_bw = Flatten()(_input_bw)
        _input_bw = Dense(256, )(_input_bw)
        _input_bw = Dense(128, )(_input_bw)
        bw_outputs = Dense(self._seq_len, name='bw_outputs')(_input_bw)

        self.model = Model(inputs=input_tensor, outputs=[fw_outputs, bw_outputs], name='fwbw-lstm')

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def construct_fwbw_lstm_no_sc(self):
        # Input
        input_tensor = Input(shape=self._input_shape, name='input')

        # Forward Network
        fw_lstm_layer = LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True)(input_tensor)
        fw_drop_out = Dropout(self._drop_out)(fw_lstm_layer)
        fw_flat_layer = TimeDistributed(Flatten())(fw_drop_out)
        fw_dense_1 = TimeDistributed(Dense(64, ))(fw_flat_layer)
        fw_dense_2 = TimeDistributed(Dense(32, ))(fw_dense_1)
        fw_outputs = TimeDistributed(Dense(1, ), name='fw_outputs')(fw_dense_2)

        # Backward Network
        bw_lstm_layer = LSTM(self._hidden, input_shape=self._input_shape,
                             return_sequences=True, go_backwards=True)(input_tensor)
        bw_drop_out = Dropout(self._drop_out)(bw_lstm_layer)
        bw_flat_layer = TimeDistributed(Flatten())(bw_drop_out)
        bw_dense_1 = TimeDistributed(Dense(64, ))(bw_flat_layer)
        bw_dense_2 = TimeDistributed(Dense(32, ))(bw_dense_1)
        bw_outputs = TimeDistributed(Dense(1, ))(bw_dense_2)

        self.model = Model(inputs=input_tensor, outputs=[fw_outputs, bw_outputs], name='fwbw-lstm')

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def construct_res_fwbw_lstm(self):
        # Input
        input_tensor = Input(shape=self._input_shape, name='input')
        input_2 = Input(shape=(self._seq_len, 1), name='input2')

        # Forward Network
        fw_lstm_layer = LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True)(input_tensor)
        fw_drop_out = Dropout(self._drop_out)(fw_lstm_layer)
        fw_flat_layer = TimeDistributed(Flatten())(fw_drop_out)
        fw_dense_1 = TimeDistributed(Dense(64, ))(fw_flat_layer)
        fw_dense_2 = TimeDistributed(Dense(32, ))(fw_dense_1)
        fw_output = TimeDistributed(Dense(1, ))(fw_dense_2)

        # fw_input_tensor_flatten = Reshape((self.input_shape[0] * self.input_shape[1], 1))(input_tensor)
        _input_fw = Add()([input_2, fw_output])

        _input_fw = Flatten()(_input_fw)
        _input_fw = Dense(64, )(_input_fw)
        fw_outputs = Dense(self._seq_len, name='fw_outputs')(_input_fw)

        # Backward Network
        bw_lstm_layer = LSTM(self._hidden, input_shape=self._input_shape,
                             return_sequences=True, go_backwards=True)(input_tensor)
        bw_drop_out = Dropout(self._drop_out)(bw_lstm_layer)
        bw_flat_layer = TimeDistributed(Flatten())(bw_drop_out)
        bw_dense_1 = TimeDistributed(Dense(64, ))(bw_flat_layer)
        bw_dense_2 = TimeDistributed(Dense(32, ))(bw_dense_1)
        bw_output = TimeDistributed(Dense(1, ))(bw_dense_2)

        _input_bw = Add()([input_2, bw_output])

        _input_bw = Flatten()(_input_bw)
        _input_bw = Dense(64, )(_input_bw)
        bw_outputs = Dense(self._seq_len, name='bw_outputs')(_input_bw)

        self.model = Model(inputs=[input_tensor, input_2], outputs=[fw_outputs, bw_outputs], name='fwbw-lstm')

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

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

        dataX = np.zeros(shape=(data.shape[1], self._seq_len, self._input_dim), dtype='float32')
        for flow_id in range(data.shape[1]):
            x = data[:, flow_id]
            label = m_indicator[:, flow_id]

            dataX[flow_id, :, 0] = x
            dataX[flow_id, :, 1] = label

        return dataX

    def _ims_tm_prediction(self, init_data, init_labels):
        multi_steps_tm = np.zeros(shape=(init_data.shape[0] + self._horizon, init_data.shape[1]),
                                  dtype='float32')
        multi_steps_tm[0:self._seq_len] = init_data

        m_indicator = np.zeros(shape=(init_labels.shape[0] + self._horizon, init_labels.shape[1]),
                               dtype='float32')
        m_indicator[0:self._seq_len] = init_labels

        bw_outputs = None

        for ts_ahead in range(self._horizon):
            rnn_input = self._prepare_input(data=multi_steps_tm[ts_ahead:ts_ahead + self._seq_len],
                                            m_indicator=m_indicator[ts_ahead:ts_ahead + self._seq_len])
            predictX_1, predictX_2 = self.model.predict(rnn_input)
            predictX_1 = predictX_1[:, -1, 0]
            multi_steps_tm[ts_ahead + self._seq_len] = predictX_1

            if ts_ahead == 0:
                bw_outputs = predictX_2.copy()

        return multi_steps_tm[-self._horizon:], bw_outputs

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

    def data_correction_v3(self, rnn_input, pred_backward, labels):
        # Shape = (#n_flows, #time-steps)

        beta = np.zeros(rnn_input.shape)

        corrected_range = int(self._seq_len / self._r)

        for i in range(rnn_input.shape[1] - corrected_range):
            mu = np.sum(labels[:, i + 1:i + corrected_range + 1], axis=1) / corrected_range

            h = np.arange(1, corrected_range + 1)

            rho = (1 / (np.log(corrected_range) + 1)) * np.sum(
                labels[:, i + 1:i + corrected_range + 1] / h, axis=1)

            beta[:, i] = mu * rho

        considered_backward = pred_backward[:, 1:]
        considered_rnn_input = rnn_input[:, 0:-1]

        beta[beta > 0.5] = 0.5

        alpha = 1.0 - beta

        alpha = alpha[:, 0:-1]
        beta = beta[:, 0:-1]
        # gamma = gamma[:, 1:-1]

        # corrected_data = considered_rnn_input * alpha + considered_rnn_input * beta + considered_backward * gamma
        print('shape rnn input: {}'.format(considered_rnn_input.shape))
        print('shape alpha: {}'.format(alpha.shape))
        print('shape backward: {}'.format(considered_backward.shape))
        print('shape beta: {}'.format(beta.shape))
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

        # Predict the TM from time slot look_back
        for ts in tqdm(range(test_data_norm.shape[0] - self._horizon - self._seq_len)):
            # This block is used for iterated multi-step traffic matrices prediction

            fw_outputs, bw_outputs = self._ims_tm_prediction(init_data=tm_pred[ts:ts + self._seq_len],
                                                             init_labels=m_indicator[ts:ts + self._seq_len])

            # Get the TM prediction of next time slot
            corrected_data = self.data_correction_v3(rnn_input=tm_pred[ts: ts + self._seq_len],
                                                     pred_backward=bw_outputs,
                                                     labels=m_indicator[ts: ts + self._seq_len])
            # corrected_data = data_correction_v2(rnn_input=np.copy(tm_pred[ts: ts + config['model']['seq_len']]),
            #                                     pred_backward=bw_outputs,
            #                                     labels=labels[ts: ts + config['model']['seq_len']])

            measured_data = tm_pred[ts:ts + self._seq_len - 1] * m_indicator[ts:ts + self._seq_len - 1]
            pred_data = corrected_data * (1.0 - m_indicator[ts:ts + self._seq_len - 1])
            tm_pred[ts:ts + self._seq_len - 1] = measured_data + pred_data

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
            self.save_model_history(training_fw_history)

    def evaluate(self):

        scaler = self._data['scaler']

        y_pred_1, y_pred_2 = self.model.predict(self._data['x_eval'])
        y_pred_1 = scaler.inverse_transform(y_pred_1)
        y_truth_1 = scaler.inverse_transform(self._data['y_eval_1'])
        y_truth_2 = scaler.inverse_transform(self._data['y_eval_2'])

        mse = metrics.masked_mse_np(preds=y_pred_1, labels=y_truth_1, null_val=0)
        mae = metrics.masked_mae_np(preds=y_pred_1, labels=y_truth_1, null_val=0)
        mape = metrics.masked_mape_np(preds=y_pred_1, labels=y_truth_1, null_val=0)
        rmse = metrics.masked_rmse_np(preds=y_pred_1, labels=y_truth_1, null_val=0)
        self._logger.info(
            " Forward results: MSE: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}".format(
                mse, mae, rmse, mape
            )
        )

        mse_2 = metrics.masked_mse_np(preds=y_pred_2, labels=y_truth_2, null_val=0)
        mae_2 = metrics.masked_mae_np(preds=y_pred_2, labels=y_truth_2, null_val=0)
        mape_2 = metrics.masked_mape_np(preds=y_pred_2, labels=y_truth_2, null_val=0)
        rmse_2 = metrics.masked_rmse_np(preds=y_pred_2, labels=y_truth_2, null_val=0)
        self._logger.info(
            "Backward results: MSE: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}".format(
                mse_2, mae_2, rmse_2, mape_2
            )
        )

    def test(self):
        scaler = self._data['scaler']

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

            tm_pred = scaler.inverse_transform(tm_pred)
            g_truth = scaler.inverse_transform(self._data['test_data_norm'][self._seq_len:-self._horizon])

            er = error_ratio(y_pred=tm_pred,
                             y_true=g_truth,
                             measured_matrix=m_indicator)
            self._save_results(g_truth=g_truth, pred_tm=tm_pred, m_indicator=m_indicator, tag=str(i))

            print('ER: {}'.format(er))

    def _save_results(self, g_truth, pred_tm, m_indicator, tag):
        np.save(self._log_dir + '/g_truth{}'.format(tag), g_truth)
        np.save(self._log_dir + '/pred_tm_{}'.format(tag), pred_tm)
        np.save(self._log_dir + '/m_indicator{}'.format(tag), m_indicator)

    def load(self):
        self.model.load_weights(self._log_dir + 'best_model.hdf5')
