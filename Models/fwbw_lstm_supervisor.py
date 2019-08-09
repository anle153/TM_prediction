import os
import time

from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Input, Concatenate, Reshape, Add
from keras.models import Model
from keras.utils import plot_model

from Models.AbstractModel import AbstractModel
from lib import utils


class FwbwLstmRegression(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        self._data = utils.load_dataset_fwbw_lstm(seq_len=self._seq_len, horizon=self._horizon,
                                                  input_dim=self._input_dim,
                                                  mon_ratio=self._mon_ratio, test_size=self._test_size,
                                                  **self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Model
        self.model = None

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
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
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
        plot_model(model=self.model, to_file=self.saving_path + '/model.png', show_shapes=True)

    def plot_training_history(self, model_history):
        import matplotlib.pyplot as plt

        plt.plot(model_history.history['loss'], label='loss')
        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self.saving_path + '[loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self.saving_path + '[val_loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

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

    def load(self):
        self.model.load_weights(self.saving_path+'best_model.hdf5')