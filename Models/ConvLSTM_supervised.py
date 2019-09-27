import os
import time

import keras.callbacks as keras_callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model
from keras.utils import plot_model

from lib import utils


class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class ConvLSTM():

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

        self._output_dim = self._model_kwargs.get('output_dim')
        self._nodes = self._model_kwargs.get('num_nodes')
        self._wide = self._model_kwargs.get('wide')
        self._high = self._model_kwargs.get('high')
        self._channel = self._model_kwargs.get('channel')

        self._filters = self._model_kwargs.get('filters')
        self._kernel_size = self._model_kwargs.get('kernel_size')
        self._strides = self._model_kwargs.get('strides')
        self._input_shape = (self._seq_len, self._wide, self._high, self._channel)

        # Train's args
        self._conv_dropout = self._train_kwargs.get('conv_dropout')
        self._rnn_dropout = self._train_kwargs.get('rnn_dropout')
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
        self._data = utils.load_dataset_conv_lstm(seq_len=self._seq_len,
                                                  wide=self._wide, high=self._high, channel=self._channel,
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

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            mon_r = kwargs['mon_ratio']
            run_id = 'conv_lstm_%g_%d_%s_%g_%d_%s/' % (
                mon_r,
                horizon, structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def construct_conv_lstm(self):
        input = Input(shape=self._input_shape, name='input')

        lstm_layer1 = ConvLSTM2D(filters=self._filters[0],
                                 kernel_size=self._kernel_size[0],
                                 strides=self._strides[0],
                                 padding='same',
                                 dropout=self._conv_dropout,
                                 return_sequences=True,
                                 recurrent_dropout=self._rnn_dropout,
                                 data_format='channels_last')(input)

        BatchNormalization_layer1 = BatchNormalization()(lstm_layer1)

        lstm_layer2 = ConvLSTM2D(filters=self._filters[1],
                                 kernel_size=self._kernel_size[1],
                                 strides=self._strides[1],
                                 padding='same',
                                 dropout=self._conv_dropout,
                                 return_sequences=True,
                                 recurrent_dropout=self._rnn_dropout,
                                 data_format='channels_last')(BatchNormalization_layer1)

        BatchNormalization_layer2 = BatchNormalization()(lstm_layer2)

        outputs = Flatten()(BatchNormalization_layer2)

        outputs = Dense(512, )(outputs)
        outputs = Dropout(self._rnn_dropout)(outputs)
        outputs = Dense(256, )(outputs)
        outputs = Dropout(self._rnn_dropout)(outputs)

        outputs = Dense(self._wide * self._high, )(outputs)

        self.model = Model(inputs=input, outputs=outputs, name='Model')
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def plot_models(self):
        plot_model(model=self.model, to_file=self._log_dir + '/model.png', show_shapes=True)
