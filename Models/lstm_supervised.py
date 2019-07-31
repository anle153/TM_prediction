from keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Input, Concatenate, Flatten, Reshape, Add
from keras.models import Sequential, Model
from keras.utils import plot_model

from Models.AbstractModel import AbstractModel
from lib import utils


class lstm(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')
        self._model_kwargs = kwargs.get('model')

        self._alg_name = self._kwargs.get('alg')

        self._hidden = self._model_kwargs.get('rnn_units')

        self._seq_len = self._model_kwargs.get('seq_len')
        self._input_dim = self._model_kwargs.get('input_dim')
        self._input_shape = (self._seq_len, self._input_dim)

        self._output_dim = self._model_kwargs.get('output_dim')

        self._drop_out = self._train_kwargs.get('dropout')

        self._epochs = self._train_kwargs.get('epochs')

        self._batch_size = self._data_kwargs.get('batch_size')

        self.model = None

        self._data = utils.load_dataset_lstm(**self._data_kwargs)

    def normal_model_contruction(self):
        """
        Construct RNN model from the beginning
        :param input_shape:
        :param output_dim:
        :return:
        """
        self.model = Sequential()
        self.model.add(LSTM(self._hidden, input_shape=self._input_shape))
        self.model.add(Dropout(self._drop_out))
        self.model.add(Dense(1))

    def seq2seq_model_construction(self):
        """

        :param n_timesteps:
        :param n_features:
        :return:
        """
        self.model = Sequential()
        self.model.add(LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True))
        self.model.add(Dropout(self._drop_out))
        self.model.add(TimeDistributed(Dense(64)))
        self.model.add(TimeDistributed(Dense(32)))
        self.model.add(TimeDistributed(Dense(1)))

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def res_lstm_construction(self):

        input_tensor = Input(shape=self._input_shape, name='input')

        # res lstm network
        lstm_layer = LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True)(input_tensor)
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
        lstm_layer = LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True)(input_tensor)
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
                self.model.add(LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True))
            else:
                self.model.add(LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True))
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
                self.model.add(LSTM(self._hidden, input_shape=self._input_shape, return_sequences=True))
            else:
                self.model.add(LSTM(self._hidden, input_shape=self._input_shape, return_sequences=False))
                self.model.add(Dense(1))

            if layer != 0:
                self.model.add(Dropout(self._drop_out))

    def bidirectional_model_construction(self, input_shape, drop_out=0.3):
        self.model = Sequential()
        self.model.add(
            Bidirectional(LSTM(self._hidden, return_sequences=True), input_shape=input_shape))
        self.model.add(Dropout(drop_out))
        self.model.add(TimeDistributed(Dense(1)))

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
                                             y=self._data['y_train'],
                                             batch_size=self._batch_size,
                                             epochs=self._epochs,
                                             callbacks=self.callbacks_list,
                                             validation_data=(self._data['x_val'], self._data['y_val']),
                                             shuffle=True,
                                             verbose=2)
        if training_fw_history is not None:
            self.plot_training_history(training_fw_history)
            self.save_model_history(training_fw_history)

    def load(self):
        self.model.load_weights(self.saving_path + 'best_model.hdf5')
