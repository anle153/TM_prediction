from keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Input, Concatenate, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils import plot_model

from Models.AbstractModel import AbstractModel


class lstm(AbstractModel):

    def __init__(self, saving_path, input_shape, hidden, drop_out,
                 alg_name=None, tag=None, early_stopping=False, check_point=False):
        super().__init__(alg_name=alg_name, tag=tag, early_stopping=early_stopping, check_point=check_point,
                         saving_path=saving_path)

        self.n_timestep = input_shape[0]
        self.hidden = hidden
        self.input_shape = input_shape
        self.drop_out = drop_out
        self.model = None

    def normal_model_contruction(self):
        """
        Construct RNN model from the beginning
        :param input_shape:
        :param output_dim:
        :return:
        """
        self.model = Sequential()
        self.model.add(LSTM(self.hidden, input_shape=self.input_shape))
        self.model.add(Dropout(self.drop_out))
        self.model.add(Dense(1))

    def seq2seq_model_construction(self):
        """

        :param n_timesteps:
        :param n_features:
        :return:
        """
        self.model = Sequential()
        self.model.add(LSTM(self.hidden, input_shape=self.input_shape, return_sequences=True))
        self.model.add(Dropout(self.drop_out))
        self.model.add(TimeDistributed(Dense(64)))
        self.model.add(TimeDistributed(Dense(32)))
        self.model.add(TimeDistributed(Dense(1)))

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def res_lstm_construction(self):

        input_tensor = Input(shape=self.input_shape, name='input')

        # res lstm network
        lstm_layer = LSTM(self.hidden, input_shape=self.input_shape, return_sequences=True)(input_tensor)
        drop_out = Dropout(self.drop_out)(lstm_layer)
        flat_layer = TimeDistributed(Flatten())(drop_out)
        dense_1 = TimeDistributed(Dense(64, ))(flat_layer)
        dense_2 = TimeDistributed(Dense(32, ))(dense_1)
        output = TimeDistributed(Dense(1, ))(dense_2)

        input_tensor_flatten = Reshape((self.input_shape[0] * self.input_shape[1], 1))(input_tensor)
        _input = Concatenate(axis=1)([input_tensor_flatten, output])

        _input = Flatten()(_input)
        _input = Dense(256, )(_input)
        _input = Dense(128, )(_input)
        outputs = Dense((self.n_timestep, 1), name='outputs')(_input)

        self.model = Model(inputs=input_tensor, outputs=outputs, name='res-lstm')
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def seq2seq_deep_model_construction(self, n_layers):
        self.model = Sequential()
        for layer in range(n_layers):

            if layer != (n_layers - 1):
                self.model.add(LSTM(self.hidden, input_shape=self.input_shape, return_sequences=True))
            else:
                self.model.add(LSTM(self.hidden, input_shape=self.input_shape, return_sequences=True))
                self.model.add(TimeDistributed(Dense(64)))
                self.model.add(TimeDistributed(Dense(32)))
                self.model.add(TimeDistributed(Dense(1)))
            if layer != 0:
                self.model.add(Dropout(self.drop_out))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def deep_rnn_io_model_construction(self, n_layers=3):
        self.model = Sequential()
        for layer in range(n_layers):

            if layer != (n_layers - 1):
                self.model.add(LSTM(self.hidden, input_shape=self.input_shape, return_sequences=True))
            else:
                self.model.add(LSTM(self.hidden, input_shape=self.input_shape, return_sequences=False))
                self.model.add(Dense(1))

            if layer != 0:
                self.model.add(Dropout(self.drop_out))

    def bidirectional_model_construction(self, input_shape, drop_out=0.3):
        self.model = Sequential()
        self.model.add(
            Bidirectional(LSTM(self.hidden, return_sequences=True), input_shape=input_shape))
        self.model.add(Dropout(drop_out))
        self.model.add(TimeDistributed(Dense(1)))

    def plot_models(self):
        plot_model(model=self.model, to_file=self.saving_path + '/model.png', show_shapes=True)
