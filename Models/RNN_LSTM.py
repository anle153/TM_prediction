from keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from keras.models import Sequential

from Models.AbstractModel import AbstractModel


class lstm(AbstractModel):

    def __init__(self, saving_path, input_shape, hidden, drop_out,
                 alg_name=None, tag=None, early_stopping=False, check_point=False):
        super().__init__(alg_name=alg_name, tag=tag, early_stopping=early_stopping, check_point=check_point,
                         saving_path=saving_path)

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
