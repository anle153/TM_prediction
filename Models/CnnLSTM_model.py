import os

from keras.layers import LSTM, Dense, TimeDistributed, BatchNormalization, Conv2D, Flatten, Dropout
from keras.models import Sequential
from keras.utils import plot_model

from Models.AbstractModel import AbstractModel


class CnnLSTM(AbstractModel):

    def __init__(self, saving_path, input_shape,
                 cnn_layers, a_filters, a_strides, dropouts, kernel_sizes, rnn_dropouts,
                 alg_name=None, tag=None, early_stopping=False, check_point=False):

        super().__init__(alg_name=alg_name, tag=tag, early_stopping=early_stopping, check_point=check_point,
                         saving_path=saving_path)

        if cnn_layers != len(a_filters) or cnn_layers != len(a_strides) or cnn_layers != len(
                rnn_dropouts) or cnn_layers != len(dropouts):
            print('|--- [CnnLSTM] Error: size of filters and/or strides need to be equal to the cnn_layers!')
            exit(-1)

        self.cnn_layers = cnn_layers
        self.a_filters = a_filters
        self.a_strides = a_strides
        self.dropout = dropouts
        self.rnn_dropout = rnn_dropouts
        self.kernel_sizes = kernel_sizes

        self.n_timsteps = input_shape[0]
        self.high = input_shape[1]
        self.wide = input_shape[2]
        self.channel = input_shape[3]
        self.saving_path = saving_path
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        self.model = Sequential()

        self.model.add(TimeDistributed(Conv2D(filters=self.a_filters[0],
                                              kernel_size=self.kernel_sizes[0],
                                              strides=self.a_strides[0],
                                              activation='relu',
                                              data_format='channels_last',
                                              padding='same',
                                              input_shape=(self.wide, self.high, self.channel),
                                              dropouts=0.2),
                                       input_shape=(self.n_timsteps, self.wide, self.high, self.channel)))

        self.model.add(TimeDistributed(BatchNormalization()))

        self.model.add(TimeDistributed(Conv2D(filters=self.a_filters[1],
                                              kernel_size=self.kernel_sizes[1],
                                              strides=self.a_strides[1],
                                              activation='relu',
                                              padding='same',
                                              data_format='channels_last',
                                              input_shape=(self.wide, self.high, self.channel),
                                              dropouts=0.2)))

        self.model.add(TimeDistributed(BatchNormalization()))

        self.model.add(TimeDistributed(Flatten()))

        self.model.add(LSTM(units=128, recurrent_dropout=0.25, return_sequences=True))
        self.model.add(Dropout(0.25))
        self.model.add(TimeDistributed(Dense(self.wide * self.high, )))

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def plot_models(self):
        plot_model(model=self.model, to_file=self.saving_path + '/model.png', show_shapes=True)
