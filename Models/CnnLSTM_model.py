import os

from keras.layers import *
from keras.models import Sequential

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
        self.height = input_shape[1]
        self.weight = input_shape[2]
        self.depth = input_shape[3]
        self.saving_path = saving_path
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        convolution_model = Sequential()

        convolution_model.add(
            Reshape((Config['CNN_FEAT'], 1), batch_size=Config['NUM_STEP'], input_shape=(Config['CNN_FEAT'], 1))
        )

        convolution_model.add(
            ZeroPadding1D((0, 25))
        )

        convolution_model.add(
            Reshape((55, 55, 1))
        )

        convolution_model.add(
            Conv2D(filters=16,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   activation='relu',
                   input_shape=(55, 55),
                   data_format='channels_last')
        )
        convolution_model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2)
            )
        )
        convolution_model.add(
            Conv2D(filters=32,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   activation='relu',
                   data_format='channels_last')
        )
        convolution_model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2)
            )
        )
        convolution_model.add(
            Reshape((2 * 2 * 32,))
        )
        convolution_model.add(Dense(32, activation='relu'))

        _lstm_model = Sequential()
        _lstm_model.add(TimeDistributed(convolution_model, input_shape=(Config['NUM_STEP'], Config['CNN_FEAT'], 1)))
        _lstm_model.add(LSTM(Config['LSTM_HIDDEN_SIZE'], return_sequences=False))
        _lstm_model.add(Dropout(0.5))
        _lstm_model.add(Dense(1, activation='relu'))

        self.model = _lstm_model
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae'])
