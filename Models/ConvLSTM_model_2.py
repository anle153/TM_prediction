import os

from keras.layers import *
from keras.models import Model

from Models.AbstractModel import AbstractModel


class ConvLSTM(AbstractModel):

    def __init__(self, saving_path, input_shape,
                 cnn_layers, a_filters, a_strides, dropouts, kernel_sizes, rnn_dropouts,
                 alg_name=None, tag=None, early_stopping=False, check_point=False):

        super().__init__(alg_name=alg_name, tag=tag, early_stopping=early_stopping, check_point=check_point,
                         saving_path=saving_path)

        if cnn_layers != len(a_filters) or cnn_layers != len(a_strides) or cnn_layers != len(
                rnn_dropouts) or cnn_layers != len(dropouts):
            print('|--- [ConvLSTM] Error: size of filters and/or strides need to be equal to the cnn_layers!')
            exit(-1)

        self.cnn_layers = cnn_layers
        self.a_filters = a_filters
        self.a_strides = a_strides
        self.dropout = dropouts
        self.rnn_dropout = rnn_dropouts
        self.kernel_sizes = kernel_sizes

        self.n_timsteps = input_shape[0]
        self.wide = input_shape[1]
        self.high = input_shape[2]
        self.channel = input_shape[3]
        self.saving_path = saving_path
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        input = Input(shape=(self.n_timsteps, self.wide, self.high, self.channel), name='input')

        layer_0 = ConvLSTM2D(filters=self.a_filters[0],
                             kernel_size=self.kernel_sizes[0],
                             strides=[1, 1],
                             padding='same',
                             dropout=self.dropout[0],
                             return_sequences=True,
                             recurrent_dropout=self.rnn_dropout[0],
                             data_format='channels_last'
                             )(input)

        BatchNormalization_0 = BatchNormalization()(layer_0)

        first_Pooling = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(
            BatchNormalization_0)

        layer_1 = ConvLSTM2D(filters=self.a_filters[1],
                             kernel_size=self.kernel_sizes[1],
                             strides=[1, 1],
                             padding='same',
                             dropout=self.dropout[1],
                             return_sequences=True,
                             recurrent_dropout=self.rnn_dropout[1],
                             data_format='channels_last'
                             )(first_Pooling)

        BatchNormalization_1 = BatchNormalization()(layer_1)
        second_Pooling = MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_last')(
            BatchNormalization_1)

        flat_layer = TimeDistributed(Flatten())(second_Pooling)

        first_Dense = TimeDistributed(Dense(512, ))(flat_layer)
        second_Dense = TimeDistributed(Dense(256, ))(first_Dense)
        outputs = TimeDistributed(Dense(144, ))(second_Dense)

        self.model = Model(inputs=input, outputs=outputs, name='Model')

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
