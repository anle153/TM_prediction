import os

from keras.layers import Input, ConvLSTM2D, BatchNormalization, Flatten, Dense, ReLU
from keras.models import Model
from keras.utils import plot_model

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

        lstm_layer1 = ConvLSTM2D(filters=self.a_filters[0],
                                 kernel_size=self.kernel_sizes[0],
                                 strides=[1, 1],
                                 padding='same',
                                 dropout=self.dropout[0],
                                 return_sequences=True,
                                 recurrent_dropout=self.rnn_dropout[0],
                                 data_format='channels_last')(input)

        BatchNormalization_layer1 = BatchNormalization()(lstm_layer1)

        relu = ReLU()(BatchNormalization_layer1)

        lstm_layer2 = ConvLSTM2D(filters=self.a_filters[1],
                                 kernel_size=self.kernel_sizes[1],
                                 strides=[1, 1],
                                 padding='same',
                                 dropout=self.dropout[1],
                                 return_sequences=True,
                                 recurrent_dropout=self.rnn_dropout[1],
                                 data_format='channels_last')(relu)

        BatchNormalization_layer2 = BatchNormalization()(lstm_layer2)

        relu_2 = ReLU()(BatchNormalization_layer2)

        outputs = Flatten()(relu_2)

        outputs = Dense(self.wide * self.high, )(outputs)

        self.model = Model(inputs=input, outputs=outputs, name='Model')
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    def plot_models(self):
        plot_model(model=self.model, to_file=self.saving_path + '/model.png', show_shapes=True)
