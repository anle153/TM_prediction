import os

from keras.layers import *
from keras.layers import Activation
from keras.models import Sequential

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
        self.height = input_shape[1]
        self.weight = input_shape[2]
        self.depth = input_shape[3]
        self.saving_path = saving_path
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        self.model = Sequential()

        for cnn_layer in range(self.cnn_layers):
            self.model.add(ConvLSTM2D(filters=self.a_filters[cnn_layer],
                                      kernel_size=self.kernel_sizes[cnn_layer],
                                      strides=[1, 1],
                                      padding='same',
                                      dropout=self.dropout[cnn_layer],
                                      input_shape=(None, self.weight, self.height, self.depth),
                                      return_sequences=True,
                                      recurrent_dropout=self.rnn_dropout[cnn_layer]))

            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))

        self.model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                              activation='relu',
                              padding='same', data_format='channels_last'))