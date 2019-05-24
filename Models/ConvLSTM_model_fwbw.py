import os

import tensorflow as tf
from keras.layers import Input, ConvLSTM2D, BatchNormalization, TimeDistributed, Flatten, Dense
from keras.models import Model

from Models.AbstractModel import AbstractModel


class ConvLSTM_FWBW(AbstractModel):

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

        fw_lstm_layer1 = ConvLSTM2D(filters=self.a_filters[0],
                                    kernel_size=self.kernel_sizes[0],
                                    strides=[1, 1],
                                    padding='same',
                                    dropout=self.dropout[0],
                                    return_sequences=True,
                                    recurrent_dropout=self.rnn_dropout[0],
                                    data_format='channels_last'
                                    )(input)

        fw_BatchNormalization_layer1 = BatchNormalization()(fw_lstm_layer1)

        fw_lstm_layer2 = ConvLSTM2D(filters=self.a_filters[1],
                                    kernel_size=self.kernel_sizes[1],
                                    strides=[1, 1],
                                    padding='same',
                                    dropout=self.dropout[1],
                                    return_sequences=True,
                                    recurrent_dropout=self.rnn_dropout[1],
                                    data_format='channels_last'
                                    )(fw_BatchNormalization_layer1)

        fw_BatchNormalization_layer2 = BatchNormalization()(fw_lstm_layer2)

        fw_flat_layer = TimeDistributed(Flatten())(fw_BatchNormalization_layer2)

        fw_first_Dense = TimeDistributed(Dense(512, ))(fw_flat_layer)
        fw_second_Dense = TimeDistributed(Dense(256, ))(fw_first_Dense)
        fw_outputs = TimeDistributed(Dense(144, ))(fw_second_Dense)

        self.fw_model = Model(inputs=input, outputs=fw_outputs, name='Model')

        self.fw_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

        # ------------------------------- bw net -----------------------------------------------------------------------
        bw_lstm_layer1 = ConvLSTM2D(filters=self.a_filters[0],
                                    kernel_size=self.kernel_sizes[0],
                                    strides=[1, 1],
                                    padding='same',
                                    dropout=self.dropout[0],
                                    return_sequences=True,
                                    recurrent_dropout=self.rnn_dropout[0],
                                    data_format='channels_last',
                                    go_backwards=True)(input)

        bw_BatchNormalization_layer1 = BatchNormalization()(bw_lstm_layer1)

        bw_lstm_layer2 = ConvLSTM2D(filters=self.a_filters[1],
                                    kernel_size=self.kernel_sizes[1],
                                    strides=[1, 1],
                                    padding='same',
                                    dropout=self.dropout[1],
                                    return_sequences=True,
                                    recurrent_dropout=self.rnn_dropout[1],
                                    data_format='channels_last')(bw_BatchNormalization_layer1)

        bw_BatchNormalization_layer2 = BatchNormalization()(bw_lstm_layer2)

        bw_flat_layer = TimeDistributed(Flatten())(bw_BatchNormalization_layer2)

        bw_first_Dense = TimeDistributed(Dense(512, ))(bw_flat_layer)
        bw_second_Dense = TimeDistributed(Dense(256, ))(bw_first_Dense)
        bw_outputs = TimeDistributed(Dense(144, ))(bw_second_Dense)

        self.bw_model = Model(inputs=input, outputs=bw_outputs, name='Model')

        self.bw_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

        # ------------------------------------ Data correction ---------------------------------------------------------
        _fw_output = fw_outputs[0:-2]
        _bw_output = bw_outputs[2:]

        _input = input[1:-1]

        data_corr_input = tf.concat([_fw_output, _bw_output, _input], axis=3)

        data_corr_input = Input(shape=(self.n_timsteps - 2, self.wide, self.high, 3))
