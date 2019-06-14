import os

from keras.layers import *
from keras.models import Model
from keras.utils import plot_model

from Models.AbstractModel import AbstractModel


class FWBW_CONV_LSTM(AbstractModel):

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
                                    data_format='channels_last')(input)

        fw_BatchNormalization_layer1 = BatchNormalization()(fw_lstm_layer1)

        fw_lstm_layer2 = ConvLSTM2D(filters=self.a_filters[1],
                                    kernel_size=self.kernel_sizes[1],
                                    strides=[1, 1],
                                    padding='same',
                                    dropout=self.dropout[1],
                                    return_sequences=True,
                                    recurrent_dropout=self.rnn_dropout[1],
                                    data_format='channels_last')(fw_BatchNormalization_layer1)

        fw_BatchNormalization_layer2 = BatchNormalization()(fw_lstm_layer2)

        fw_flat_layer = TimeDistributed(Flatten())(fw_BatchNormalization_layer2)

        fw_outputs = TimeDistributed(Dense(512, ))(fw_flat_layer)
        fw_outputs = Dropout(0.25)(fw_outputs)
        fw_outputs = TimeDistributed(Dense(256, ))(fw_outputs)
        fw_outputs = Dropout(0.25)(fw_outputs)
        fw_outputs = TimeDistributed(Dense(self.wide * self.high, ), name='fw_outputs')(fw_outputs)

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

        bw_outputs = TimeDistributed(Dense(512, ))(bw_flat_layer)
        bw_outputs = Dropout(0.25)(bw_outputs)
        bw_outputs = TimeDistributed(Dense(256, ))(bw_outputs)
        bw_outputs = Dropout(0.25)(bw_outputs)
        bw_outputs = TimeDistributed(Dense(self.wide * self.high, ), name='bw_outputs')(bw_outputs)

        self.model = Model(inputs=input, outputs=[fw_outputs, bw_outputs], name='Model')
        self.model.compile(loss={'fw_outputs': 'mse', 'bw_outputs': 'mse'}, optimizer='adam', metrics=['mse', 'mae'])

    def plot_models(self):
        plot_model(model=self.model, to_file=self.saving_path + '/model.png', show_shapes=True)

    def plot_training_history(self, model_history):
        import matplotlib.pyplot as plt
        plt.plot(model_history.history['loss'], label='loss')
        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self.saving_path + '[loss]{}-{}.png'.format(self.alg_name, self.tag))
        plt.legend()
        plt.close()

        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self.saving_path + '[val_loss]{}-{}.png'.format(self.alg_name, self.tag))
        plt.legend()
        plt.close()

        plt.plot(model_history.history['val_fw_outputs_loss'], label='val_fw_outputs_loss')
        plt.legend()
        plt.savefig(self.saving_path + '[val_fw_outputs_los]{}-{}.png'.format(self.alg_name, self.tag))
        plt.close()

        plt.plot(model_history.history['val_bw_outputs_loss'], label='val_bw_outputs_loss')
        plt.savefig(self.saving_path + '[val_bw_outputs_loss]{}-{}.png'.format(self.alg_name, self.tag))
        plt.legend()
        plt.close()
