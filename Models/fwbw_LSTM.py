import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Input
from keras.models import Model

from Models.AbstractModel import AbstractModel


class fwbw_lstm_model(AbstractModel):

    def __init__(self, saving_path, input_shape, hidden, drop_out,
                 alg_name=None, tag=None, early_stopping=False, check_point=False):
        super().__init__(alg_name=alg_name, tag=tag, early_stopping=early_stopping, check_point=check_point,
                         saving_path=saving_path)

        self.hidden = hidden
        self.input_shape = input_shape
        self.drop_out = drop_out

        input_tensor = Input(shape=self.input_shape, name='input')

        fw_lstm_layer = LSTM(self.hidden, input_shape=self.input_shape, return_sequences=True)(input_tensor)

        fw_drop_out = Dropout(self.drop_out)(fw_lstm_layer)

        fw_flat_layer = TimeDistributed(Flatten())(fw_drop_out)
        fw_dense_1 = TimeDistributed(Dense(64, ))(fw_flat_layer)
        fw_dense_2 = TimeDistributed(Dense(32, ))(fw_dense_1)
        fw_outputs = TimeDistributed(Dense(1, ))(fw_dense_2)

        # self.fw_model = Model(inputs=input_tensor, outputs=fw_outputs, name='FW_Model')
        #
        # self.fw_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

        bw_lstm_layer = LSTM(self.hidden, input_shape=self.input_shape,
                             return_sequences=True, go_backwards=True)(input_tensor)

        bw_drop_out = Dropout(self.drop_out)(bw_lstm_layer)

        bw_flat_layer = TimeDistributed(Flatten())(bw_drop_out)
        bw_dense_1 = TimeDistributed(Dense(64, ))(bw_flat_layer)
        bw_dense_2 = TimeDistributed(Dense(32, ))(bw_dense_1)
        bw_outputs = TimeDistributed(Dense(1, ))(bw_dense_2)

        # self.bw_model = Model(inputs=input_tensor, outputs=bw_outputs, name='BW_Model')
        #
        # self.bw_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

        _fw_outs = fw_outputs[:, :-2, 0]
        dims = [1]
        _bw_outs = tf.reverse(bw_outputs, dims)[:, 2:, 0]
        _input = input_tensor[:, 1:-1, 0]
        _labels = input_tensor[:, 1:-1, 1]

        _in_tensor = tf.concat([_fw_outs, _bw_outs, _input, _labels], axis=1)

        fc_1 = Dense(64, )(_in_tensor)
        fc_2 = Dense(32, )(fc_1)
        fc_3 = Dense(24, )(fc_2)

        first_output = tf.expand_dims(tf.expand_dims(input_tensor[:, 0, 0], axis=1), axis=2)
        last_output = tf.expand_dims(tf.expand_dims(fw_outputs[:, -1, 0], axis=1), axis=2)

        outputs = tf.concat([first_output, fc_3, last_output], axis=1)

        self.model = Model(inputs=input_tensor, outputs=outputs, name='fwbw-lstm')

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
