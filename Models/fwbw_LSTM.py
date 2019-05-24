from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Input, Concatenate, Reshape
from keras.models import Model
from keras.utils import plot_model

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
        fw_outputs = TimeDistributed(Dense(1, name='fw_output'))(fw_dense_2)

        fw_out = Flatten()(fw_outputs)
        fw_out = Dense(256, )(fw_out)
        fw_out = Dropout(0.2)(fw_out)
        fw_out = Dense(128, )(fw_out)
        fw_out = Dropout(0.2)(fw_out)
        fw_out = Dense(64, )(fw_out)
        fw_out = Dense(1, name='pred_data')(fw_out)

        bw_lstm_layer = LSTM(self.hidden, input_shape=self.input_shape,
                             return_sequences=True, go_backwards=True)(input_tensor)

        bw_drop_out = Dropout(self.drop_out)(bw_lstm_layer)

        bw_flat_layer = TimeDistributed(Flatten())(bw_drop_out)
        bw_dense_1 = TimeDistributed(Dense(64, ))(bw_flat_layer)
        bw_dense_2 = TimeDistributed(Dense(32, ))(bw_dense_1)
        bw_outputs = TimeDistributed(Dense(1, ))(bw_dense_2)

        input_tensor_flatten = Reshape((self.input_shape[0] * self.input_shape[1], 1))(input_tensor)
        _input = Concatenate(axis=1)([input_tensor_flatten, fw_outputs, bw_outputs])

        _input = Flatten()(_input)
        x = Dense(128, )(_input)
        x = Dense(64, )(x)
        outputs = Dense(24, name='corr_data')(x)

        self.model = Model(inputs=input_tensor, outputs=[fw_out, outputs], name='fwbw-lstm')

        self.model.compile(loss={'pred_data': 'mse', 'corr_data': 'mse'}, optimizer='adam', metrics=['mse', 'mae'])

    def plot_models(self):
        plot_model(model=self.model, to_file=self.saving_path + '/model.png')
