# Config conv_lstm
CNN_LAYERS = 2
FILTERS = [8, 8]
KERNEL_SIZE = [[3, 3], [3, 3]]
STRIDES = [[1, 1], [1, 1]]
DROPOUTS = [0.0, 0.0]
RNN_DROPOUTS = [0.2, 0.2]

# Input Config
CNN_WIDE = 12
CNN_HIGH = 12
CNN_CHANNEL = 2
LSTM_STEP = 26

IMS_STEP = 26

# Problem hyperparams
MON_RAIO = None

HYPERPARAMS = [2.0, 0.1, 5.0, 0.4]
