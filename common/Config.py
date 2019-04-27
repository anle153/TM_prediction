# Data Config
DATA_PATH = './Dataset/'
MODEL_SAVE = './trained_models/'
RESULTS_PATH = './results/'
ADDED_RESULT_NAME = 'test_150d'

TESTING_TIME = 1

ABILENE_DAY_SIZE = 288
GEANT_DAY_SIZE = 96

ALL_DATA = False
NUM_DAYS = 160

# Training lstm-based model config
N_EPOCH = 100
BATCH_SIZE = 64
NUM_ITER = 20000

# Testing config
LSTM_BEST_CHECKPOINT = 64
CONV_LSTM_BEST_CHECKPOINT = 64
FW_BEST_CHECKPOINT = 100
BW_BEST_CHECKPOINT = 99

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

# LSTM Config
LSTM_STEP = 26
LSTM_FEATURES = 2
LSTM_HIDDEN_UNIT = 64
LSTM_DROPOUT = 0.5
IMS_STEP = 12

LSTM_DEEP = False
LSTM_DEEP_NLAYERS = 3

# Arima Config
ARIMA_UPDATE = 7

# Holt-Winter Config
HOLT_WINTER_TREND = 'add'
HOLT_WINTER_SEASONAL = 'add'
HOLT_WINTER_UPDATE = 7

# Problem hyperparams
FWBW_CONV_LSTM_RANDOM_ACTION = False
MON_RAIO = 0.3
HYPERPARAMS = [2.0, 0.1, 5.0, 0.4]
