# --------------- Data Config -----------------

DATA_PATH = './Dataset/'
MODEL_SAVE = './trained_models/'
RESULTS_PATH = './results/'
ADDED_RESULT_NAME = 'random'

ABILENE_DAY_SIZE = 288
GEANT_DAY_SIZE = 96

ALL_DATA = True
NUM_DAYS = 160

SCALERS = ['power-transform', 'standard-scaler', 'minmax-scaler', 'box-cox', 'robust-scaler']
SCALER = SCALERS[2]
# ----------------------------------------------

# ----------------- LSTM Config ---------------
LSTM_N_EPOCH = 100
LSTM_BATCH_SIZE = 64
LSTM_HIDDEN_UNIT = 64
LSTM_DROPOUT = 0.5

LSTM_DEEP = False
LSTM_DEEP_NLAYERS = 3

LSTM_STEP = 26
LSTM_FEATURES = 2
LSTM_IMS_STEP = 12

LSTM_MON_RAIO = 0.4

LSTM_BEST_CHECKPOINT = 37
LSTM_TESTING_TIME = 1

LSTM_IMS = False

# ----------------------------------------------

# ------------- FWBW_CONV_LSTM Config ----------
FWBW_CONV_LSTM_N_EPOCH = 100
FWBW_CONV_LSTM_BATCH_SIZE = 64

FWBW_CONV_LSTM_LAYERS = 2
FWBW_CONV_LSTM_FILTERS = [8, 8]
FWBW_CONV_LSTM_KERNEL_SIZE = [[3, 3], [3, 3]]
FWBW_CONV_LSTM_STRIDES = [[1, 1], [1, 1]]
FWBW_CONV_LSTM_DROPOUTS = [0.0, 0.0]
FWBW_CONV_LSTM_RNN_DROPOUTS = [0.2, 0.2]

FWBW_CONV_LSTM_WIDE = 12
FWBW_CONV_LSTM_HIGH = 12
FWBW_CONV_LSTM_CHANNEL = 2

FWBW_CONV_LSTM_MON_RAIO = 0.45

FWBW_CONV_LSTM_IMS_STEP = 4
FWBW_CONV_LSTM_STEP = 26

FWBW_CONV_LSTM_TESTING_TIME = 1
FW_BEST_CHECKPOINT = 99
BW_BEST_CHECKPOINT = 99
FWBW_CONV_LSTM_RANDOM_ACTION = True
FWBW_CONV_LSTM_HYPERPARAMS = [2.0, 0.1, 5.0, 0.4]

FWBW_IMS = False

# ----------------------------------------------

# ----------- ARIMA Config ----------------------
ARIMA_UPDATE = 7
ARIMA_TESTING_TIME = 1
ARIMA_MON_RATIO = 0.3

ARIMA_IMS_STEP = 12
ARIMA_IMS = False
# ----------------------------------------------

# -------- HOLT_WINTERS Config -----------------
HOLT_WINTER_TREND = 'add'
HOLT_WINTER_SEASONAL = 'add'
HOLT_WINTER_UPDATE = 7

HOLT_WINTER_TESTING_TIME = 1
HOLT_WINTER_MON_RATIO = 0.3

HOLT_WINTER_IMS_STEP = 12
HOLT_WINTER_IMS = False

# ----------- CONV_LSTM Config ----------------------

CONV_LSTM_N_EPOCH = 100
CONV_LSTM_BATCH_SIZE = 64

CONV_LSTM_IMS_STEP = 12
CONV_LSTM_STEP = 26

CONV_LSTM_BEST_CHECKPOINT = 94
CONV_LSTM_TESTING_TIME = 1

CONV_LSTM_LAYERS = 2
CONV_LSTM_FILTERS = [8, 8]
CONV_LSTM_KERNEL_SIZE = [[3, 3], [3, 3]]
CONV_LSTM_STRIDES = [[1, 1], [1, 1]]
CONV_LSTM_DROPOUTS = [0.0, 0.0]
CONV_LSTM_RNN_DROPOUTS = [0.2, 0.2]

CONV_LSTM_WIDE = 12
CONV_LSTM_HIGH = 12
CONV_LSTM_CHANNEL = 2

CONV_LSTM_MON_RAIO = 0.3

CONV_LSTM_IMS = False


def set_comet_params_fwbw_conv_lstm():
    params = {
        'cnn_layers': FWBW_CONV_LSTM_LAYERS,
        'epochs': FWBW_CONV_LSTM_N_EPOCH,
        'batch_size': FWBW_CONV_LSTM_BATCH_SIZE,
        'mon_ratio': FWBW_CONV_LSTM_MON_RAIO,
        'lstm_step': FWBW_CONV_LSTM_STEP,
        'random_action': FWBW_CONV_LSTM_RANDOM_ACTION
    }

    for i in range(FWBW_CONV_LSTM_LAYERS):
        params['layer{}_filter'.format(i + 1)] = FWBW_CONV_LSTM_FILTERS[i]
        params['layer{}_kernel_size'.format(i + 1)] = FWBW_CONV_LSTM_KERNEL_SIZE[i]
        params['layer{}_stride'.format(i + 1)] = FWBW_CONV_LSTM_STRIDES[i]
        params['layer{}_dropout'.format(i + 1)] = FWBW_CONV_LSTM_DROPOUTS[i]
        params['layer{}_rnn_dropout'.format(i + 1)] = FWBW_CONV_LSTM_RNN_DROPOUTS[i]

    return params


def set_comet_params_conv_lstm():
    params = {
        'cnn_layers': CONV_LSTM_LAYERS,
        'epochs': CONV_LSTM_N_EPOCH,
        'batch_size': CONV_LSTM_BATCH_SIZE,
        'mon_ratio': CONV_LSTM_MON_RAIO,
        'lstm_step': CONV_LSTM_STEP
    }

    for i in range(FWBW_CONV_LSTM_LAYERS):
        params['layer{}_filter'.format(i + 1)] = CONV_LSTM_FILTERS[i]
        params['layer{}_kernel_size'.format(i + 1)] = CONV_LSTM_KERNEL_SIZE[i]
        params['layer{}_stride'.format(i + 1)] = CONV_LSTM_STRIDES[i]
        params['layer{}_dropout'.format(i + 1)] = CONV_LSTM_DROPOUTS[i]
        params['layer{}_rnn_dropout'.format(i + 1)] = CONV_LSTM_RNN_DROPOUTS[i]

    return params


def set_comet_params_lstm_nn():
    params = {
        'deep_lstm': LSTM_DEEP,
        'epochs': LSTM_N_EPOCH,
        'batch_size': LSTM_BATCH_SIZE,
        'mon_ratio': LSTM_MON_RAIO,
        'lstm_step': LSTM_STEP,
        'drop_out': LSTM_DROPOUT,
        'hidden_unit': LSTM_HIDDEN_UNIT
    }

    return params
