
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

LSTM_MON_RAIO = 0.25

LSTM_BEST_CHECKPOINT = 14
LSTM_TESTING_TIME = 1

LSTM_IMS = False

# ----------------- FWBW_LSTM Config ---------------
FWBW_LSTM_N_EPOCH = 100
FWBW_LSTM_BATCH_SIZE = 64
FWBW_LSTM_HIDDEN_UNIT = 64
FWBW_LSTM_DROPOUT = 0.5

FWBW_LSTM_DEEP = False
FWBW_LSTM_DEEP_NLAYERS = 3

FWBW_LSTM_STEP = 26
FWBW_LSTM_FEATURES = 2
FWBW_LSTM_IMS_STEP = 12

FWBW_LSTM_MON_RAIO = 0.25

FW_LSTM_BEST_CHECKPOINT = 14
BW_LSTM_BEST_CHECKPOINT = 14
FWBW_LSTM_TESTING_TIME = 1

FWBW_LSTM_IMS = False

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

FWBW_CONV_LSTM_MON_RAIO = 0.30

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
CONV_LSTM_BATCH_SIZE = 256

CONV_LSTM_IMS_STEP = 12
CONV_LSTM_STEP = 26

CONV_LSTM_BEST_CHECKPOINT = 100
CONV_LSTM_TESTING_TIME = 1

CONV_LSTM_LAYERS = 3
CONV_LSTM_FILTERS = [8, 16, 8]
CONV_LSTM_KERNEL_SIZE = [[3, 3], [5, 5], [3, 3]]
CONV_LSTM_STRIDES = [[1, 1], [1, 1], [1, 1]]
CONV_LSTM_DROPOUTS = [0.5, 0.5, 0.5]
CONV_LSTM_RNN_DROPOUTS = [0.5, 0.5, 0.5]

CONV_LSTM_WIDE = 12
CONV_LSTM_HIGH = 12
CONV_LSTM_CHANNEL = 2

CONV_LSTM_MON_RAIO = 0.3

CONV_LSTM_IMS = False

# ----------- XGB Config ----------------------
XGB_STEP = 288 * 2
XGB_MON_RATIO = 0.3
XGB_IMS = False
XGB_IMS_STEP = 12
XGB_TESTING_TIME = 1
XGB_NJOBS = 16
XGB_FEATURES = 19

# ----------- RUNNING Config ----------------------

RUN_MODES = ['train', 'test', 'plot']
ALGS = ['fwbw-conv-lstm', 'conv-lstm', 'lstm-nn', 'arima', 'holt-winter', 'xgb']
SCALERS = ['power-transform', 'standard-scaler', 'minmax-scaler', 'box-cox', 'robust-scaler']

DATA_NAME = 'Abilene2d'

RUN_MODE = RUN_MODES[0]
ALG = ALGS[1]
GPU = 1
SCALER = SCALERS[2]

# --------------- Data Config -----------------

DATA_PATH = './Dataset/'
MODEL_SAVE = './trained_models/'
RESULTS_PATH = './results/'
ADDED_RESULT_NAME = 'random'

ABILENE_DAY_SIZE = 288
GEANT_DAY_SIZE = 96

ALL_DATA = True
NUM_DAYS = 160


# -----------------------------------------------------------------------------------------------------------------------
if ALG == ALGS[0]:
    filters = ''
    kernel = ''
    for layer in range(FWBW_CONV_LSTM_LAYERS):
        filters = filters + '{:02d}_'.format(FWBW_CONV_LSTM_FILTERS[layer])
        kernel = kernel + '{:02d}_'.format(FWBW_CONV_LSTM_KERNEL_SIZE[layer][0])

    TAG = 'mon_{:02d}_lstm_{:02d}_layers_{:02d}_filters_{}kernel_{}batch_{:03d}'.format(
        int(FWBW_CONV_LSTM_MON_RAIO * 100),
        FWBW_CONV_LSTM_STEP,
        FWBW_CONV_LSTM_LAYERS,
        filters, kernel,
        FWBW_CONV_LSTM_BATCH_SIZE)
elif ALG == ALGS[1]:
    filters = ''
    kernel = ''
    for layer in range(CONV_LSTM_LAYERS):
        filters = filters + '{:02d}_'.format(CONV_LSTM_FILTERS[layer])
        kernel = kernel + '{:02d}_'.format(CONV_LSTM_KERNEL_SIZE[layer][0])

    TAG = 'mon_{:02d}_lstm_{:02d}_layers_{:02d}_filters_{}kernel_{}batch_{:03d}'.format(int(CONV_LSTM_MON_RAIO * 100),
                                                                                        CONV_LSTM_STEP,
                                                                                        CONV_LSTM_LAYERS,
                                                                                        filters, kernel,
                                                                                        CONV_LSTM_BATCH_SIZE)
elif ALG == ALGS[2]:
    TAG = 'mon_{:02d}_lstm_{:02d}_batch_{:03d}_hidden_{:03d}'.format(int(LSTM_MON_RAIO * 100),
                                                                     LSTM_STEP,
                                                                     LSTM_BATCH_SIZE,
                                                                     LSTM_HIDDEN_UNIT)
elif ALG == ALGS[3]:
    TAG = 'mon_{:2d}_update_{:2d}'.format(int(ARIMA_MON_RATIO * 100), ARIMA_UPDATE)
elif ALG == ALGS[4]:
    TAG = 'mon_{:2d}_update_{:2d}'.format(int(HOLT_WINTER_MON_RATIO * 100), HOLT_WINTER_UPDATE)
elif ALG == ALGS[5]:
    TAG = 'mon_{:2d}_features_{:2d}_steps_{:2d}'.format(int(XGB_MON_RATIO * 100), XGB_FEATURES, XGB_STEP)
else:
    raise Exception('Unknown alg!')

########################################################################################################################


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

    for i in range(CONV_LSTM_LAYERS):
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


def set_comet_params_fwbw_lstm():
    params = {
        'deep_lstm': FWBW_LSTM_DEEP,
        'epochs': FWBW_LSTM_N_EPOCH,
        'batch_size': FWBW_LSTM_BATCH_SIZE,
        'mon_ratio': FWBW_LSTM_MON_RAIO,
        'lstm_step': FWBW_LSTM_STEP,
        'drop_out': FWBW_LSTM_DROPOUT,
        'hidden_unit': FWBW_LSTM_HIDDEN_UNIT
    }

    return params
