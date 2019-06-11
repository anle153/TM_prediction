
# ----------------- LSTM Config ---------------
LSTM_N_EPOCH = 100
LSTM_BATCH_SIZE = 512
LSTM_HIDDEN_UNIT = 64
LSTM_DROPOUT = 0.5

LSTM_DEEP = False
LSTM_DEEP_NLAYERS = 3

LSTM_STEP = 26
LSTM_FEATURES = 2
LSTM_IMS_STEP = 12

LSTM_MON_RAIO = 0.30

LSTM_BEST_CHECKPOINT = 2
LSTM_TESTING_TIME = 10

LSTM_IMS = False
LSTM_VALID_TEST = True

# ----------------- FWBW_LSTM Config ---------------
FWBW_LSTM_N_EPOCH = 30
FWBW_LSTM_BATCH_SIZE = 256
FWBW_LSTM_HIDDEN_UNIT = 128
FWBW_LSTM_DROPOUT = 0.5

FWBW_LSTM_DEEP = False
FWBW_LSTM_DEEP_NLAYERS = 3

FWBW_LSTM_STEP = 26
FWBW_LSTM_FEATURES = 2
FWBW_LSTM_IMS_STEP = 12

FWBW_LSTM_MON_RAIO = 0.30

FWBW_LSTM_BEST_CHECKPOINT = 10
FWBW_LSTM_TESTING_TIME = 10

FWBW_LSTM_IMS = False

FWBW_LSTM_VALID_TEST = True

# ----------------------------------------------

# ------------- FWBW_CONV_LSTM Config ----------
FWBW_CONV_LSTM_N_EPOCH = 100
FWBW_CONV_LSTM_BATCH_SIZE = 256

FWBW_CONV_LSTM_LAYERS = 2
FWBW_CONV_LSTM_FILTERS = [2, 2]
FWBW_CONV_LSTM_KERNEL_SIZE = [[3, 3], [3, 3]]
FWBW_CONV_LSTM_STRIDES = [[1, 1], [1, 1]]
FWBW_CONV_LSTM_DROPOUTS = [0.25, 0.25]
FWBW_CONV_LSTM_RNN_DROPOUTS = [0.25, 0.25]

FWBW_CONV_LSTM_WIDE = 12
FWBW_CONV_LSTM_HIGH = 12
FWBW_CONV_LSTM_CHANNEL = 2

FWBW_CONV_LSTM_MON_RAIO = 0.30

FWBW_CONV_LSTM_IMS_STEP = 4
FWBW_CONV_LSTM_STEP = 20

FWBW_CONV_LSTM_TESTING_TIME = 20
FWBW_CONV_LSTM_BEST_CHECKPOINT = 169
FWBW_CONV_LSTM_RANDOM_ACTION = True
FWBW_CONV_LSTM_HYPERPARAMS = [2.0, 0.1, 5.0, 0.4]

FWBW_IMS = False
FWBW_CONV_LSTM_VALID_TEST = True

# ----------------------------------------------
# ------------- FWBW_CONVLSTM Config ----------
FWBW_CONVLSTM_N_EPOCH = 20
FWBW_CONVLSTM_BATCH_SIZE = 256

FWBW_CONVLSTM_LAYERS = 2
FWBW_CONVLSTM_FILTERS = [2, 2]
FWBW_CONVLSTM_KERNEL_SIZE = [[3, 3], [3, 3]]
FWBW_CONVLSTM_STRIDES = [[1, 1], [1, 1]]
FWBW_CONVLSTM_DROPOUTS = [0.25, 0.25]
FWBW_CONVLSTM_RNN_DROPOUTS = [0.25, 0.25]

FWBW_CONVLSTM_WIDE = 12
FWBW_CONVLSTM_HIGH = 12
FWBW_CONVLSTM_CHANNEL = 2

FWBW_CONVLSTM_MON_RAIO = 0.30

FWBW_CONVLSTM_IMS_STEP = 4
FWBW_CONVLSTM_STEP = 26

FWBW_CONVLSTM_TESTING_TIME = 10
FWBW_CONVLSTM_BEST_CHECKPOINT = 15
FWBW_CONVLSTM_RANDOM_ACTION = True
FWBW_CONVLSTM_HYPERPARAMS = [2.0, 0.1, 5.0, 0.4]

FWBW_CONVLSTM_IMS = False

# ----------------------------------------------

# ----------- ARIMA Config ----------------------
ARIMA_UPDATE = 7
ARIMA_TESTING_TIME = 1
ARIMA_MON_RATIO = 0.2

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

CONV_LSTM_N_EPOCH = 200
CONV_LSTM_BATCH_SIZE = 256

CONV_LSTM_IMS_STEP = 12
CONV_LSTM_STEP = 26

CONV_LSTM_BEST_CHECKPOINT = 200
CONV_LSTM_TESTING_TIME = 10

CONV_LSTM_LAYERS = 2
CONV_LSTM_FILTERS = [2, 4]
CONV_LSTM_KERNEL_SIZE = [[3, 3], [3, 3]]
CONV_LSTM_STRIDES = [[1, 1], [1, 1]]
CONV_LSTM_DROPOUTS = [0.25, 0.25]
CONV_LSTM_RNN_DROPOUTS = [0.25, 0.25]

# CONV_LSTM_LAYERS = 1
# CONV_LSTM_FILTERS = [16]
# CONV_LSTM_KERNEL_SIZE = [[7, 7]]
# CONV_LSTM_STRIDES = [[1, 1]]
# CONV_LSTM_DROPOUTS = [0.25]
# CONV_LSTM_RNN_DROPOUTS = [0.25]
#

CONV_LSTM_WIDE = 12
CONV_LSTM_HIGH = 12
CONV_LSTM_CHANNEL = 2

CONV_LSTM_MON_RAIO = 0.30

CONV_LSTM_IMS = False

CONV_LSTM_DATA_GENERATE_TIME = 3
CONV_LSTM_VALID_TEST = True

# ----------- CONV_LSTM Config ----------------------

CNNLSTM_N_EPOCH = 100
CNNLSTM_BATCH_SIZE = 64

CNNLSTM_IMS_STEP = 12
CNNLSTM_STEP = 26

CNNLSTM_BEST_CHECKPOINT = 95
CNNLSTM_TESTING_TIME = 10

CNNLSTM_LAYERS = 2
CNNLSTM_FILTERS = [16, 32]
CNNLSTM_KERNEL_SIZE = [[3, 3], [5, 5]]
CNNLSTM_STRIDES = [[1, 1], [2, 2]]
CNNLSTM_DROPOUTS = [0.25, 0.25]
CNNLSTM_RNN_DROPOUTS = [0.25, 0.25]

CNNLSTM_WIDE = 12
CNNLSTM_HIGH = 12
CNNLSTM_CHANNEL = 2

CNNLSTM_MON_RAIO = 0.30

CNNLSTM_IMS = False

CNNLSTM_DATA_GENERATE_TIME = 3
CNNLSTM_VALID_TEST = True


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
ALGS = ['fwbw-conv-lstm', 'conv-lstm', 'lstm-nn', 'arima', 'holt-winter', 'xgb', 'fwbw-lstm', 'fwbw-convlstm',
        'cnnlstm']
SCALERS = ['power-transform', 'standard-scaler', 'minmax-scaler', 'box-cox', 'robust-scaler', 'sd_scaler']


DATA_NAME = 'Abilene2d'

RUN_MODE = RUN_MODES[0]
ALG = ALGS[0]
GPU = 0
SCALER = SCALERS[5]

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
elif ALG == ALGS[7]:
    filters = ''
    kernel = ''
    for layer in range(FWBW_CONVLSTM_LAYERS):
        filters = filters + '{:02d}_'.format(FWBW_CONVLSTM_FILTERS[layer])
        kernel = kernel + '{:02d}_'.format(FWBW_CONVLSTM_KERNEL_SIZE[layer][0])

    TAG = 'mon_{:02d}_lstm_{:02d}_layers_{:02d}_filters_{}kernel_{}batch_{:03d}'.format(
        int(FWBW_CONVLSTM_MON_RAIO * 100),
        FWBW_CONVLSTM_STEP,
        FWBW_CONVLSTM_LAYERS,
        filters, kernel,
        FWBW_CONVLSTM_BATCH_SIZE)
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
elif ALG == ALGS[8]:
    filters = ''
    kernel = ''
    for layer in range(CNNLSTM_LAYERS):
        filters = filters + '{:02d}_'.format(CNNLSTM_FILTERS[layer])
        kernel = kernel + '{:02d}_'.format(CNNLSTM_KERNEL_SIZE[layer][0])

    TAG = 'mon_{:02d}_lstm_{:02d}_layers_{:02d}_filters_{}kernel_{}batch_{:03d}'.format(int(CNNLSTM_MON_RAIO * 100),
                                                                                        CNNLSTM_STEP,
                                                                                        CNNLSTM_LAYERS,
                                                                                        filters, kernel,
                                                                                        CNNLSTM_BATCH_SIZE)
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
elif ALG == ALGS[6]:
    TAG = 'mon_{:02d}_lstm_{:02d}_batch_{:03d}_hidden_{:03d}'.format(int(FWBW_LSTM_MON_RAIO * 100),
                                                                     FWBW_LSTM_STEP,
                                                                     FWBW_LSTM_BATCH_SIZE,
                                                                     FWBW_LSTM_HIDDEN_UNIT)
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


def set_comet_params_fwbw_convlstm():
    params = {
        'cnn_layers': FWBW_CONVLSTM_LAYERS,
        'epochs': FWBW_CONVLSTM_N_EPOCH,
        'batch_size': FWBW_CONVLSTM_BATCH_SIZE,
        'mon_ratio': FWBW_CONVLSTM_MON_RAIO,
        'lstm_step': FWBW_CONVLSTM_STEP,
        'random_action': FWBW_CONVLSTM_RANDOM_ACTION
    }

    for i in range(FWBW_CONVLSTM_LAYERS):
        params['layer{}_filter'.format(i + 1)] = FWBW_CONVLSTM_FILTERS[i]
        params['layer{}_kernel_size'.format(i + 1)] = FWBW_CONVLSTM_KERNEL_SIZE[i]
        params['layer{}_stride'.format(i + 1)] = FWBW_CONVLSTM_STRIDES[i]
        params['layer{}_dropout'.format(i + 1)] = FWBW_CONVLSTM_DROPOUTS[i]
        params['layer{}_rnn_dropout'.format(i + 1)] = FWBW_CONVLSTM_RNN_DROPOUTS[i]

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


def set_comet_params_cnnlstm():
    params = {
        'cnn_layers': CNNLSTM_LAYERS,
        'epochs': CNNLSTM_N_EPOCH,
        'batch_size': CNNLSTM_BATCH_SIZE,
        'mon_ratio': CNNLSTM_MON_RAIO,
        'lstm_step': CNNLSTM_STEP
    }

    for i in range(CNNLSTM_LAYERS):
        params['layer{}_filter'.format(i + 1)] = CNNLSTM_FILTERS[i]
        params['layer{}_kernel_size'.format(i + 1)] = CNNLSTM_KERNEL_SIZE[i]
        params['layer{}_stride'.format(i + 1)] = CNNLSTM_STRIDES[i]
        params['layer{}_dropout'.format(i + 1)] = CNNLSTM_DROPOUTS[i]
        params['layer{}_rnn_dropout'.format(i + 1)] = CNNLSTM_RNN_DROPOUTS[i]

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
