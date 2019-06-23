CONV_LSTM_N_EPOCH = 50
CONV_LSTM_BATCH_SIZE = 512

CONV_LSTM_IMS_STEP = 12
CONV_LSTM_STEP = 30

CONV_LSTM_BEST_CHECKPOINT = 50

CONV_LSTM_LAYERS = 2
CONV_LSTM_FILTERS = [2, 2]
CONV_LSTM_KERNEL_SIZE = [[3, 3], [3, 3]]
CONV_LSTM_STRIDES = [[1, 1], [1, 1]]
CONV_LSTM_DROPOUTS = [0.5, 0.5]
CONV_LSTM_RNN_DROPOUTS = [0.5, 0.5]

CONV_LSTM_WIDE = 12
CONV_LSTM_HIGH = 12
CONV_LSTM_CHANNEL = 2

CONV_LSTM_MON_RATIO = 0.30

CONV_LSTM_IMS = False

CONV_LSTM_VALID_TEST = True

FLOW_SELECTIONS = ['random', 'fairness', 'weights']
CONV_LSTM_FLOW_SELECTION = FLOW_SELECTIONS[0]

if CONV_LSTM_FLOW_SELECTION == FLOW_SELECTIONS[0]:
    CONV_LSTM_TESTING_TIME = 50
else:
    CONV_LSTM_TESTING_TIME = 1

RUN_MODES = ['train', 'test', 'plot']
SCALERS = ['power-transform', 'standard-scaler', 'minmax-scaler', 'box-cox', 'robust-scaler', 'sd_scaler']
DATA_SETS = ['Abilene2d', 'Geant2d']

DATA_NAME = DATA_SETS[0]
RUN_MODE = RUN_MODES[0]
ALG = 'conv-lstm'
GPU = 1
SCALER = SCALERS[5]

# --------------- Data Config -----------------

DATA_PATH = './Dataset/'
MODEL_SAVE = './trained_models/'
RESULTS_PATH = './results/'
ADDED_RESULT_NAME = 'random'

ABILENE_DAY_SIZE = 288
GEANT_DAY_SIZE = 96

NUM_DAYS = 160

# -----------------------------------------------------------------------------------------------------------------------
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
