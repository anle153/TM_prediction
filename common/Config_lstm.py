FLOW_SELECTIONS = ['random', 'fairness', 'weights']

LSTM_N_EPOCH = 20
LSTM_BATCH_SIZE = 512
LSTM_HIDDEN_UNIT = 64
LSTM_DROPOUT = 0.5

LSTM_DEEP = False
LSTM_DEEP_NLAYERS = 3

LSTM_STEP = 30
LSTM_FEATURES = 2
LSTM_IMS_STEP = 3

LSTM_MON_RATIO = 0.90

LSTM_BEST_CHECKPOINT = 2

LSTM_IMS = True
LSTM_VALID_TEST = True

LSTM_TEST_DAYS = 25

LSTM_FLOW_SELECTION = FLOW_SELECTIONS[1]

if LSTM_FLOW_SELECTION == FLOW_SELECTIONS[0]:
    if LSTM_IMS:
        LSTM_TESTING_TIME = 20
    else:
        LSTM_TESTING_TIME = 50
else:
    LSTM_TESTING_TIME = 1


DATA_PATH = './Dataset/'
MODEL_SAVE = './trained_models/'
RESULTS_PATH = './results/'
ADDED_RESULT_NAME = 'random'

ABILENE_DAY_SIZE = 288
GEANT_DAY_SIZE = 96

ALL_DATA = True
NUM_DAYS = 160

RUN_MODES = ['train', 'test', 'plot']
SCALERS = ['power-transform', 'standard-scaler', 'minmax-scaler', 'box-cox', 'robust-scaler', 'sd_scaler']
DATA_SETS = ['Abilene2d', 'Geant2d']

DATA_NAME = DATA_SETS[0]
RUN_MODE = RUN_MODES[1]
ALG = 'lstm-nn'
GPU = 0
SCALER = SCALERS[5]

TAG = 'mon_{:02d}_lstm_{:02d}_batch_{:03d}_hidden_{:03d}'.format(int(LSTM_MON_RATIO * 100),
                                                                 LSTM_STEP,
                                                                 LSTM_BATCH_SIZE,
                                                                 LSTM_HIDDEN_UNIT)
