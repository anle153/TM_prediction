FWBW_LSTM_N_EPOCH = 10
FWBW_LSTM_BATCH_SIZE = 512
FWBW_LSTM_HIDDEN_UNIT = 128
FWBW_LSTM_DROPOUT = 0.5

FWBW_LSTM_DEEP = False
FWBW_LSTM_DEEP_NLAYERS = 3

FWBW_LSTM_STEP = 288
FWBW_LSTM_FEATURES = 2
FWBW_LSTM_IMS_STEP = 3

FWBW_LSTM_MON_RATIO = 0.40

FWBW_LSTM_BEST_CHECKPOINT = 20

FWBW_LSTM_IMS = False

FWBW_LSTM_VALID_TEST = True

FWBW_LSTM_TEST_DAYS = 25

FWBW_LSTM_HYPERPARAMS = [2.6, 1.0]

FLOW_SELECTIONS = ['random', 'fairness', 'weights']
FWBW_LSTM_FLOW_SELECTION = FLOW_SELECTIONS[2]

if FWBW_LSTM_FLOW_SELECTION == FLOW_SELECTIONS[0]:
    FWBW_LSTM_TESTING_TIME = 20
else:
    FWBW_LSTM_TESTING_TIME = 1

FWBW_LSTM_R = int(FWBW_LSTM_STEP / 3)

# ----------- RUNNING Config ----------------------

RUN_MODES = ['train', 'test', 'plot']
SCALERS = ['power-transform', 'standard-scaler', 'minmax-scaler', 'box-cox', 'robust-scaler', 'sd_scaler']
DATA_SETS = ['Abilene2d', 'Geant2d']

DATA_NAME = DATA_SETS[0]
RUN_MODE = RUN_MODES[1]
ALG = 'fwbw-lstm'
GPU = 1
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
TAG = 'mon_{:02d}_lstm_{:02d}_batch_{:03d}_hidden_{:03d}'.format(int(FWBW_LSTM_MON_RATIO * 100),
                                                                 FWBW_LSTM_STEP,
                                                                 FWBW_LSTM_BATCH_SIZE,
                                                                 FWBW_LSTM_HIDDEN_UNIT)

########################################################################################################################
