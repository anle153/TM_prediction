# ----------------- FWBW_LSTM Config ---------------
RES_FWBW_LSTM_N_EPOCH = 20
RES_FWBW_LSTM_BATCH_SIZE = 512
RES_FWBW_LSTM_HIDDEN_UNIT = 128
RES_FWBW_LSTM_DROPOUT = 0.5

RES_FWBW_LSTM_DEEP = False
RES_FWBW_LSTM_DEEP_NLAYERS = 3

RES_FWBW_LSTM_STEP = 30
RES_FWBW_LSTM_FEATURES = 2
RES_FWBW_LSTM_IMS_STEP = 12

RES_FWBW_LSTM_MON_RAIO = 0.30

RES_FWBW_LSTM_BEST_CHECKPOINT = 10
RES_FWBW_LSTM_TESTING_TIME = 10

RES_FWBW_LSTM_IMS = False

RES_FWBW_LSTM_VALID_TEST = True
RES_FWBW_LSTM_RANDOM_ACTION = True
RES_FWBW_LSTM_TEST_DAYS = 10
RES_FWBW_LSTM_HYPERPARAMS = [1.5, 2.0, 1.0]

# ----------- RUNNING Config ----------------------

RUN_MODES = ['train', 'test', 'plot']
SCALERS = ['power-transform', 'standard-scaler', 'minmax-scaler', 'box-cox', 'robust-scaler', 'sd_scaler']
DATA_SETS = ['Abilene2d', 'Geant2d']

DATA_NAME = DATA_SETS[0]
RUN_MODE = RUN_MODES[0]
ALG = 'res-fwbw-lstm'
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

TAG = 'mon_{:02d}_lstm_{:02d}_batch_{:03d}_hidden_{:03d}'.format(int(RES_FWBW_LSTM_MON_RAIO * 100),
                                                                 RES_FWBW_LSTM_STEP,
                                                                 RES_FWBW_LSTM_BATCH_SIZE,
                                                                 RES_FWBW_LSTM_HIDDEN_UNIT)
