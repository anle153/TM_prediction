# ----------------- FWBW_LSTM Config ---------------
FWBW_LSTM_2_N_EPOCH = 20
FWBW_LSTM_2_BATCH_SIZE = 512
FWBW_LSTM_2_HIDDEN_UNIT = 128
FWBW_LSTM_2_DROPOUT = 0.5

FWBW_LSTM_2_DEEP = False
FWBW_LSTM_2_DEEP_NLAYERS = 3

FWBW_LSTM_2_STEP = 30
FWBW_LSTM_2_FEATURES = 2
FWBW_LSTM_2_IMS_STEP = 12

FWBW_LSTM_2_MON_RAIO = 0.30

FWBW_LSTM_2_BEST_CHECKPOINT = 12
FWBW_LSTM_2_TESTING_TIME = 10

FWBW_LSTM_2_IMS = False

FWBW_LSTM_2_VALID_TEST = True
FWBW_LSTM_2_RANDOM_ACTION = True
FWBW_LSTM_2_TEST_DAYS = 25
FWBW_LSTM_2_HYPERPARAMS = [1.5, 2.0, 1.0]

# ----------------------------------------------

RUN_MODES = ['train', 'test', 'plot']
ALGS = ['fwbw-conv-lstm', 'conv-lstm', 'lstm-nn', 'arima', 'holt-winter', 'xgb', 'fwbw-lstm', 'fwbw-convlstm',
        'cnnlstm', 'fwbw_lstm_2', 'res-lstm']
SCALERS = ['power-transform', 'standard-scaler', 'minmax-scaler', 'box-cox', 'robust-scaler', 'sd_scaler']
DATA_SETS = ['Abilene2d', 'Geant2d']

DATA_NAME = DATA_SETS[0]
RUN_MODE = RUN_MODES[0]
ALG = 'fwbw-lstm-2'
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

TAG = 'mon_{:02d}_lstm_{:02d}_batch_{:03d}_hidden_{:03d}'.format(int(FWBW_LSTM_2_MON_RAIO * 100),
                                                                 FWBW_LSTM_2_STEP,
                                                                 FWBW_LSTM_2_BATCH_SIZE,
                                                                 FWBW_LSTM_2_HIDDEN_UNIT)
