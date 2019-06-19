RES_LSTM_N_EPOCH = 20
RES_LSTM_BATCH_SIZE = 512
RES_LSTM_HIDDEN_UNIT = 64
RES_LSTM_DROPOUT = 0.5

RES_LSTM_DEEP = False
RES_LSTM_DEEP_NLAYERS = 3

RES_LSTM_STEP = 30
RES_LSTM_FEATURES = 2
RES_LSTM_IMS_STEP = 12

RES_LSTM_MON_RAIO = 0.30

RES_LSTM_BEST_CHECKPOINT = 13
RES_LSTM_TESTING_TIME = 10

RES_LSTM_IMS = False
RES_LSTM_VALID_TEST = True

RES_LSTM_TEST_DAYS = 25

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
ALG = 'res-lstm'
GPU = 0
SCALER = SCALERS[5]

TAG = 'mon_{:02d}_lstm_{:02d}_batch_{:03d}_hidden_{:03d}'.format(int(RES_LSTM_MON_RAIO * 100),
                                                                 RES_LSTM_STEP,
                                                                 RES_LSTM_BATCH_SIZE,
                                                                 RES_LSTM_HIDDEN_UNIT)
