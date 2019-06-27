# ----------- ARIMA Config ----------------------
ARIMA_UPDATE = 7
ARIMA_TESTING_TIME = 3
ARIMA_MON_RATIO = 0.90
ARIMA_STEP = 30
ARIMA_IMS_STEP = 3
ARIMA_IMS = True
ARIMA_TEST_DAYS = 1
# ----------------------------------------------


RUN_MODES = ['train', 'test', 'plot']
SCALERS = ['power-transform', 'standard-scaler', 'minmax-scaler', 'box-cox', 'robust-scaler', 'sd_scaler']
DATA_SETS = ['Abilene2d', 'Geant2d']

DATA_NAME = DATA_SETS[0]
RUN_MODE = RUN_MODES[1]
ALG = 'arima'
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
TAG = 'mon_{:2d}_steps_{:2d}'.format(int(ARIMA_MON_RATIO * 100), ARIMA_STEP)
