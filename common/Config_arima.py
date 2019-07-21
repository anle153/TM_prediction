# ----------- ARIMA Config ----------------------
ARIMA_UPDATE = 7
ARIMA_TESTING_TIME = 1
ARIMA_MON_RATIO = 0.40
ARIMA_STEP = 36
ARIMA_IMS_STEP = 3
ARIMA_IMS = False
ARIMA_TEST_DAYS = 0.1
# ----------------------------------------------


RUN_MODES = ['train', 'test', 'plot']
SCALERS = ['power-transform']
DATA_SETS = ['Abilene2d', 'Geant2d']

DATA_NAME = DATA_SETS[0]
RUN_MODE = RUN_MODES[1]
ALG = 'arima'
GPU = 0
SCALER = SCALERS[0]

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
