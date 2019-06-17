from common import Config


def print_fwbw_conv_lstm_info():
    print('|--- MON_RATIO:\t{}'.format(Config.FWBW_CONV_LSTM_MON_RAIO))
    print('            -----------            ')

    print('|--- CONV_LAYERS:\t{}'.format(Config.FWBW_CONV_LSTM_LAYERS))
    print('|--- FILTERS:\t{}'.format(Config.FWBW_CONV_LSTM_FILTERS))
    print('|--- KERNEL_SIZE:\t{}'.format(Config.FWBW_CONV_LSTM_KERNEL_SIZE))
    print('|--- STRIDES:\t{}'.format(Config.FWBW_CONV_LSTM_STRIDES))
    print('|--- DROPOUTS:\t{}'.format(Config.FWBW_CONV_LSTM_DROPOUTS))
    print('|--- RNN_DROPOUTS:\t{}'.format(Config.FWBW_CONV_LSTM_RNN_DROPOUTS))
    print('|--- WIDE:\t{}'.format(Config.FWBW_CONV_LSTM_WIDE))
    print('|--- HIGH:\t{}'.format(Config.FWBW_CONV_LSTM_HIGH))
    print('|--- CHANNEL:\t{}'.format(Config.FWBW_CONV_LSTM_CHANNEL))
    print('            -----------            ')

    print('|--- RANDOM_ACTION:\t{}'.format(Config.FWBW_CONV_LSTM_RANDOM_ACTION))

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        print('|--- N_EPOCH:\t{}'.format(Config.FWBW_CONV_LSTM_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.FWBW_CONV_LSTM_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.FWBW_CONV_LSTM_STEP))
        if Config.FWBW_CONV_LSTM_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.FWBW_CONV_LSTM_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.FWBW_CONV_LSTM_TESTING_TIME))
        print('|--- FWBW_BEST_CHECKPOINT:\t{}'.format(Config.FWBW_CONV_LSTM_BEST_CHECKPOINT))
        if not Config.FWBW_CONV_LSTM_RANDOM_ACTION:
            print('|--- HYPERPARAMS:\t{}'.format(Config.FWBW_CONV_LSTM_HYPERPARAMS))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_fwbw_convlstm_info():
    print('|--- MON_RATIO:\t{}'.format(Config.FWBW_CONVLSTM_MON_RAIO))
    print('            -----------            ')

    print('|--- CONV_LAYERS:\t{}'.format(Config.FWBW_CONVLSTM_LAYERS))
    print('|--- FILTERS:\t{}'.format(Config.FWBW_CONVLSTM_FILTERS))
    print('|--- KERNEL_SIZE:\t{}'.format(Config.FWBW_CONVLSTM_KERNEL_SIZE))
    print('|--- STRIDES:\t{}'.format(Config.FWBW_CONVLSTM_STRIDES))
    print('|--- DROPOUTS:\t{}'.format(Config.FWBW_CONVLSTM_DROPOUTS))
    print('|--- RNN_DROPOUTS:\t{}'.format(Config.FWBW_CONVLSTM_RNN_DROPOUTS))
    print('|--- WIDE:\t{}'.format(Config.FWBW_CONVLSTM_WIDE))
    print('|--- HIGH:\t{}'.format(Config.FWBW_CONVLSTM_HIGH))
    print('|--- CHANNEL:\t{}'.format(Config.FWBW_CONVLSTM_CHANNEL))
    print('            -----------            ')

    print('|--- RANDOM_ACTION:\t{}'.format(Config.FWBW_CONVLSTM_RANDOM_ACTION))

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        print('|--- N_EPOCH:\t{}'.format(Config.FWBW_CONVLSTM_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.FWBW_CONVLSTM_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.FWBW_CONVLSTM_STEP))
        if Config.FWBW_CONVLSTM_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.FWBW_CONVLSTM_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.FWBW_CONVLSTM_TESTING_TIME))
        print('|--- BEST_CHECKPOINT:\t{}'.format(Config.FWBW_CONVLSTM_BEST_CHECKPOINT))
        if not Config.FWBW_CONVLSTM_RANDOM_ACTION:
            print('|--- HYPERPARAMS:\t{}'.format(Config.FWBW_CONVLSTM_HYPERPARAMS))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_conv_lstm_info():
    print('|--- MON_RATIO:\t{}'.format(Config.CONV_LSTM_MON_RAIO))
    print('|--- DATA_GENERATE_TIME:\t{}'.format(Config.CONV_LSTM_DATA_GENERATE_TIME))
    print('            -----------            ')

    print('|--- CONV_LAYERS:\t{}'.format(Config.CONV_LSTM_LAYERS))
    print('|--- FILTERS:\t{}'.format(Config.CONV_LSTM_FILTERS))
    print('|--- KERNEL_SIZE:\t{}'.format(Config.CONV_LSTM_KERNEL_SIZE))
    print('|--- STRIDES:\t{}'.format(Config.CONV_LSTM_STRIDES))
    print('|--- DROPOUTS:\t{}'.format(Config.CONV_LSTM_DROPOUTS))
    print('|--- RNN_DROPOUTS:\t{}'.format(Config.CONV_LSTM_RNN_DROPOUTS))

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        print('|--- N_EPOCH:\t{}'.format(Config.CONV_LSTM_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.CONV_LSTM_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.CONV_LSTM_STEP))
        if Config.CONV_LSTM_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.CONV_LSTM_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.CONV_LSTM_TESTING_TIME))
        print('|--- BEST_CHECKPOINT:\t{}'.format(Config.CONV_LSTM_BEST_CHECKPOINT))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_cnnlstm_info():
    print('|--- MON_RATIO:\t{}'.format(Config.CNNLSTM_MON_RAIO))
    print('|--- DATA_GENERATE_TIME:\t{}'.format(Config.CNNLSTM_DATA_GENERATE_TIME))
    print('            -----------            ')

    print('|--- CONV_LAYERS:\t{}'.format(Config.CNNLSTM_LAYERS))
    print('|--- FILTERS:\t{}'.format(Config.CNNLSTM_FILTERS))
    print('|--- KERNEL_SIZE:\t{}'.format(Config.CNNLSTM_KERNEL_SIZE))
    print('|--- STRIDES:\t{}'.format(Config.CNNLSTM_STRIDES))
    print('|--- DROPOUTS:\t{}'.format(Config.CNNLSTM_DROPOUTS))
    print('|--- RNN_DROPOUTS:\t{}'.format(Config.CNNLSTM_RNN_DROPOUTS))

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        print('|--- N_EPOCH:\t{}'.format(Config.CNNLSTM_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.CNNLSTM_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.CNNLSTM_STEP))
        if Config.CNNLSTM_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.CNNLSTM_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.CNNLSTM_TESTING_TIME))
        print('|--- BEST_CHECKPOINT:\t{}'.format(Config.CNNLSTM_BEST_CHECKPOINT))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_lstm_info():
    print('|--- MON_RATIO:\t{}'.format(Config.LSTM_MON_RAIO))
    print('            -----------            ')

    print('|--- LSTM_DEEP:\t{}'.format(Config.LSTM_DEEP))
    if Config.LSTM_DEEP:
        print('|--- LSTM_DEEP_NLAYERS:\t{}'.format(Config.LSTM_DEEP_NLAYERS))
    print('|--- LSTM_DROPOUT:\t{}'.format(Config.LSTM_DROPOUT))
    print('|--- LSTM_HIDDEN_UNIT:\t{}'.format(Config.LSTM_HIDDEN_UNIT))

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        print('|--- N_EPOCH:\t{}'.format(Config.LSTM_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.LSTM_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.LSTM_STEP))
        if Config.LSTM_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.LSTM_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.LSTM_TESTING_TIME))
        print('|--- BEST_CHECKPOINT:\t{}'.format(Config.LSTM_BEST_CHECKPOINT))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_fwbw_lstm_info():
    print('|--- MON_RATIO:\t{}'.format(Config.FWBW_LSTM_MON_RAIO))
    print('            -----------            ')

    print('|--- LSTM_DEEP:\t{}'.format(Config.FWBW_LSTM_DEEP))
    if Config.FWBW_LSTM_DEEP:
        print('|--- LSTM_DEEP_NLAYERS:\t{}'.format(Config.FWBW_LSTM_DEEP_NLAYERS))
    print('|--- LSTM_DROPOUT:\t{}'.format(Config.FWBW_LSTM_DROPOUT))
    print('|--- LSTM_HIDDEN_UNIT:\t{}'.format(Config.FWBW_LSTM_HIDDEN_UNIT))
    print('|--- RANDOM_ACTION:\t{}'.format(Config.FWBW_LSTM_RANDOM_ACTION))
    if not Config.FWBW_LSTM_RANDOM_ACTION:
        print('|--- FWBW_LSTM_HYPERPARAMS:\t{}'.format(Config.FWBW_LSTM_HYPERPARAMS))

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        print('|--- N_EPOCH:\t{}'.format(Config.FWBW_LSTM_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.FWBW_LSTM_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.FWBW_LSTM_STEP))
        if Config.FWBW_LSTM_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.FWBW_LSTM_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.FWBW_LSTM_TESTING_TIME))
        print('|--- BEST_CHECKPOINT:\t{}'.format(Config.FWBW_LSTM_BEST_CHECKPOINT))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_fwbw_lstm_2_info():
    print('|--- MON_RATIO:\t{}'.format(Config.FWBW_LSTM_2_MON_RAIO))
    print('            -----------            ')

    print('|--- LSTM_DEEP:\t{}'.format(Config.FWBW_LSTM_2_DEEP))
    if Config.FWBW_LSTM_2_DEEP:
        print('|--- LSTM_DEEP_NLAYERS:\t{}'.format(Config.FWBW_LSTM_2_DEEP_NLAYERS))
    print('|--- LSTM_DROPOUT:\t{}'.format(Config.FWBW_LSTM_2_DROPOUT))
    print('|--- LSTM_HIDDEN_UNIT:\t{}'.format(Config.FWBW_LSTM_2_HIDDEN_UNIT))
    print('|--- RANDOM_ACTION:\t{}'.format(Config.FWBW_LSTM_2_RANDOM_ACTION))
    if not Config.FWBW_LSTM_2_RANDOM_ACTION:
        print('|--- FWBW_LSTM_HYPERPARAMS:\t{}'.format(Config.FWBW_LSTM_2_HYPERPARAMS))

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        print('|--- N_EPOCH:\t{}'.format(Config.FWBW_LSTM_2_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.FWBW_LSTM_2_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.FWBW_LSTM_2_STEP))
        if Config.FWBW_LSTM_2_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.FWBW_LSTM_2_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.FWBW_LSTM_2_TESTING_TIME))
        print('|--- BEST_CHECKPOINT:\t{}'.format(Config.FWBW_LSTM_2_BEST_CHECKPOINT))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_arima_info():
    print('|--- MON_RATIO:\t{}'.format(Config.ARIMA_MON_RATIO))
    print('            -----------            ')

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        if Config.ARIMA_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.ARIMA_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.ARIMA_TESTING_TIME))
        print('|--- ARIMA_UPDATE:\t{}'.format(Config.ARIMA_UPDATE))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_holt_winter_info():
    print('|--- MON_RATIO:\t{}'.format(Config.HOLT_WINTER_MON_RATIO))
    print('            -----------            ')

    print('|--- HOLT_WINTER_SEASONAL:\t{}'.format(Config.HOLT_WINTER_SEASONAL))
    print('|--- HOLT_WINTER_TREND:\t{}'.format(Config.HOLT_WINTER_TREND))

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        if Config.HOLT_WINTER_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.HOLT_WINTER_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.HOLT_WINTER_TESTING_TIME))
        print('|--- HOLT_WINTER_UPDATE:\t{}'.format(Config.HOLT_WINTER_UPDATE))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_xgb_info():
    print('|--- MON_RATIO:\t{}'.format(Config.XGB_MON_RATIO))
    print('            -----------            ')
    print('|--- STEPS:\t{}'.format(Config.XGB_STEP))
    print('|--- FEATURES:\t{}'.format(Config.XGB_FEATURES))
    print('|--- N_JOBS:\t{}'.format(Config.XGB_NJOBS))

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        if Config.XGB_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.XGB_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.XGB_TESTING_TIME))
    else:
        raise Exception('Unknown RUN_MODE!')


def print_info():

    print('----------------------- INFO -----------------------')
    if not Config.ALL_DATA:
        print('|--- Train/Test with {}d of data'.format(Config.NUM_DAYS))
    else:
        print('|--- Train/Test with ALL of data'.format(Config.NUM_DAYS))

    print('|--- MODE:\t{}'.format(Config.RUN_MODE))
    print('|--- ALG:\t{}'.format(Config.ALG))
    print('|--- TAG:\t{}'.format(Config.TAG))
    print('|--- DATA:\t{}'.format(Config.DATA_NAME))
    print('|--- GPU:\t{}'.format(Config.GPU))
    if Config.SCALER == Config.SCALERS[0]:
        print('|--- SCALER:\t{}'.format(Config.SCALERS[0]))
    elif Config.SCALER == Config.SCALERS[1]:
        print('|--- SCALER:\t{}'.format(Config.SCALERS[1]))
    elif Config.SCALER == Config.SCALERS[2]:
        print('|--- SCALER:\t{}'.format(Config.SCALERS[2]))
    elif Config.SCALER == Config.SCALERS[3]:
        print('|--- SCALER:\t{}'.format(Config.SCALERS[3]))
    elif Config.SCALER == Config.SCALERS[4]:
        print('|--- SCALER:\t{}'.format(Config.SCALERS[4]))
    elif Config.SCALER == Config.SCALERS[5]:
        print('|--- SCALER:\t{}'.format(Config.SCALERS[5]))
    else:
        raise Exception('Unknown scaler!')

    if Config.ALG == Config.ALGS[0]:
        print_fwbw_conv_lstm_info()
    elif Config.ALG == Config.ALGS[1]:
        print_conv_lstm_info()
    elif Config.ALG == Config.ALGS[2]:
        print_lstm_info()
    elif Config.ALG == Config.ALGS[3]:
        print_arima_info()
    elif Config.ALG == Config.ALGS[4]:
        print_holt_winter_info()
    elif Config.ALG == Config.ALGS[5]:
        print_xgb_info()
    elif Config.ALG == Config.ALGS[6]:
        print_fwbw_lstm_info()
    elif Config.ALG == Config.ALGS[7]:
        print_fwbw_convlstm_info()
    elif Config.ALG == Config.ALGS[8]:
        print_cnnlstm_info()
    elif Config.ALG == Config.ALGS[9]:
        print_fwbw_lstm_2_info()

    else:
        raise ValueError('Unkown alg!')
    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')
