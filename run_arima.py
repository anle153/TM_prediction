import numpy as np

from algs.arima import test_arima_2
from common import Config_arima as Config


def print_arima_info():
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

    print('|--- MON_RATIO:\t{}'.format(Config.ARIMA_MON_RATIO))
    print('            -----------            ')

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        if Config.ARIMA_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.ARIMA_IMS_STEP))
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        print('|--- TESTING_TIME:\t{}'.format(Config.ARIMA_TESTING_TIME))
    else:
        raise Exception('Unknown RUN_MODE!')
    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


if __name__ == '__main__':
    data = np.load(Config.DATA_PATH + '{}.npy'.format(Config.DATA_NAME))
    print_arima_info()
    test_arima_2(data)
