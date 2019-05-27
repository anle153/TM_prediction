import numpy as np
from comet_ml import Experiment

from common import Config
from common.Config import DATA_PATH
from common.DataHelper import create_abilene_data_2d, create_abilene_data_3d, create_Geant2d, create_Geant3d
from common.cmd_utils import print_info


def train():
    data = np.load(DATA_PATH + '{}.npy'.format(Config.DATA_NAME))

    if Config.ALG == Config.ALGS[0]:
        from algs.fwbw_conv_lstm import train_fwbw_conv_lstm
        experiment = Experiment(project_name='tmp-fwbw-conv-lstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        train_fwbw_conv_lstm(data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[1]:
        from algs.conv_lstm2 import train_conv_lstm
        experiment = Experiment(project_name='tmp-conv-lstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        train_conv_lstm(data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[2]:
        from algs.lstm_nn import train_lstm_nn
        experiment = Experiment(project_name='tmp-lstm-nn', api_key='RzFughRSAY2raEySCf69bjiFn')
        train_lstm_nn(data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[3]:
        from algs.arima import train_arima
        train_arima(data=data)
    elif Config.ALG == Config.ALGS[4]:
        from algs.holt_winter import train_holt_winter
        train_holt_winter(data=data)
    elif Config.ALG == Config.ALGS[5]:
        from algs.boosting_based import train_xgboost
        train_xgboost(data=data)
    elif Config.ALG == Config.ALGS[6]:
        from algs.fwbw_lstm import train_fwbw_lstm
        experiment = Experiment(project_name='tmp-fwbw-lstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        train_fwbw_lstm(data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[7]:
        from algs.convlstm_fwbw import train_fwbw_convlstm
        experiment = Experiment(project_name='tmp-fwbw-convlstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        train_fwbw_convlstm(data=data, experiment=experiment)
    else:
        raise ValueError('Unkown alg!')


def test():
    data = np.load(DATA_PATH + '{}.npy'.format(Config.DATA_NAME))

    if Config.ALG == Config.ALGS[0]:
        from algs.fwbw_conv_lstm import test_fwbw_conv_lstm
        experiment = Experiment(project_name='tmp-fwbw-conv-lstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        test_fwbw_conv_lstm(data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[1]:
        from algs.conv_lstm import test_conv_lstm
        experiment = Experiment(project_name='tmp-conv-lstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        test_conv_lstm(data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[2]:
        from algs.lstm_nn import test_lstm_nn
        experiment = Experiment(project_name='tmp-lstm-nn', api_key='RzFughRSAY2raEySCf69bjiFn')
        test_lstm_nn(data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[3]:
        from algs.arima import test_arima
        test_arima(data=data)
    elif Config.ALG == Config.ALGS[4]:
        from algs.holt_winter import test_holt_winter
        test_holt_winter(data=data)
    elif Config.ALG == Config.ALGS[7]:
        from algs.convlstm_fwbw import test_fwbw_convlstm
        experiment = Experiment(project_name='tmp-fwbw-convlstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        test_fwbw_convlstm(data=data, experiment=experiment)
    # elif Config.ALG == Config.ALGS[5]:
    #     from algs.boosting_based import run_test
    #     test_holt_winter(data=data)
    else:
        raise ValueError('Unkown alg!')


def main():
    print_info()

    data_name = Config.DATA_NAME

    import os
    if not os.path.isfile(Config.DATA_PATH + '{}.npy'.format(Config.DATA_NAME)):
        if data_name == 'Abilene':
            create_abilene_data_3d('/home/anle/AbileneTM-all/')
        elif data_name == 'Abilene2d':
            create_abilene_data_2d('/home/anle/AbileneTM-all/')
        elif data_name == 'Geant':
            create_Geant3d()
        elif data_name == 'Geant2d':
            create_Geant2d()
        else:
            raise ('Unknown dataset name!')

    if Config.RUN_MODE == Config.RUN_MODES[0]:
        train()
    elif Config.RUN_MODE == Config.RUN_MODES[1]:
        test()
    elif Config.RUN_MODE == Config.RUN_MODES[2]:
        from common.ResultProcessing import plot_pred_results
        plot_pred_results(Config.DATA_NAME, Config.ALG, Config.TAG, 12)
    else:
        raise Exception('Unknown RUN_MODE!')

    return


if __name__ == '__main__':
    # create_Geant2d(save_csv=True)
    # create_abilene_data_2d(path='/home/anle/AbileneTM-all/', save_csv=True)
    main()
