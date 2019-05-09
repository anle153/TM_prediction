import sys

import numpy as np
from comet_ml import Experiment

from common import Config
from common.Config import DATA_PATH
from common.DataHelper import create_abilene_data_2d, create_abilene_data_3d, create_Geant2d, create_Geant3d
from common.cmd_utils import parse_unknown_args, common_arg_parser, print_info


def train(args):
    data = np.load(DATA_PATH + '{}.npy'.format(Config.ALG))

    if Config.ALG == Config.ALGS[0]:
        from algs.fwbw_conv_lstm import train_fwbw_conv_lstm
        experiment = Experiment(project_name='tmp-fwbw-conv-lstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        train_fwbw_conv_lstm(args=args, data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[1]:
        from algs.conv_lstm import train_conv_lstm
        experiment = Experiment(project_name='tmp-conv-lstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        train_conv_lstm(args=args, data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[2]:
        from algs.lstm_nn import train_lstm_nn
        experiment = Experiment(project_name='tmp-lstm-nn', api_key='RzFughRSAY2raEySCf69bjiFn')
        train_lstm_nn(args=args, data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[3]:
        from algs.arima import train_arima
        train_arima(args=args, data=data)
    elif Config.ALG == Config.ALGS[4]:
        from algs.holt_winter import train_holt_winter
        train_holt_winter(args=args, data=data)
    elif Config.ALG == Config.ALGS[5]:
        from algs.boosting_based import train_xgboost
        train_xgboost(args=args, data=data)
    else:
        raise ValueError('Unkown alg!')


def test(args):
    alg_name = args.alg

    data = np.load(DATA_PATH + '{}.npy'.format(args.data_name))

    if Config.ALG == Config.ALGS[0]:
        from algs.fwbw_conv_lstm import test_fwbw_conv_lstm
        experiment = Experiment(project_name='tmp-fwbw-conv-lstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        test_fwbw_conv_lstm(args=args, data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[1]:
        from algs.conv_lstm import test_conv_lstm
        experiment = Experiment(project_name='tmp-conv-lstm', api_key='RzFughRSAY2raEySCf69bjiFn')
        test_conv_lstm(args=args, data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[2]:
        from algs.lstm_nn import test_lstm_nn
        experiment = Experiment(project_name='tmp-lstm-nn', api_key='RzFughRSAY2raEySCf69bjiFn')
        test_lstm_nn(args=args, data=data, experiment=experiment)
    elif Config.ALG == Config.ALGS[3]:
        from algs.arima import test_arima
        test_arima(args=args, data=data)
    elif Config.ALG == Config.ALGS[4]:
        from algs.holt_winter import test_holt_winter
        test_holt_winter(args=args, data=data)
    # elif Config.ALG == Config.ALGS[5]:
    #     from algs.boosting_based import run_test
    #     test_holt_winter(args=args, data=data)
    else:
        raise ValueError('Unkown alg!')


def parse_cmdline_kwargs(args):
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    print_info(args)

    data_name = args.data_name

    import os
    if not os.path.isfile(Config.DATA_PATH + '{}.npy'.format(args.data_name)):
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

    if 'train' in args.run_mode or 'training' in args.run_mode:
        train(args)
    elif 'test' in args.run_mode:
        test(args)
    else:
        from common.ResultProcessing import plot_pred_results
        plot_pred_results(args.data_name, args.alg, args.tag, 12)

    return


if __name__ == '__main__':
    # create_Geant2d(save_csv=True)
    # create_abilene_data_2d(path='/home/anle/AbileneTM-all/', save_csv=True)
    main(sys.argv)
