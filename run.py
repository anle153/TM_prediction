import sys

import numpy as np

from common.Config import DATA_PATH
from common.cmd_utils import parse_unknown_args, common_arg_parser, print_info

from common.DataHelper import create_abilene_data_2d, create_abilene_data_3d, create_Geant2d, create_Geant3d
from common import Config


def train(args):
    alg_name = args.alg

    data = np.load(DATA_PATH + '{}.npy'.format(args.data_name))

    if 'fwbw-conv-lstm' in alg_name or 'fwbw-convlstm' in alg_name:
        from algs.fwbw_conv_lstm import train_fwbw_conv_lstm
        train_fwbw_conv_lstm(args=args, data=data)
    elif 'conv-lstm' in alg_name or 'convlstm' in alg_name:
        from algs.conv_lstm import train_conv_lstm
        train_conv_lstm(args=args, data=data)
    elif 'lstm-nn' in alg_name:
        from algs.lstm_nn import train_lstm_nn
        train_lstm_nn(args=args, data=data)
    elif 'arima' in alg_name:
        from algs.arima import train_arima
        train_arima(args=args, data=data)
    elif 'holt-winter' in alg_name:
        from algs.holt_winter import train_holt_winter
        train_holt_winter(args=args, data=data)
    else:
        raise ValueError('Unkown alg!')


def test(args):
    alg_name = args.alg

    data = np.load(DATA_PATH + '{}.npy'.format(args.data_name))

    if 'fwbw-conv-lstm' in alg_name or 'fwbw-convlstm' in alg_name:
        from algs.fwbw_conv_lstm import test_fwbw_conv_lstm
        test_fwbw_conv_lstm(args=args, data=data)
    elif 'conv-lstm' in alg_name or 'convlstm' in alg_name:
        from algs.conv_lstm import test_conv_lstm
        test_conv_lstm(args=args, data=data)
    elif 'lstm-nn' in alg_name:
        from algs.lstm_nn import test_lstm_nn
        test_lstm_nn(args=args, data=data)
    elif 'arima' in alg_name:
        from algs.arima import test_arima
        test_arima(args=args, data=data)
    elif 'holt-winter' in alg_name:
        from algs.holt_winter import test_holt_winter
        test_holt_winter(args=args, data=data)
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
        plot_pred_results(args.data_name, args.alg, args.tag, 10, 5)

    return


if __name__ == '__main__':
    main(sys.argv)
