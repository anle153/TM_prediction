import sys

import numpy as np

from common.Config import DATA_PATH
from common.cmd_utils import parse_unknown_args, common_arg_parser

from common.DataHelper import create_abilene_data_2d, create_abilene_data_3d, create_Geant2d
from common import Config


def print_info(args):
    alg_name = args.alg
    data_name = args.data_name
    tag = args.tag
    gpu = args.gpu

    print('----------------------- INFO -----------------------')
    if not Config.ALL_DATA:
        print('|--- Train/Test with {}d of data'.format(Config.NUM_DAYS))
    else:
        print('|--- Train/Test with ALL of data'.format(Config.NUM_DAYS))

    print('|--- Mode:\t{}'.format(args.run_mode))
    print('|--- Alg:\t{}'.format(alg_name))
    print('|--- Tag:\t{}'.format(tag))
    print('|--- Data:\t{}'.format(data_name))
    print('|--- GPU:\t{}'.format(gpu))
    print('|--- MON_RATIO:\t{}'.format(Config.MON_RAIO))
    print('            -----------            ')

    if 'lstm' in alg_name:
        if 'train' in args.run_mode or 'training' in args.run_mode:
            print('|--- N_EPOCH:\t{}'.format(Config.N_EPOCH))
            print('|--- BATCH_SIZE:\t{}'.format(Config.BATCH_SIZE))
            print('|--- NUM_ITER:\t{}'.format(Config.NUM_ITER))
            print('|--- LSTM_STEP:\t{}'.format(Config.LSTM_STEP))
            print('|--- IMS_STEP:\t{}'.format(Config.IMS_STEP))
        else:
            print('|--- TESTING_TIME:\t{}'.format(Config.TESTING_TIME))
            if 'conv-lstm' in alg_name:
                print('|--- FW_BEST_CHECKPOINT:\t{}'.format(Config.FW_BEST_CHECKPOINT))
                print('|--- BW_BEST_CHECKPOINT:\t{}'.format(Config.BW_BEST_CHECKPOINT))
            else:
                print('|--- LSTM_BEST_CHECKPOINT:\t{}'.format(Config.LSTM_BEST_CHECKPOINT))

        if 'lstm-nn' in alg_name:
            print('|--- LSTM_DEEP:\t{}'.format(Config.LSTM_DEEP))
            print('|--- LSTM_DEEP_NLAYERS:\t{}'.format(Config.LSTM_DEEP_NLAYERS))
            print('|--- LSTM_DROPOUT:\t{}'.format(Config.LSTM_DROPOUT))
            print('|--- LSTM_HIDDEN_UNIT:\t{}'.format(Config.LSTM_HIDDEN_UNIT))
        elif 'conv-lstm' in alg_name:
            print('|--- CONV_LAYERS:\t{}'.format(Config.CNN_LAYERS))
            print('|--- FILTERS:\t{}'.format(Config.FILTERS))
            print('|--- KERNEL_SIZE:\t{}'.format(Config.KERNEL_SIZE))
            print('|--- STRIDES:\t{}'.format(Config.STRIDES))
            print('|--- DROPOUTS:\t{}'.format(Config.DROPOUTS))
            print('|--- RNN_DROPOUTS:\t{}'.format(Config.RNN_DROPOUTS))

    elif 'arima' in alg_name:
        print('|--- ARIMA_UPDATE:\t{}-days'.format(Config.ARIMA_UPDATE))

    elif 'holt-winter' in alg_name:
        print('|--- HOLT_WINTER_UPDATE:\t{}-days'.format(Config.HOLT_WINTER_UPDATE))
        print('|--- HOLT_WINTER_TREND:\t{}'.format(Config.HOLT_WINTER_TREND))
        print('|--- HOLT_WINTER_SEASONAL:\t{}'.format(Config.HOLT_WINTER_SEASONAL))
    else:
        raise ValueError('Unkown alg!')

    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


def train(args):
    alg_name = args.alg

    data = np.load(DATA_PATH + '{}.npy'.format(args.data_name))

    if 'fwbw-conv-lstm' in alg_name or 'convlstm' in alg_name:
        from algs.fwbw_conv_lstm import train_fwbw_conv_lstm
        train_fwbw_conv_lstm(args=args, data=data)
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

    if 'fwbw-conv-lstm' in alg_name or 'convlstm' in alg_name:
        from algs.fwbw_conv_lstm import test_fwbw_conv_lstm
        test_fwbw_conv_lstm(args=args, data=data)
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

    if 'train' in args.run_mode or 'training' in args.run_mode:
        train(args)
    else:
        test(args)

    return


if __name__ == '__main__':
    # create_abilene_data_2d('/home/anle/AbileneTM-all/')
    # create_Geant2d()
    main(sys.argv)
