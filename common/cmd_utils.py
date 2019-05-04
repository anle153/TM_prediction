from common import Config

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--run_mode', help='training/testing mode. Default: training', type=str, default='training')
    parser.add_argument('--data_name', help='Dataset name. Default: training', type=str, default='Abilene')
    parser.add_argument('--seed', help='Seed. Default: None', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm. Default: None', type=str, default=None)
    parser.add_argument('--tag', help='Algorithm_tag. Default: None', type=str, default=None)
    parser.add_argument('--gpu', help='Specify GPU card. Default: None', type=int, default=0)
    parser.add_argument('--visualize', help='Visualize results. Default: False', default=False)
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dicitonary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval


def print_fwbw_conv_lstm_info(run_mode):
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

    if 'train' in run_mode:
        print('|--- N_EPOCH:\t{}'.format(Config.FWBW_CONV_LSTM_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.FWBW_CONV_LSTM_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.FWBW_CONV_LSTM_STEP))
        if Config.FWBW_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.FWBW_CONV_LSTM_IMS_STEP))
    else:
        print('|--- TESTING_TIME:\t{}'.format(Config.FWBW_CONV_LSTM_TESTING_TIME))
        print('|--- FW_BEST_CHECKPOINT:\t{}'.format(Config.FW_BEST_CHECKPOINT))
        print('|--- BW_BEST_CHECKPOINT:\t{}'.format(Config.BW_BEST_CHECKPOINT))
        if not Config.FWBW_CONV_LSTM_RANDOM_ACTION:
            print('|--- HYPERPARAMS:\t{}'.format(Config.FWBW_CONV_LSTM_HYPERPARAMS))


def print_conv_lstm_info(run_mode):
    print('|--- MON_RATIO:\t{}'.format(Config.CONV_LSTM_MON_RAIO))
    print('            -----------            ')

    print('|--- CONV_LAYERS:\t{}'.format(Config.CONV_LSTM_LAYERS))
    print('|--- FILTERS:\t{}'.format(Config.CONV_LSTM_FILTERS))
    print('|--- KERNEL_SIZE:\t{}'.format(Config.CONV_LSTM_KERNEL_SIZE))
    print('|--- STRIDES:\t{}'.format(Config.CONV_LSTM_STRIDES))
    print('|--- DROPOUTS:\t{}'.format(Config.CONV_LSTM_DROPOUTS))
    print('|--- RNN_DROPOUTS:\t{}'.format(Config.CONV_LSTM_RNN_DROPOUTS))

    if 'train' in run_mode:
        print('|--- N_EPOCH:\t{}'.format(Config.CONV_LSTM_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.CONV_LSTM_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.CONV_LSTM_STEP))
        if Config.CONV_LSTM_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.CONV_LSTM_IMS_STEP))
    else:
        print('|--- TESTING_TIME:\t{}'.format(Config.CONV_LSTM_TESTING_TIME))
        print('|--- BEST_CHECKPOINT:\t{}'.format(Config.CONV_LSTM_BEST_CHECKPOINT))


def print_lstm_info(run_mode):
    print('|--- MON_RATIO:\t{}'.format(Config.LSTM_MON_RAIO))
    print('            -----------            ')

    print('|--- LSTM_DEEP:\t{}'.format(Config.LSTM_DEEP))
    if Config.LSTM_DEEP:
        print('|--- LSTM_DEEP_NLAYERS:\t{}'.format(Config.LSTM_DEEP_NLAYERS))
    print('|--- LSTM_DROPOUT:\t{}'.format(Config.LSTM_DROPOUT))
    print('|--- LSTM_HIDDEN_UNIT:\t{}'.format(Config.LSTM_HIDDEN_UNIT))

    if 'train' in run_mode:
        print('|--- N_EPOCH:\t{}'.format(Config.LSTM_N_EPOCH))
        print('|--- BATCH_SIZE:\t{}'.format(Config.LSTM_BATCH_SIZE))
        print('|--- LSTM_STEP:\t{}'.format(Config.LSTM_STEP))
        if Config.LSTM_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.LSTM_IMS_STEP))
    else:
        print('|--- TESTING_TIME:\t{}'.format(Config.LSTM_TESTING_TIME))
        print('|--- BEST_CHECKPOINT:\t{}'.format(Config.LSTM_BEST_CHECKPOINT))


def print_arima_info(run_mode):
    print('|--- MON_RATIO:\t{}'.format(Config.ARIMA_MON_RATIO))
    print('            -----------            ')

    if 'train' in run_mode:
        if Config.ARIMA_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.ARIMA_IMS_STEP))
    else:
        print('|--- TESTING_TIME:\t{}'.format(Config.ARIMA_TESTING_TIME))
        print('|--- ARIMA_UPDATE:\t{}'.format(Config.ARIMA_UPDATE))


def print_holt_winter_info(run_mode):
    print('|--- MON_RATIO:\t{}'.format(Config.HOLT_WINTER_MON_RATIO))
    print('            -----------            ')

    print('|--- HOLT_WINTER_SEASONAL:\t{}'.format(Config.HOLT_WINTER_SEASONAL))
    print('|--- HOLT_WINTER_TREND:\t{}'.format(Config.HOLT_WINTER_TREND))

    if 'train' in run_mode:
        if Config.HOLT_WINTER_IMS:
            print('|--- IMS_STEP:\t{}'.format(Config.HOLT_WINTER_IMS_STEP))
    else:
        print('|--- TESTING_TIME:\t{}'.format(Config.HOLT_WINTER_TESTING_TIME))
        print('|--- HOLT_WINTER_UPDATE:\t{}'.format(Config.HOLT_WINTER_UPDATE))


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
    print('|--- MODE:\t{}'.format(args.run_mode))
    print('|--- ALG:\t{}'.format(alg_name))
    print('|--- TAG:\t{}'.format(tag))
    print('|--- DATA:\t{}'.format(data_name))
    print('|--- GPU:\t{}'.format(gpu))
    print('|--- MIN_MAX_SCALER:\t{}'.format(Config.MIN_MAX_SCALER))

    if 'fwbw-conv-lstm' in alg_name or 'fwbw-convlstm' in alg_name:
        print_fwbw_conv_lstm_info(args.run_mode)
    elif 'conv-lstm' in alg_name or 'convlstm' in alg_name:
        print_conv_lstm_info(args.run_mode)
    elif 'lstm-nn' in alg_name:
        print_lstm_info(args.run_mode)
    elif 'arima' in alg_name:
        print_arima_info(args.run_mode)
    elif 'holt-winter' in alg_name:
        print_holt_winter_info(args.run_mode)
    else:
        raise ValueError('Unkown alg!')
    print('|--- RESULT_NAME:\t{}'.format(Config.ADDED_RESULT_NAME))
    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')
