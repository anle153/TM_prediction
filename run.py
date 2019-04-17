import sys

import numpy as np

from algs.fwbw_conv_lstm import train_fwbw_conv_lstm
from common.Config import DATA_PATH
# from common.DataHelper import create_abilene_data_3d
from common.cmd_utils import parse_unknown_args


def train(args):
    alg_name = args.alg

    data = np.load(DATA_PATH + '{}.npy'.format(args.data_name))

    if 'fwbw-convlstm' in alg_name:
        train_func = train_fwbw_conv_lstm
    else:
        raise ValueError('Unkown alg!')

    train_func(args=args, data=data)


def test(args):
    pass


def parse_cmdline_kwargs(args):
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def main(args):
    # create_abilene_data_3d('/home/anle/AbileneTM-all/')
    data = np.load(DATA_PATH + '{}.npy'.format(args.data_name))

    print(data[0, :, :])
    # arg_parser = common_arg_parser()
    # args, unknown_args = arg_parser.parse_known_args(args)
    # extra_args = parse_cmdline_kwargs(unknown_args)
    #
    # if args.run_mode == 'training':
    #     train(args)
    # else:
    #     test(args)

    return


if __name__ == '__main__':
    main(sys.argv)
