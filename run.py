import sys

import numpy as np

from algs.fwbw_conv_lstm import train_fwbw_conv_lstm
from common.Config import DATA_PATH
from common.cmd_utils import parse_unknown_args, common_arg_parser


def train(args):
    alg_name = args.alg

    data = np.load(DATA_PATH + '{}.npy'.format(args.data_name))

    if 'fwbw-conv-lstm' in alg_name:
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
    import tensorflow as tf
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    gpu = args.gpu

    if gpu is None:
        gpu = 0

    with tf.device('/device:GPU:{}'.format(gpu)):

        if args.run_mode == 'training':
            train(args)
        else:
            test(args)

    return


if __name__ == '__main__':
    main(sys.argv)
