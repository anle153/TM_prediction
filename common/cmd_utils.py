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
    parser.add_argument('--save_path', help='Path to save trained model to',
                        default='/home/anle/kaggle/earthquake_prediction/models_save/',
                        type=str)
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
