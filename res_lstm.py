import numpy as np

from algs.res_fwbw_lstm import train_res_fwbw_lstm
from common import Config

if __name__ == '__main':
    data = np.load(Config.DATA_PATH + '{}.npy'.format(Config.DATA_NAME))
    train_res_fwbw_lstm(data)
