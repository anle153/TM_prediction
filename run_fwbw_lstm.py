import numpy as np

from algs.fwbw_lstm import train_fwbw_lstm
from common import Config

if __name__ == '__main__':
    data = np.load(Config.DATA_PATH + '{}.npy'.format(Config.DATA_NAME))
    train_fwbw_lstm(data)
