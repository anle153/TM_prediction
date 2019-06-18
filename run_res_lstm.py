import numpy as np

from algs.reslstm_nn import train_reslstm_nn
from common import Config

if __name__ == '__main__':
    data = np.load(Config.DATA_PATH + '{}.npy'.format(Config.DATA_NAME))
    train_reslstm_nn(data)
