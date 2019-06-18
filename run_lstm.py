import numpy as np

from algs.lstm_nn import train_lstm_nn
from common import Config

if __name__ == '__main__':
    data = np.load(Config.DATA_PATH + '{}.npy'.format(Config.DATA_NAME))
    train_lstm_nn(data)
