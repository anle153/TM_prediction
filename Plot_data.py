import numpy as np

from common import Config

Abilene_data2d = np.load(Config.DATA_PATH + '{}.npy'.format(Config.DATA_NAME))

np.savetxt(Config.DATA_PATH + '{}-1000.csv'.format(Config.DATA_NAME), Abilene_data2d[:1000])
