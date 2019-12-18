import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

data = np.load('Dataset/Abilene/Abilene.npy')

for i in tqdm(range(data.shape[1])):
    plt.plot(data[:288 * 2, i])
    plt.savefig('Dataset/Abilene/raw_plot/flow_{}.png'.format(i), )
    plt.close()
