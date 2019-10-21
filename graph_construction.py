import os

import numpy as np
from tslearn.neighbors import NearestNeighbors

from lib.utils import prepare_train_valid_test_2d

day_size = 288

Dataset_dir = './Dataset/'
data = np.load('./Dataset/Abilene2d.npy')

print('|--- Splitting train-test set.')
train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
print('|--- Normalizing the train set.')

# reduced_data = time_series_representation(train_data2d, seq_len=36)
#
# cdwt = cdist_dtw(reduced_data.transpose())
#
# print(cdwt.shape)
# print(cdwt)


knn = NearestNeighbors(radius=1.6, n_jobs=8, metric='correlation').fit(np.transpose(train_data2d))
knn_graph = knn.kneighbors_graph(np.transpose(train_data2d))
knn_graph = knn_graph.toarray()
np.save(os.path.join(Dataset_dir, 'knn_radius_1.6_correlation'))
