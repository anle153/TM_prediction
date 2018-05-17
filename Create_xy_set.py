from Utils.DataPreprocessing import *
import numpy as np
import os

HOME = os.path.expanduser('~')


def load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene.csv'):
    """
    Load Abilene dataset from csv file. If file is not found, create the one from original matlab file and remove noise
    :param csv_file_path:
    :return: A traffic matrix (m x k)
    """
    return np.genfromtxt(csv_file_path, delimiter=',')


def createxy(data, look_back, dataset_name='Abilene24s'):
    splitting_ratio = [0.7, 0.3]
    a_lookback = 26
    errors = np.empty((0, 3))
    n_unique = 10000
    test_name = 'attention'

    train_set, test_set = prepare_train_test_set(data=data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    seperated_train_set, centers_train_set = mean_std_flows_clustering(train_set)

    for cluster in range(len(seperated_train_set)):

        ndays = int(data.shape[0] / 288)
        processing_data = seperated_train_set[cluster + 1]

        for day in range(ndays):

            training_set, train_unique_step = parallel_one_hot_encoder(data=processing_data[day * 288:(day + 1) * 288],
                                                                       n_unique=n_unique,
                                                                       nproc=48)

            trainX_path = HOME + '/TM_estimation_dataset/' + dataset_name + \
                          '/Data_by_day/TrainX_' + test_name + '_Abilene24s_' + str(
                splitting_ratio[0]) + '_look_back_%i_cluster_%i_day_%i.npy' % (look_back, cluster, day)
            trainY_path = HOME + '/TM_estimation_dataset/' + dataset_name + \
                          '/Data_by_day/TrainY_' + test_name + '_Abilene24s_' + str(
                splitting_ratio[0]) + '_look_back_%i_cluster_%i_day_%i.npy' % (look_back, cluster, day)

            if not os.path.isfile(trainX_path):
                if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/Data_by_day/'):
                    os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/Data_by_day/')
                print('--- Create TrainX, TrainY ---')
                trainX, trainY = parallel_create_xy_set_encoded(training_set,
                                                                look_back=look_back,
                                                                nproc=48)

                np.save(trainX_path, trainX)
                np.save(trainY_path, trainY)
            else:
                print('--- Load TrainX, TrainY from file ---')
                trainX = np.load(trainX_path)
                trainY = np.load(trainY_path)

            trainX = None
            trainY = None


if __name__ == '__main__':
    Abilene24s_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24s.csv')

    print(Abilene24s_data.shape)
    createxy(data=Abilene24s_data, look_back=26)
