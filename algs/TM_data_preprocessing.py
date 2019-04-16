from multiprocessing import cpu_count

from Utils.DataPreprocessing import *

from common.DataHelper import *


def create_xy_set_2d(raw_data, dataset_name, n_timesteps):
    test_name = 'normal_rnn'

    splitting_ratio = [0.8, 0.2]
    nprocs = cpu_count()
    random_eps = 1
    sampling_ratio = 0.3

    print('|--- Splitting train-test set')
    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)
    print('|--- Data normalization')
    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    training_set = (train_set - mean_train) / std_train

    print("|--- Create XY set.")
    if not os.path.isfile(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps):
            os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps)

        trainX, trainY = parallel_create_xy_set_by_random(training_set, n_timesteps, sampling_ratio, random_eps,
                                                          nprocs)

        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX.npy',
            trainX)
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY.npy',
            trainY)
    else:

        print(
            "|---  XY set have been saved at: " + HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps)

    ####################################################################################################################

    training_set_backward = np.flip(training_set, axis=0)
    if not os.path.isfile(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX_backward.npy'):

        if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/'):
            os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/')

        trainX_backward, trainY_backward = parallel_create_xy_set_by_random(training_set_backward, n_timesteps,
                                                                            sampling_ratio, random_eps,
                                                                            nprocs)

        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX_backward.npy',
            trainX_backward)
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY_backward.npy',
            trainY_backward)
    else:
        print(
            "|---  XY set backward have been saved at: " + HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps)


def create_xy_set_3d(raw_data, dataset_name, n_timesteps):
    test_name = 'cnn_brnn'
    splitting_ratio = [0.8, 0.2]

    sampling_ratio = 0.3

    print('|--- Splitting train-test set.')
    train_set, test_set = prepare_train_test_set_3d(data=raw_data,
                                                    sampling_itvl=5,
                                                    splitting_ratio=splitting_ratio)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_set)
    std_train = np.std(train_set)
    training_set = (train_set - mean_train) / std_train

    if not os.path.isfile(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps):
            os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps)

        print("|--- Create XY sets.")

        train_x, train_y = create_xy_set_3d_by_random(raw_data=training_set,
                                                      n_timesteps=n_timesteps,
                                                      sampling_ratio=sampling_ratio,
                                                      random_eps=1)

        # Save xy sets to file
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX.npy',
            train_x)
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY.npy',
            train_y)

    else:  # Load xy sets from file

        print("|--- Load xy sets.")

    # Create the xy backward set
    training_set_backward = np.flip(training_set, axis=0)

    if not os.path.isfile(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX_backward.npy'):

        # Flip the training set in order to create the xy backward sets

        train_x_backward, train_y_backward = create_xy_set_3d_by_random(raw_data=training_set_backward,
                                                                        n_timesteps=n_timesteps,
                                                                        sampling_ratio=sampling_ratio,
                                                                        random_eps=1)

        # Save xy backward sets to file

        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX_backward.npy',
            train_x_backward)
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY_backward.npy',
            train_y_backward)

    else:  # Load xy backward sets from file

        print("Load xy backward sets")


if __name__ == '__main__':
    Abilene24 = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24.csv')

    if not os.path.isfile(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/Abilene24_3d/'):
            os.makedirs(HOME + '/TM_estimation_dataset/Abilene24_3d/')
        load_abilene_3d()

    Abilene24_3d = np.load(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d.npy')

    for n_timesteps in [20, 30, 40]:
        create_xy_set_2d(raw_data=Abilene24, dataset_name='Abilene24', n_timesteps=n_timesteps)
        create_xy_set_3d(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=n_timesteps)
