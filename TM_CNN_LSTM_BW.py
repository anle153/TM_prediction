import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from Utils.DataHelper import *
from Utils.DataPreprocessing import *
from Models.ConvLSTM_model import *

# PATH CONFIGURATION
FIGURE_DIR = './figures/'
MODEL_RECORDED = './Model_Recorded/'

# DATASET CONFIGURATION
DATASET = ['Geant', 'Geant_noise_removed', 'Abilene', 'Abilene_noise_removed']
GEANT = 0
GEANT_NOISE_REMOVED = 1
ABILENE = 2
ABILENE_NOISE_REMOVE = 3

HOME = os.path.expanduser('~')

# Abilene dataset path.
""" The Abilene dataset contains 
    
    X           a 2016x2016 matrix of flow volumes
    A           a 30x121 matrix of routing of the 121 flows over 30 edge between adjacent nodes in Abilene network.
    odnames     a 121x1 character vector of OD pair names
    edgenames   a 30x1 character vector of node pairs sharing an edge 
"""
ABILENE_DATASET_PATH = './Dataset/SAND_TM_Estimation_Data.mat'

# Geant dataset path.
""" The Geant dataset contains 

    X: a (10772 x 529) matrix of flow volumes
"""
DATAPATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices'
GEANT_DATASET_PATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices/Geant_dataset.csv'
GEANT_DATASET_NOISE_REMOVED_PATH = DATAPATH + '/Gean_noise_removed.csv'

N_GPUS = 2

# RNN CONFIGURATION
INPUT_DIM = 100
HIDDEN_DIM = 150
LOOK_BACK = 26
N_EPOCH = 100
BATCH_SIZE = 128

# TRAINING MODES
SPLIT_TRAINING_SET_MODE = 0
HIDDEN_DIM_MODE = 1
BATCH_SIZE_MODE = 2
LOOK_BACK_MODE = 3

TRAINING_MODES = ['Split_dataset', 'hidden_layers', 'batch_size', 'lookback']

# Scaler mode
STANDARDSCALER = 0
MINMAX = 1
NORMALIZER = 2
DATANORMALIZER = 3
SCALER = ['StandardScaler', 'MinMax', 'SKNormalizer', 'MeanStdScale']


def predict_cnn_lstm(test_set, n_timesteps, model, sampling_ratio):
    """

    :param test_set: array-like, shape=(timeslot, od x od)
    :param look_back: int, default 26
    :param model: rnn model for  lstm
    :param sampling_ratio: sampling ratio at each time slot
    :param hyper_parameters:
    [adjust_loss, recovery_loss_weight, dfa_weight,consecutive_loss_weight]

    :return: prediction traffic matrix shape = (timeslot, od)
    """

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = np.copy(test_set[0:n_timesteps, :, :])  # rnn input shape = (timeslot , od, od)
    # Results TM
    # The TF array for random choosing the measured flows

    tf = np.array([True, False])
    # measured_matrix = np.random.choice(tf,
    #                                    size=(rnn_input.shape[0], rnn_input.shape[1], rnn_input.shape[2]),
    #                                    p=(sampling_ratio, 1 - sampling_ratio))
    #
    # labels = measured_matrix.astype(float)
    #
    # rnn_input[labels == 0.0] = np.random.uniform(rnn_input[labels == 0.0] - 1, rnn_input[labels == 0.0] + 1)

    labels = np.ones(shape=(rnn_input.shape[0], rnn_input.shape[1], rnn_input.shape[2]))

    tm_labels = np.concatenate([np.expand_dims(rnn_input, axis=3), np.expand_dims(labels, axis=3)], axis=3)

    day_size = 24 * (60 / 5)

    # Predict the TM from time slot look_back
    for ts in range(0, test_set.shape[0] - n_timesteps, 1):
        date = int(ts / day_size)

        rnn_input = tm_labels[ts:(ts + n_timesteps), :, :, :]

        rnn_input = np.expand_dims(rnn_input, axis=0)

        # Get the TM prediction of next time slot
        predictX = model.predict(rnn_input)
        # Remove first dimension (No. samples )
        predictX = np.squeeze(predictX, axis=0)

        predict_tm = predictX[-1, :, :, :]
        sampling = np.random.choice(tf, size=(12, 12, 1), p=(sampling_ratio, 1 - sampling_ratio))
        inv_sampling = np.invert(sampling)

        pred_tm = predict_tm * inv_sampling

        correct_data = np.copy(test_set[ts + n_timesteps, :, :])

        ground_truth = np.expand_dims(correct_data, axis=2) * sampling

        new_input = pred_tm + ground_truth

        new_input = np.concatenate([new_input, sampling], axis=2)
        new_input = np.expand_dims(new_input, axis=0)
        tm_labels = np.concatenate([tm_labels, new_input], axis=0)

        # Print error ratio for each timestep prediction
        # y_true = np.copy(test_set[ts+n_timesteps, :, :])
        # y_pred = np.copy(tm_labels[ts+n_timesteps, :, :, 0])
        # measured_matrix = np.copy(tm_labels[ts+n_timesteps, :, :, 1])
        # print('|--- Timestep: %i, error ratio %.4f' %(ts+n_timesteps,
        #                                               error_ratio(y_true=y_true,
        #                                                           y_pred=y_pred,
        #                                                           measured_matrix=measured_matrix)))

    return tm_labels


def cnn_lstm_backward(raw_data, dataset_name='Abilene24_3d', n_timesteps=26):
    test_name = 'cnn_lstm_backward'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    sampling_ratio = 0.3

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, n_timesteps)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    print('|--- Splitting train-test set')
    train_set, test_set = prepare_train_test_set_3d(data=raw_data,
                                                    sampling_itvl=5,
                                                    splitting_ratio=splitting_ratio)

    print("|--- Create XY set.")
    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    training_set = (train_set - mean_train) / std_train

    training_set_backward = np.flip(training_set, axis=0)

    if not os.path.isfile(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX_backward.npy'):

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

        print("|--- Load xy backward sets at " +
              HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps)

        train_x_backward = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX_backward.npy')
        train_y_backward = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY_backward.npy')

    print("|--- Creating cnn-lstm-backward model")
    # CNN_ 5 layers configuration for backward network
    # cnn_layers_backward = 5
    # filters_backward=[32, 32, 48, 48, 64]
    # strides_backward=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    # dropouts_backward = [0, 0, 0, 0, 0]
    # rnn_dropouts_backward = [0.2, 0.2, 0.2, 0.2, 0.2]

    # CNN 4 layers configuration for backward network

    # cnn_layers_backward = 4
    # filters_backward=[16, 32, 48, 64]
    # strides_backward=[[1, 1], [1, 1], [1, 1], [1, 1]]
    # kernel_sizes_backward=[[3, 3], [3, 3], [5, 5], [3, 3]]
    # dropouts_backward = [0, 0, 0, 0]
    # rnn_dropouts_backward = [0.2, 0.2, 0.2, 0.2]

    # CNN 3 layers configuration for backward network
    # cnn_layers_backward = 3
    # filters_backward=[32, 48, 64]
    # kernel_sizes_backward=[[3, 3], [3, 3], [3, 3]]
    # strides_backward=[[1, 1], [1, 1], [1, 1]]
    # dropouts_backward = [0.0, 0.0, 0.0]
    # rnn_dropouts_backward = [0.2, 0.2, 0.2]

    # CNN 2 layers configuration for backward network
    cnn_layers_backward = 2
    filters_backward = [16, 16]
    kernel_sizes_backward = [[3, 3], [3, 3]]
    strides_backward = [[1, 1], [1, 1]]
    dropouts_backward = [0.0, 0.0]
    rnn_dropouts_backward = [0.2, 0.2]

    filters_2_str_backward = ''
    for filter_backward in filters_backward:
        filters_2_str_backward = filters_2_str_backward + '_' + str(filter_backward)
    filters_2_str_backward = filters_2_str_backward + '_'

    kernel_2_str_backward = ''
    for kernel_size_backward in kernel_sizes_backward:
        kernel_2_str_backward = kernel_2_str_backward + '_' + str(kernel_size_backward[0])
    kernel_2_str_backward = kernel_2_str_backward + '_'

    dropouts_2_str_backward = ''
    for dropout_backward in dropouts_backward:
        dropouts_2_str_backward = dropouts_2_str_backward + '_' + str(dropout_backward)
    dropouts_2_str_backward = dropouts_2_str_backward + '_'

    rnn_dropouts_2_str_backward = ''
    for rnn_dropout_backward in rnn_dropouts_backward:
        rnn_dropouts_2_str_backward = rnn_dropouts_2_str_backward + '_' + str(rnn_dropout_backward)

    backward_model_name = 'Backward_CNN_layers_%i_filters%skernels%sdropouts%srnn_dropouts%s' % \
                          (cnn_layers_backward, filters_2_str_backward, kernel_2_str_backward, dropouts_2_str_backward,
                           rnn_dropouts_2_str_backward)

    # CNN_BRNN backward model
    cnn_lstm_model_backward = ConvLSTM(n_timsteps=n_timesteps, height=12, weight=12, depth=2,
                                       cnn_layers=cnn_layers_backward, a_filters=filters_backward,
                                       a_strides=strides_backward, dropouts=dropouts_backward,
                                       kernel_sizes=kernel_sizes_backward,
                                       rnn_dropouts=rnn_dropouts_backward,
                                       check_point=True,
                                       saving_path=model_recorded_path + 'Sampling_%.2f_timesteps_%i/%s/'
                                                   % (sampling_ratio, n_timesteps, backward_model_name))

    result_path = HOME + '/TM_estimation_results/%s/%s/%s/' % (dataset_name, test_name, backward_model_name)

    if os.path.isfile(path=cnn_lstm_model_backward.saving_path + 'weights-%i-0.00.hdf5' % N_EPOCH):
        print('|--- Model exist!')
        cnn_lstm_model_backward.load_model_from_check_point(_from_epoch=N_EPOCH, weights_file_type='hdf5')
    else:
        print('[%s]---Compile model. Saving path %s --- ' % (test_name, cnn_lstm_model_backward.saving_path))

        from_epoch = cnn_lstm_model_backward.load_model_from_check_point(weights_file_type='hdf5')
        if from_epoch > 0:
            cnn_lstm_model_backward.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])
            print('[%s]--- Continue training model from epoch %i --- ' % (test_name, from_epoch))
            training_history = cnn_lstm_model_backward.model.fit(train_x_backward,
                                                                 train_y_backward,
                                                                 batch_size=BATCH_SIZE,
                                                                 epochs=N_EPOCH,
                                                                 initial_epoch=from_epoch,
                                                                 validation_split=0.25,
                                                                 callbacks=cnn_lstm_model_backward.callbacks_list)
            cnn_lstm_model_backward.plot_model_metrics(training_history,
                                                       plot_prefix_name='Metrics')

        else:

            cnn_lstm_model_backward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae',
                                                                                                        'accuracy'])

            training_history = cnn_lstm_model_backward.model.fit(train_x_backward,
                                                                 train_y_backward,
                                                                 batch_size=BATCH_SIZE,
                                                                 epochs=N_EPOCH,
                                                                 validation_split=0.25,
                                                                 callbacks=cnn_lstm_model_backward.callbacks_list)
            cnn_lstm_model_backward.plot_model_metrics(training_history,
                                                       plot_prefix_name='Metrics')

    print(cnn_lstm_model_backward.model.summary())

    return


def cnn_lstm_backward_test_loop(test_set, testing_set, cnn_lstm_model,
                                epoch, n_timesteps, sampling_ratio,
                                std_train, mean_train, n_running_time=10):
    err_ratio_temp = []

    for running_time in range(n_running_time):
        cnn_lstm_model.load_model_from_check_point(_from_epoch=epoch, weights_file_type='hdf5')
        cnn_lstm_model.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

        _testing_set = np.copy(testing_set)
        _test_set = np.copy(test_set)
        print('|--- Epoch %i - Run time: %i' % (epoch, running_time))

        tm_labels = predict_cnn_lstm(test_set=_testing_set,
                                     n_timesteps=n_timesteps,
                                     model=cnn_lstm_model.model,
                                     sampling_ratio=sampling_ratio)

        pred_tm = np.copy(tm_labels[:, :, :, 0])
        measured_matrix = np.copy(tm_labels[:, :, :, 1])

        pred_tm = pred_tm * std_train + mean_train

        err_ratio_temp.append(error_ratio(y_true=_test_set, y_pred=pred_tm, measured_matrix=measured_matrix))
        print('|--- error: %.3f' % error_ratio(y_true=_test_set, y_pred=pred_tm, measured_matrix=measured_matrix))

    return err_ratio_temp


def cnn_lstm_backward_test(raw_data, dataset_name, n_timesteps, with_epoch=0, from_epoch=0, to_epoch=0):
    test_name = 'cnn_lstm_backward'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    errors = np.empty((0, 3))
    sampling_ratio = 0.3

    train_set, test_set = prepare_train_test_set_3d(data=raw_data,
                                                    sampling_itvl=5,
                                                    splitting_ratio=splitting_ratio)

    test_set = test_set[0:-864, :, :]
    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    test_set = np.flip(test_set, axis=0)
    copy_test_set = np.copy(test_set)

    testing_set = np.copy(test_set)

    testing_set = (testing_set - mean_train) / std_train

    copy_testing_set = np.copy(testing_set)

    print("|--- Creating cnn-lstm-backward model")
    # CNN_ 5 layers configuration for backward network
    # cnn_layers_backward = 5
    # filters_backward=[32, 32, 48, 48, 64]
    # strides_backward=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    # dropouts_backward = [0, 0, 0, 0, 0]
    # rnn_dropouts_backward = [0.2, 0.2, 0.2, 0.2, 0.2]

    # CNN 4 layers configuration for backward network

    # cnn_layers_backward = 4
    # filters_backward=[16, 32, 48, 64]
    # strides_backward=[[1, 1], [1, 1], [1, 1], [1, 1]]
    # kernel_sizes_backward=[[3, 3], [3, 3], [5, 5], [3, 3]]
    # dropouts_backward = [0, 0, 0, 0]
    # rnn_dropouts_backward = [0.2, 0.2, 0.2, 0.2]

    # CNN 3 layers configuration for backward network
    # cnn_layers_backward = 3
    # filters_backward=[32, 48, 64]
    # kernel_sizes_backward=[[3, 3], [3, 3], [3, 3]]
    # strides_backward=[[1, 1], [1, 1], [1, 1]]
    # dropouts_backward = [0.0, 0.0, 0.0]
    # rnn_dropouts_backward = [0.2, 0.2, 0.2]

    # CNN 2 layers configuration for backward network
    cnn_layers_backward = 2
    filters_backward = [8, 8]
    kernel_sizes_backward = [[3, 3], [3, 3]]
    strides_backward = [[1, 1], [1, 1]]
    dropouts_backward = [0.0, 0.0]
    rnn_dropouts_backward = [0.2, 0.2]

    filters_2_str_backward = ''
    for filter_backward in filters_backward:
        filters_2_str_backward = filters_2_str_backward + '_' + str(filter_backward)
    filters_2_str_backward = filters_2_str_backward + '_'

    kernel_2_str_backward = ''
    for kernel_size_backward in kernel_sizes_backward:
        kernel_2_str_backward = kernel_2_str_backward + '_' + str(kernel_size_backward[0])
    kernel_2_str_backward = kernel_2_str_backward + '_'

    dropouts_2_str_backward = ''
    for dropout_backward in dropouts_backward:
        dropouts_2_str_backward = dropouts_2_str_backward + '_' + str(dropout_backward)
    dropouts_2_str_backward = dropouts_2_str_backward + '_'

    rnn_dropouts_2_str_backward = ''
    for rnn_dropout_backward in rnn_dropouts_backward:
        rnn_dropouts_2_str_backward = rnn_dropouts_2_str_backward + '_' + str(rnn_dropout_backward)

    backward_model_name = 'Backward_CNN_layers_%i_filters%skernels%sdropouts%srnn_dropouts%s' % \
                          (cnn_layers_backward, filters_2_str_backward, kernel_2_str_backward, dropouts_2_str_backward,
                           rnn_dropouts_2_str_backward)

    # CNN_BRNN backward model
    cnn_lstm_model_backward = ConvLSTM(n_timsteps=n_timesteps, height=12, weight=12, depth=2,
                                       cnn_layers=cnn_layers_backward, a_filters=filters_backward,
                                       a_strides=strides_backward, dropouts=dropouts_backward,
                                       kernel_sizes=kernel_sizes_backward,
                                       rnn_dropouts=rnn_dropouts_backward,
                                       check_point=True,
                                       saving_path=model_recorded_path + 'Sampling_%.2f_timesteps_%i/%s/'
                                                   % (sampling_ratio, n_timesteps, backward_model_name))

    sampling_timesteps = 'Sampling_%.2f_timesteps_%d' % (sampling_ratio, n_timesteps)

    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/' % (
    dataset_name, test_name, sampling_timesteps, backward_model_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if with_epoch != 0:
        n_running_time = 20
        err_ratio_temp = cnn_lstm_backward_test_loop(test_set=copy_test_set,
                                                     testing_set=copy_testing_set,
                                                     cnn_lstm_model=cnn_lstm_model_backward,
                                                     epoch=with_epoch,
                                                     n_timesteps=n_timesteps,
                                                     sampling_ratio=sampling_ratio,
                                                     std_train=std_train,
                                                     mean_train=mean_train,
                                                     n_running_time=n_running_time)

        err_ratio_temp = np.array(err_ratio_temp)
        err_ratio_temp = np.reshape(err_ratio_temp, newshape=(n_running_time, 1))
        err_ratio = np.mean(err_ratio_temp)
        err_ratio_std = np.std(err_ratio_temp)

        print('Error_mean: %.5f - Error_std: %.5f' % (err_ratio, err_ratio_std))
        print('|-------------------------------------------------------')

        results = np.empty(shape=(n_running_time, 0))
        epochs = np.arange(0, n_running_time)
        epochs = np.reshape(epochs, newshape=(n_running_time, 1))
        results = np.concatenate([results, epochs], axis=1)
        results = np.concatenate([results, err_ratio_temp], axis=1)

        # Save results:
        print('|--- Results have been saved at %s' % (result_path + 'Epoch_%i_n_running_time_%i.csv' %
                                                      (with_epoch, n_running_time)))

        np.savetxt(fname=result_path + 'Epoch_%i_n_running_time_%i.csv' % (with_epoch, n_running_time),
                   X=results,
                   delimiter=',')

    else:

        list_weights_files_rnn = fnmatch.filter(os.listdir(cnn_lstm_model_backward.saving_path), '*.hdf5')

        if len(list_weights_files_rnn) == 0:
            print('----> [RNN-load_model_from_check_point] --- Found no weights file at %s---' %
                  cnn_lstm_model_backward.saving_path)
            return -1

        list_weights_files_rnn = sorted(list_weights_files_rnn, key=lambda x: int(x.split('-')[1]))

        _max_epoch_rnn = int(list_weights_files_rnn[-1].split('-')[1])

        for epoch in range(1, _max_epoch_rnn + 1, 1):
            err_ratio_temp = cnn_lstm_backward_test_loop(test_set=copy_test_set,
                                                         testing_set=copy_testing_set,
                                                         cnn_lstm_model=cnn_lstm_model_backward,
                                                         epoch=epoch,
                                                         n_timesteps=n_timesteps,
                                                         sampling_ratio=sampling_ratio,
                                                         std_train=std_train,
                                                         mean_train=mean_train,
                                                         n_running_time=10)

            err_ratio_temp = np.array(err_ratio_temp)
            err_ratio = np.mean(err_ratio_temp)
            err_ratio_std = np.std(err_ratio_temp)

            error = np.expand_dims(np.array([epoch, err_ratio, err_ratio_std]), axis=0)
            errors = np.concatenate([errors, error], axis=0)
            print('|--- Errors by epoches ---')
            print(errors)
            print('|-------------------------------------------------------')

        np.savetxt(cnn_lstm_model_backward.saving_path + 'Errors_by_epoch.csv', errors, delimiter=',')

    return


if __name__ == "__main__":
    np.random.seed(10)

    if not os.path.isfile(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/Abilene24_3d/'):
            os.makedirs(HOME + '/TM_estimation_dataset/Abilene24_3d/')
        load_abilene_3d()

    Abilene24_3d = np.load(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d.npy')

    # Abilene24s_3d = Abilene24_3d[0:8064, :, :]

    # for ntimesteps in ntimesteps_range:
    #     cnn_lstm(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=ntimesteps)

    # Abilene24_3d = shuffling_data_3d_by_day(data=Abilene24_3d, sampling_itvl=5)

    with tf.device('/device:GPU:0'):
        cnn_lstm_backward_test(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=26, with_epoch=150)
        # cnn_lstm_backward(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=20)
