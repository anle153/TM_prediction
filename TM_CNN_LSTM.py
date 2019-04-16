import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from common.DataHelper import *
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
N_EPOCH = 50
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


def lstm_iterated_multi_step_tm_prediction(tm_labels, rnn_model,
                                           n_timesteps,
                                           prediction_steps,
                                           iterated_multi_steps_tm):
    multi_steps_tm = np.copy(tm_labels[-n_timesteps:, :, :, :])

    for ts_ahead in range(prediction_steps):
        rnn_input = multi_steps_tm[-n_timesteps:, :, :, :]

        rnn_input = np.expand_dims(rnn_input, axis=0)

        predictX = rnn_model.predict(rnn_input)

        predictX = np.squeeze(predictX, axis=0)

        pred = predictX[-1, :, :, :]

        sampling = np.zeros(shape=(12, 12, 1))

        new_input = np.concatenate([pred, sampling], axis=2)
        new_input = np.expand_dims(new_input, axis=0)

        multi_steps_tm = np.concatenate([multi_steps_tm, new_input], axis=0)

    multi_steps_tm = multi_steps_tm[n_timesteps:, :, :, 0]
    multi_steps_tm = np.expand_dims(multi_steps_tm, axis=0)

    iterated_multi_steps_tm = np.concatenate([iterated_multi_steps_tm, multi_steps_tm], axis=0)

    return iterated_multi_steps_tm


def predict_cnn_lstm(test_set, n_timesteps, model, sampling_ratio, ism_prediction_steps):
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

    iterated_multi_steps_tm = np.empty(shape=(0, ism_prediction_steps, 12, 12))

    # Predict the TM from time slot look_back
    for ts in range(0, test_set.shape[0] - n_timesteps, 1):
        date = int(ts / day_size)

        if ts < test_set.shape[0] - n_timesteps - ism_prediction_steps:
            iterated_multi_steps_tm = lstm_iterated_multi_step_tm_prediction(tm_labels=tm_labels,
                                                                             rnn_model=model,
                                                                             prediction_steps=ism_prediction_steps,
                                                                             n_timesteps=n_timesteps,
                                                                             iterated_multi_steps_tm=iterated_multi_steps_tm)

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

    return tm_labels, iterated_multi_steps_tm


def predict_cnn_lstm_one_step(test_set, n_timesteps, model, sampling_ratio):
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

    return tm_labels


def cnn_lstm(raw_data, dataset_name='Abilene24_3d', n_timesteps=26, sampling_ratio=0.10):
    test_name = 'cnn_lstm'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    print('|--- Splitting train-test set')
    train_set, test_set = prepare_train_test_set_3d(data=raw_data,
                                                    sampling_itvl=5,
                                                    splitting_ratio=splitting_ratio)

    print("|--- Create XY set.")
    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    training_set = (train_set - mean_train) / std_train

    if not os.path.isfile(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                    sampling_ratio, n_timesteps) + dataset_name + '_trainX.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps)):
            os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps))

        train_x, train_y = create_xy_set_3d_by_random(raw_data=training_set,
                                                      n_timesteps=n_timesteps,
                                                      sampling_ratio=sampling_ratio,
                                                      random_eps=1)

        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps) + dataset_name + '_trainX.npy',
            train_x)
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps) + dataset_name + '_trainY.npy',
            train_y)
    else:

        print(
            "|---  Load xy set from " + HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps))

        train_x = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps) + dataset_name + '_trainX.npy')
        train_y = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps) + dataset_name + '_trainY.npy')

    print("|--- Creating cnn-lstm model")

    # CNN 2 layers configuration
    cnn_layers = 2
    filters = [8, 8]
    kernel_sizes = [[3, 3], [3, 3]]
    strides = [[1, 1], [1, 1]]
    dropouts = [0.0, 0.0]
    rnn_dropouts = [0.2, 0.2]

    filters_2_str = ''
    for filter in filters:
        filters_2_str = filters_2_str + '_' + str(filter)
    filters_2_str = filters_2_str + '_'

    kernel_2_str = ''
    for kernel_size in kernel_sizes:
        kernel_2_str = kernel_2_str + '_' + str(kernel_size[0])
    kernel_2_str = kernel_2_str + '_'

    dropouts_2_str = ''
    for dropout in dropouts:
        dropouts_2_str = dropouts_2_str + '_' + str(dropout)
    dropouts_2_str = dropouts_2_str + '_'

    rnn_dropouts_2_str = ''
    for rnn_dropout in rnn_dropouts:
        rnn_dropouts_2_str = rnn_dropouts_2_str + '_' + str(rnn_dropout)

    model_name = 'CNN_layers_%i_filters%skernels%sdropouts%srnn_dropouts%s' % \
                 (cnn_layers, filters_2_str, kernel_2_str, dropouts_2_str, rnn_dropouts_2_str)
    cnn_lstm_model = ConvLSTM(n_timsteps=n_timesteps, height=12, weight=12, depth=2,
                              cnn_layers=cnn_layers, a_filters=filters, a_strides=strides, dropouts=dropouts,
                              kernel_sizes=kernel_sizes,
                              rnn_dropouts=rnn_dropouts,
                              check_point=True,
                              saving_path=model_recorded_path + 'Sampling_%.2f_timesteps_%i/%s/' %
                                          (sampling_ratio, n_timesteps, model_name))

    if os.path.isfile(path=cnn_lstm_model.saving_path + 'model.json'):
        cnn_lstm_model.load_model_from_disk(model_json_file='model.json',
                                            model_weight_file='model.h5')
    else:
        print('[%s]---Compile model. Saving path %s --- ' % (test_name, cnn_lstm_model.saving_path))

        from_epoch = cnn_lstm_model.load_model_from_check_point(weights_file_type='hdf5')
        if from_epoch > 0:
            cnn_lstm_model.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])
            print('[%s]--- Continue training model from epoch %i --- ' % (test_name, from_epoch))
            training_history = cnn_lstm_model.model.fit(train_x,
                                                        train_y,
                                                        batch_size=BATCH_SIZE,
                                                        epochs=N_EPOCH,
                                                        initial_epoch=from_epoch,
                                                        validation_split=0.25,
                                                        callbacks=cnn_lstm_model.callbacks_list)
            cnn_lstm_model.plot_model_metrics(training_history,
                                              plot_prefix_name='Metrics')

        else:

            cnn_lstm_model.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae',
                                                                                               'accuracy'])

            training_history = cnn_lstm_model.model.fit(train_x,
                                                        train_y,
                                                        batch_size=BATCH_SIZE,
                                                        epochs=N_EPOCH,
                                                        validation_split=0.25,
                                                        callbacks=cnn_lstm_model.callbacks_list)
            cnn_lstm_model.plot_model_metrics(training_history,
                                              plot_prefix_name='Metrics')

    print(cnn_lstm_model.model.summary())

    return


def cnn_lstm_test_loop(test_set, testing_set, cnn_lstm_model,
                       epoch, n_timesteps, sampling_ratio,
                       std_train, mean_train, n_running_time,
                       results_path):
    err_ratio_temp = []
    err_ratio_ims_temp = []

    for running_time in range(5, 5 + n_running_time, 1):
        cnn_lstm_model.load_model_from_check_point(_from_epoch=epoch, weights_file_type='hdf5')
        cnn_lstm_model.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

        _testing_set = np.copy(testing_set)
        _test_set = np.copy(test_set)
        print('|--- Epoch %i - Run time: %i' % (epoch, running_time))

        ism_prediction_steps = 12

        tm_labels, iterated_multi_steps_tm = predict_cnn_lstm(test_set=_testing_set,
                                                              n_timesteps=n_timesteps,
                                                              model=cnn_lstm_model.model,
                                                              sampling_ratio=sampling_ratio,
                                                              ism_prediction_steps=ism_prediction_steps)

        pred_tm = np.copy(tm_labels[:, :, :, 0])
        measured_matrix = np.copy(tm_labels[:, :, :, 1])

        pred_tm = pred_tm * std_train + mean_train

        err_ratio_temp.append(error_ratio(y_true=_test_set, y_pred=pred_tm, measured_matrix=measured_matrix))
        print('|--- error: %.3f' % error_ratio(y_true=_test_set, y_pred=pred_tm, measured_matrix=measured_matrix))

        iterated_multi_steps_tm = iterated_multi_steps_tm * std_train + mean_train

        err_ratio_ims = calculate_lstm_iterated_multi_step_tm_prediction_errors(
            iterated_multi_step_pred_tm=iterated_multi_steps_tm,
            test_set=_test_set,
            n_timesteps=n_timesteps,
            prediction_steps=ism_prediction_steps)
        err_ratio_ims_temp.append(err_ratio_ims)
        print('|--- error_IMS: %.3f' % err_ratio_ims)

        np.save(file=results_path + 'Predicted_tm_running_time_%d' % running_time,
                arr=pred_tm)
        np.save(file=results_path + 'Predicted_measured_matrix_running_time_%d' % running_time,
                arr=measured_matrix)
        np.save(file=results_path + 'Predicted_multistep_tm_running_time_%d' % running_time,
                arr=iterated_multi_steps_tm)

    return err_ratio_temp, err_ratio_ims_temp


def cnn_lstm_test_loop_one_step(test_set, testing_set, cnn_lstm_model,
                                epoch, n_timesteps, sampling_ratio,
                                std_train, mean_train, n_running_time,
                                results_path):
    err_ratio_temp = []
    err_ratio_ims_temp = []

    for running_time in range(5, 5 + n_running_time, 1):
        cnn_lstm_model.load_model_from_check_point(_from_epoch=epoch, weights_file_type='hdf5')
        cnn_lstm_model.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

        _testing_set = np.copy(testing_set)
        _test_set = np.copy(test_set)
        print('|--- Epoch %i - Run time: %i' % (epoch, running_time))

        ism_prediction_steps = 12

        tm_labels = predict_cnn_lstm_one_step(test_set=_testing_set,
                                              n_timesteps=n_timesteps,
                                              model=cnn_lstm_model.model,
                                              sampling_ratio=sampling_ratio)

        pred_tm = np.copy(tm_labels[:, :, :, 0])
        measured_matrix = np.copy(tm_labels[:, :, :, 1])

        pred_tm = pred_tm * std_train + mean_train

        er = error_ratio(y_true=_test_set, y_pred=pred_tm, measured_matrix=measured_matrix)

        r2_score = calculate_r2_score(y_true=np.copy(test_set), y_pred=np.copy(pred_tm))
        rmse = rmse_tm_prediction(y_true=np.copy(test_set), y_pred=np.copy(pred_tm))

        err_ratio_temp.append(error_ratio(y_true=_test_set, y_pred=pred_tm, measured_matrix=measured_matrix))

        print('|--- er: %.3f --- rmse: %.3f --- r2: %.3f' % (er, rmse, r2_score))

        # np.save(file=results_path + 'Predicted_tm_running_time_%d' % running_time,
        #         arr=pred_tm)
        # np.save(file=results_path + 'Predicted_measured_matrix_running_time_%d' % running_time,
        #         arr=measured_matrix)
        #
    return err_ratio_temp


def calculate_lstm_iterated_multi_step_tm_prediction_errors(iterated_multi_step_pred_tm, test_set, n_timesteps,
                                                            prediction_steps):
    iterated_multi_step_test_set = np.empty(shape=(0, prediction_steps, test_set.shape[1], test_set.shape[2]))

    for ts in range(test_set.shape[0] - n_timesteps - prediction_steps):
        multi_step_test_set = np.copy(test_set[(ts + n_timesteps): (ts + n_timesteps + prediction_steps), :, :])
        multi_step_test_set = np.expand_dims(multi_step_test_set, axis=0)
        iterated_multi_step_test_set = np.concatenate([iterated_multi_step_test_set, multi_step_test_set], axis=0)

    measured_matrix = np.zeros(shape=iterated_multi_step_test_set.shape)

    err_ratio = error_ratio(y_pred=iterated_multi_step_pred_tm,
                            y_true=iterated_multi_step_test_set,
                            measured_matrix=measured_matrix)

    return err_ratio


def cnn_lstm_test(raw_data, dataset_name, n_timesteps, with_epoch=0, from_epoch=0, to_epoch=0, sampling_ratio=0.10):
    test_name = 'cnn_lstm'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    errors = np.empty((0, 3))

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/CNN_layers_%i_timesteps_%i/' % (3, n_timesteps)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    train_set, test_set = prepare_train_test_set_3d(data=raw_data,
                                                    sampling_itvl=5,
                                                    splitting_ratio=splitting_ratio)

    test_set = test_set[0:-864, :, :]
    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    copy_test_set = np.copy(test_set)

    testing_set = np.copy(test_set)

    testing_set = (testing_set - mean_train) / std_train

    copy_testing_set = np.copy(testing_set)

    print("|--- Creating cnn-lstm model")

    cnn_layers = 2
    filters = [8, 8]
    kernel_sizes = [[3, 3], [3, 3]]
    strides = [[1, 1], [1, 1]]
    dropouts = [0.0, 0.0]
    rnn_dropouts = [0.2, 0.2]

    filters_2_str = ''
    for filter in filters:
        filters_2_str = filters_2_str + '_' + str(filter)
    filters_2_str = filters_2_str + '_'

    kernel_2_str = ''
    for kernel_size in kernel_sizes:
        kernel_2_str = kernel_2_str + '_' + str(kernel_size[0])
    kernel_2_str = kernel_2_str + '_'

    dropouts_2_str = ''
    for dropout in dropouts:
        dropouts_2_str = dropouts_2_str + '_' + str(dropout)
    dropouts_2_str = dropouts_2_str + '_'

    rnn_dropouts_2_str = ''
    for rnn_dropout in rnn_dropouts:
        rnn_dropouts_2_str = rnn_dropouts_2_str + '_' + str(rnn_dropout)

    model_name = 'CNN_layers_%i_filters%skernels%sdropouts%srnn_dropouts%s' % \
                 (cnn_layers, filters_2_str, kernel_2_str, dropouts_2_str, rnn_dropouts_2_str)

    cnn_lstm_model = ConvLSTM(n_timsteps=n_timesteps, height=12, weight=12, depth=2,
                              cnn_layers=cnn_layers, a_filters=filters, a_strides=strides, dropouts=dropouts,
                              kernel_sizes=kernel_sizes,
                              rnn_dropouts=rnn_dropouts,
                              check_point=True,
                              saving_path=model_recorded_path + 'Sampling_%.2f_timesteps_%i/%s/' %
                                          (sampling_ratio, n_timesteps, model_name))

    sampling_timesteps = 'Sampling_%.2f_timesteps_%d' % (sampling_ratio, n_timesteps)

    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/' % (
        dataset_name, test_name, sampling_timesteps, model_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if with_epoch != 0:
        n_running_time = 1
        # err_ratio_temp, err_ratio_ims_temp = cnn_lstm_test_loop(test_set=copy_test_set,
        #                                                         testing_set=copy_testing_set,
        #                                                         cnn_lstm_model=cnn_lstm_model,
        #                                                         epoch=with_epoch,
        #                                                         n_timesteps=n_timesteps,
        #                                                         sampling_ratio=sampling_ratio,
        #                                                         std_train=std_train,
        #                                                         mean_train=mean_train,
        #                                                         n_running_time=n_running_time,
        #                                                         results_path=result_path)
        err_ratio_temp = cnn_lstm_test_loop_one_step(test_set=copy_test_set,
                                                     testing_set=copy_testing_set,
                                                     cnn_lstm_model=cnn_lstm_model,
                                                     epoch=with_epoch,
                                                     n_timesteps=n_timesteps,
                                                     sampling_ratio=sampling_ratio,
                                                     std_train=std_train,
                                                     mean_train=mean_train,
                                                     n_running_time=n_running_time,
                                                     results_path=result_path)

        # err_ratio_temp = np.array(err_ratio_temp)
        # err_ratio_temp = np.reshape(err_ratio_temp, newshape=(n_running_time, 1))
        # err_ratio = np.mean(err_ratio_temp)
        # err_ratio_std = np.std(err_ratio_temp)
        #
        # print('Error_mean: %.5f - Error_std: %.5f' % (err_ratio, err_ratio_std))
        # print('|-------------------------------------------------------')
        #
        # results = np.empty(shape=(n_running_time, 0))
        # epochs = np.arange(0, n_running_time)
        # epochs = np.reshape(epochs, newshape=(n_running_time, 1))
        # results = np.concatenate([results, epochs], axis=1)
        # results = np.concatenate([results, err_ratio_temp], axis=1)
        #
        # # Save results:
        # print('|--- Results have been saved at %s' % (result_path + 'Epoch_%i_n_running_time_%i.csv' %
        #                                               (with_epoch, n_running_time)))
        #
        # np.savetxt(fname=result_path + 'Epoch_%i_n_running_time_%i.csv' % (with_epoch, n_running_time),
        #            X=results,
        #            delimiter=',')
        #
        # err_ratio_ims_temp = np.array(err_ratio_ims_temp)
        # err_ratio_ims_temp = np.reshape(err_ratio_ims_temp, newshape=(n_running_time, 1))
        # err_ratio_ims = np.mean(err_ratio_ims_temp)
        # err_ratio_ims_std = np.std(err_ratio_ims_temp)
        #
        # print('Error_mean: %.5f - Error_std: %.5f' % (err_ratio_ims, err_ratio_ims_std))
        # print('|-------------------------------------------------------')
        #
        # results_ims = np.empty(shape=(n_running_time, 0))
        # results_ims = np.concatenate([results_ims, epochs], axis=1)
        # results_ims = np.concatenate([results_ims, err_ratio_ims_temp], axis=1)
        #
        # # Save results:
        # print('|--- Results have been saved at %s' % (result_path + '|--- [IMS]Epoch_%i_n_running_time_%i.csv' %
        #                                               (with_epoch, n_running_time)))
        #
        # np.savetxt(fname=result_path + '[IMS_5]Epoch_%i_n_running_time_%i.csv' % (with_epoch, n_running_time),
        #            X=results_ims, delimiter=',')

    else:

        list_weights_files_rnn = fnmatch.filter(os.listdir(cnn_lstm_model.saving_path), '*.hdf5')

        if len(list_weights_files_rnn) == 0:
            print('----> [RNN-load_model_from_check_point] --- Found no weights file at %s---' %
                  cnn_lstm_model.saving_path)
            return -1

        list_weights_files_rnn = sorted(list_weights_files_rnn, key=lambda x: int(x.split('-')[1]))

        _max_epoch_rnn = int(list_weights_files_rnn[-1].split('-')[1])

        for epoch in range(1, _max_epoch_rnn + 1, 1):
            err_ratio_temp = cnn_lstm_test_loop(test_set=copy_test_set,
                                                testing_set=copy_testing_set,
                                                cnn_lstm_model=cnn_lstm_model,
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

        np.savetxt(cnn_lstm_model.saving_path + 'Errors_by_epoch.csv', errors, delimiter=',')

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

    for i in [0.1]:
        with tf.device('/device:GPU:0'):
            cnn_lstm_test(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=26, with_epoch=50,
                          sampling_ratio=i)
            # cnn_lstm(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=26, sampling_ratio=i)
