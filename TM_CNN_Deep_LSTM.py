from multiprocessing import cpu_count

from Models.ConvLSTM_model import *
from Utils.DataHelper import *
from Utils.DataPreprocessing import *

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
HIDDEN_DIM = 100
LOOK_BACK = 26
N_EPOCH = 500
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


def predict_cnn_lstm(test_set, n_timesteps, model, sampling_ratio,
                     hyperparams=[2.71, 1, 4.83, 1.09]):
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
    rnn_input = test_set[0:n_timesteps, :, :]  # rnn input shape = (timeslot x OD)
    # Results TM
    ret_tm = rnn_input  # ret_rm shape = (time slot x OD)
    # The TF array for random choosing the measured flows
    measured_matrix = np.ones((n_timesteps, test_set.shape[1], test_set.shape[2]), dtype=bool)
    labels = np.ones((n_timesteps, test_set.shape[1], test_set.shape[2]))
    tf = np.array([True, False])

    tm_labels = np.concatenate([np.expand_dims(rnn_input, axis=3), np.expand_dims(labels, axis=3)], axis=3)

    day_size = 24 * (60 / 5)

    # Predict the TM from time slot look_back
    for ts in range(0, test_set.shape[0] - n_timesteps, 1):
        date = int(ts / day_size)
        # print ('--- Predict at timeslot %i ---' % tslot)

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
        ground_truth = np.expand_dims(test_set[ts + n_timesteps, :, :], axis=2) * sampling

        new_input = pred_tm + ground_truth

        new_input = np.concatenate([new_input, sampling], axis=2)
        new_input = np.expand_dims(new_input, axis=0)
        tm_labels = np.concatenate([tm_labels, new_input], axis=0)

    return tm_labels


def cnn_lstm(raw_data, dataset_name='Abilene24_3d', n_timesteps=26):
    test_name = 'cnn_lstm'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'
    nprocs = cpu_count()

    errors = np.empty((0, 4))
    sampling_ratio = 0.3
    day_size = 24 * (60 / 5)

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, n_timesteps)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    ################################################################################################################
    #                                            Data Normalization                                                #

    if not os.path.isfile(HOME + '/TM_estimation_dataset/' + dataset_name + '/' + dataset_name + '_trainX.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/'):
            os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/')

        print('|--- Splitting train-test set')
        train_set, test_set = prepare_train_test_set_3d(data=raw_data,
                                                        sampling_itvl=5,
                                                        splitting_ratio=splitting_ratio)

        print("|--- Create XY set.")
        mean_train = np.mean(train_set)
        std_train = np.std(train_set)

        training_set = (train_set - mean_train) / std_train

        train_x, train_y = create_xy_set_3d_by_random(raw_data=training_set,
                                                      n_timesteps=n_timesteps,
                                                      sampling_ratio=sampling_ratio,
                                                      random_eps=1)

        np.save(HOME + '/TM_estimation_dataset/' + dataset_name + '/' + dataset_name + '_trainX.npy',
                train_x)
        np.save(HOME + '/TM_estimation_dataset/' + dataset_name + '/' + dataset_name + '_trainY.npy',
                train_y)
    else:

        print("|---  Load xy set from " + HOME + '/TM_estimation_dataset/' + dataset_name + '/' + dataset_name)

        train_x = np.load(HOME + '/TM_estimation_dataset/' + dataset_name + '/' + dataset_name + '_trainX.npy')
        train_y = np.load(HOME + '/TM_estimation_dataset/' + dataset_name + '/' + dataset_name + '_trainY.npy')

    print("|--- Creating cnn-lstm model")

    cnn_lstm_model = ConvLSTM(n_timsteps=n_timesteps,
                              height=12,
                              weight=12,
                              depth=2,
                              check_point=True,
                              saving_path=model_recorded_path + 'cnn_lstm/CNN_layers_%i_timesteps_%i_sampling_ratio_%.2f/' %
                                          (3, n_timesteps, sampling_ratio))

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


def cnn_lstm_test(raw_data, dataset_name, n_timesteps, hyperparams=[]):
    test_name = 'cnn_lstm'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    errors = np.empty((0, 2))
    sampling_ratio = 0.3
    day_size = 24 * (60 / 5)

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/CNN_layers_%i_timesteps_%i/' % (3, n_timesteps)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    train_set, test_set = prepare_train_test_set_3d(data=raw_data,
                                                    sampling_itvl=5,
                                                    splitting_ratio=splitting_ratio)
    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    testing_set = (test_set - mean_train) / std_train

    cnn_lstm_model = ConvLSTM(n_timsteps=n_timesteps,
                              height=12,
                              weight=12,
                              depth=2,
                              saving_path=model_recorded_path +
                                          'cnn_lstm/CNN_layers_%i_timesteps_%i_sampling_ratio_%.2f/' % (
                                              3, n_timesteps, sampling_ratio))

    list_weights_files_rnn = fnmatch.filter(os.listdir(cnn_lstm_model.saving_path), '*.hdf5')

    if len(list_weights_files_rnn) == 0:
        print('----> [RNN-load_model_from_check_point] --- Found no weights file at %s---' %
              cnn_lstm_model.saving_path)
        return -1

    list_weights_files_rnn = sorted(list_weights_files_rnn, key=lambda x: int(x.split('-')[1]))

    _max_epoch_rnn = int(list_weights_files_rnn[-1].split('-')[1])

    for epoch in range(1, _max_epoch_rnn + 1, 1):
        cnn_lstm_model.load_model_from_check_point(_from_epoch=epoch, weights_file_type='hdf5')
        cnn_lstm_model.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

        print(cnn_lstm_model.model.summary())

        tm_labels = predict_cnn_lstm(test_set=testing_set,
                                     n_timesteps=n_timesteps,
                                     model=cnn_lstm_model.model,
                                     sampling_ratio=sampling_ratio,
                                     hyperparams=hyperparams)

        pred_tm = tm_labels[:, :, :, 0]
        measured_matrix = tm_labels[:, :, :, 1]

        pred_tm = pred_tm * std_train + mean_train

        err_ratio = error_ratio(y_true=test_set, y_pred=pred_tm, measured_matrix=measured_matrix)
        error = np.expand_dims(np.array([epoch, err_ratio]), axis=0)
        errors = np.concatenate([errors, error], axis=0)
        print('|--- Errors by epoches ---')
        print(errors)

    np.savetxt('./Errors/[%s][CNN_layers_%i_timesteps_%i_sampling_ratio_%.2f]Errors_by_epoch.csv' % (
        test_name, 3, n_timesteps, sampling_ratio), errors, delimiter=',')

    return


if __name__ == "__main__":
    np.random.seed(10)

    if not os.path.isfile(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/Abilene24_3d/'):
            os.makedirs(HOME + '/TM_estimation_dataset/Abilene24_3d/')
        load_abilene_3d()

    Abilene24_3d = np.load(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d.npy')

    ntimesteps_range = [26]

    Abilene24s_3d = Abilene24_3d[0:8064, :, :]

    # for ntimesteps in ntimesteps_range:
    #     cnn_lstm(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=ntimesteps)

    hyperparams = [2.72, 1, 5.8, 0.4]
    cnn_lstm_test(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=26, hyperparams=hyperparams)

    # errors = np.empty((0, 9))
    # forward_loss_weight_range = np.arange(3.0, 3.5, 0.01)
    # backward_loss_weight = 1
    # consecutive_loss_weight_range = np.arange(5.8, 5.81, 0.1)
    # std_flow_weight_range = np.arange(0.4, 0.401, 0.1)
    #
    # test_name = 'forward_backward_rnn_labeled_features'

    # for std_flow_weight in np.nditer(std_flow_weight_range):
    #     for forward_loss_weight in np.nditer(forward_loss_weight_range):
    #         for consecutive_loss_weight in np.nditer(consecutive_loss_weight_range):
    #
    #             hyperparams = [forward_loss_weight, backward_loss_weight, consecutive_loss_weight,
    #                            std_flow_weight]
    #
    #             errors = try_hyper_parameter(raw_data=Abilene24s_data, dataset_name="Abilene24s",
    #                                          hyperparams=hyperparams, errors=errors)
    #             print(errors)
    #
    #         np.savetxt(
    #             './Errors/[%s][lookback_%i][%s]Errors_by_consecutive_loss.csv' % (test_name, 26, str(hyperparams)),
    #             errors,
    #             delimiter=',')
    #     np.savetxt('./Errors/[%s][lookback_%i][%s]Errors_by_forward.csv' % (test_name, 26, str(hyperparams)), errors,
    #                delimiter=',')
    # np.savetxt('./Errors/[%s][lookback_%i][%s]Errors.csv' % (test_name, 26, str(hyperparams)), errors, delimiter=',')

    # hyperparams = [2.72, 1, 5.8, 0.4]

    # for i in range(20):
    #     errors = try_hyper_parameter(raw_data=Abilene24s_data, dataset_name="Abilene24s", hyperparams=hyperparams,
    #                                  errors=errors)
    # np.savetxt('./Errors/[%s][lookback_%i][%s]Errors_random_monitoring.csv' % (test_name, 26, str(hyperparams)), errors,
    #                delimiter=',')

    # errors = try_hyper_parameter(raw_data=Abilene24s_data, dataset_name="Abilene24s", hyperparams=hyperparams,
    #                                  errors=errors)
    #
    # print(errors)
