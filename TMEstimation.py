from RNN import *
from Utils.DataHelper import *
from Utils.DataPreprocessing import *
import datetime
from sklearn.metrics import r2_score

HOME = os.path.expanduser('~')

# PATH CONFIGURATION
FIGURE_DIR = './figures/'
MODEL_RECORDED = './Model_Recorded/'

# DATASET CONFIGURATION
DATASET = ['Geant', 'Geant_noise_removed', 'Abilene', 'Abilene_noise_removed']
GEANT = 0
GEANT_NOISE_REMOVED = 1
ABILENE = 2
ABILENE_NOISE_REMOVE = 3

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
# RNN CONFIGURATION
INPUT_DIM = 100
HIDDEN_DIM = 100
LOOK_BACK = 26
N_EPOCH = 200
BATCH_SIZE = 1024

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


def predict_with_loss(test_set, look_back, rnn_model, sampling_ratio=0.3):
    """

    :param test_set: the original testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :].T
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    # Predict the TM from time slot look_back
    for tslot in range(look_back, test_set.shape[0], 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Scale the input data
        scaler_input = MinMaxScaler().fit(rnn_input)
        scaler_output = MinMaxScaler().fit(np.expand_dims(rnn_input[:, -1], axis=1))
        rnn_input = scaler_input.transform(rnn_input)

        rnn_input = np.reshape(rnn_input, (rnn_input.shape[0], rnn_input.shape[1], 1))

        # Get the TM prediction of next time slot
        predictX = rnn_model.predict(rnn_input)
        predictX = scaler_output.inverse_transform(predictX)

        #####################################################################
        # For testing: if mean flow < 0.5, => next time slot = 0
        _means = np.mean(rnn_input, axis=1)
        _low_mean_indice = np.argwhere(_means < 0.05)
        predictX[_low_mean_indice] = 0.05
        #####################################################################

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf,
                                                   size=(test_set.shape[1]),
                                                   p=[sampling_ratio, 1 - sampling_ratio]), axis=0)
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = predictX.T * inv_sampling
        measured_input = test_set[tslot, :] * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input.T + measured_input.T
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Create new rnn_input by concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=1)
        rnn_input = ret_tm[:, (tslot + 1 - look_back):ret_tm.shape[1]]

    ret_tm = np.reshape(ret_tm, (ret_tm.shape[0], ret_tm.shape[1]))
    return ret_tm.T, measured_matrix


def run_test(test_set, look_back, rnn_model, scalers, description):
    errors = np.empty((0, 4))
    for sampling_ratio in [0.2, 0.3, 0.4]:
        pred_tm, measured_matrix = predict_with_loss(test_set,
                                                     look_back=look_back,
                                                     rnn_model=rnn_model,
                                                     sampling_ratio=sampling_ratio)

        # pred_tm = scaler.inverse_transform(pred_tm)
        pred_tm[pred_tm < 0] = 0
        # ytrue = scaler.inverse_transform(test_set)

        y3 = test_set.flatten()
        y4 = pred_tm.flatten()
        a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
        a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
        pred_confident = r2_score(y3, y4)

        err_rat = error_ratio(y_true=test_set, y_pred=pred_tm, measured_matrix=measured_matrix)

        error = np.expand_dims(np.array([a_nmae, a_nmse, pred_confident, err_rat]), axis=0)

        errors = np.concatenate([errors, error]) if errors.size else error

        visualize_results_by_timeslot(y_true=test_set,
                                      y_pred=pred_tm,
                                      measured_matrix=measured_matrix,
                                      description=description + '_sampling_%f' % sampling_ratio)

        visualize_retsult_by_flows(y_true=test_set,
                                   y_pred=pred_tm,
                                   sampling_itvl=5,
                                   description=description + '_sampling_%f' % sampling_ratio,
                                   measured_matrix=measured_matrix)

    plot_errors(x_axis=[0.2, 0.3, 0.4],
                xlabel='sampling_ratio',
                errors=errors,
                filename='Error_sampling_ratio.png')

    return error


def plot_errors(x_axis, xlabel, errors, filename, saving_path='/home/anle/TM_estimation_figures/'):
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    plt.title('Errors')
    plt.plot(x_axis, errors[:, 0], label='NMAE')
    plt.plot(x_axis, errors[:, 1], label='NMSE')
    plt.xlabel(xlabel)
    if errors.shape[1] == 4:
        plt.plot(x_axis, errors[:, 3], label='Error_ratio')
    plt.legend()

    plt.savefig(saving_path + 'Errors_' + filename)
    plt.close()

    plt.title('R2-Score')
    plt.plot(x_axis, errors[:, 2])
    plt.xlabel(xlabel)
    plt.savefig(saving_path + 'R2_Score_' + filename)
    plt.close()


def run_test_deep_rnn(raw_data, hidden_dim=200, dropout=0.2, look_back=26, train_mode=HIDDEN_DIM_MODE):
    splitting_ratio = [0.6, 0.4]
    scaler = DataNormalizer()
    errors = np.empty((0, 3))
    model_recorded_path = './Model_Recorded/Abilene24s/' + 'DRNN' + '/'

    for n_layers in range(2, 6, 1):
        train_set, test_set = prepare_train_test_set(data=raw_data,
                                                     sampling_itvl=5,
                                                     splitting_ratio=splitting_ratio)

        train_scaler = MinMaxScaler().fit(train_set)
        train_set = train_scaler.transform(train_set)

        test_scaler = MinMaxScaler().fit(test_set)
        test_set = test_scaler.transform(test_set)

        # Create XY set using 4 features (xc, xp, tc, dw)
        trainX, trainY = [], []
        if not os.path.isfile('../TM_estimation_RNN/Dataset/TrainX_Abilene24s_0.7.npy'):
            print('Create XY set')
            trainX, trainY = parallel_create_xy_set_spatial_temporal(train_set,
                                                                     look_back=look_back,
                                                                     sampling_ivtl=5,
                                                                     nproc=8)
            np.save('../TM_estimation_RNN/Dataset/TrainX_Abilene24s_0.7', trainX)
            np.save('../TM_estimation_RNN/Dataset/TrainY_Abilene24s_0.7', trainY)
        else:
            print('Load xy set from file')
            trainX = np.load('../TM_estimation_RNN/Dataset/TrainX_Abilene24s_0.7.npy')
            trainY = np.load('../TM_estimation_RNN/Dataset/TrainY_Abilene24s_0.7.npy')

        rnn = RNN(raw_data=raw_data,
                  look_back=look_back,
                  n_epoch=N_EPOCH,
                  batch_size=BATCH_SIZE,
                  hidden_dim=hidden_dim,
                  saving_path=model_recorded_path + 'nlayers_%i/' % n_layers + 'nhidden_%i/' % hidden_dim)

        if os.path.isfile(path=rnn.saving_path + 'model.json'):
            rnn.load_model_from_disk()
        else:
            print('--- Compile model for nlayers %i ---' % n_layers)
            rnn.deep_rnn_io_model_construction(input_shape=(trainX.shape[1], trainX.shape[2]),
                                               n_layers=n_layers,
                                               drop_out=dropout,
                                               output_dim=1)
            history = rnn.model.fit(trainX,
                                    trainY,
                                    epochs=rnn.n_epoch,
                                    batch_size=rnn.batch_size,
                                    validation_split=0.05,
                                    callbacks=rnn.callbacks_list,
                                    verbose=0)
            rnn.save_model_to_disk()
            rnn.plot_model_history(history)

        errors = np.empty((0, 4))
        for sampling_ratio in [0.2, 0.3, 0.4]:
            pred_tm, measured_matrix = predict_with_loss_test_inputshpae(test_set,
                                                                         look_back=look_back,
                                                                         rnn_model=rnn.model,
                                                                         sampling_ratio=sampling_ratio)

            pred_tm = test_scaler.inverse_transform(pred_tm)
            pred_tm[pred_tm < 0] = 0
            ytrue = test_scaler.inverse_transform(test_set)

            y3 = ytrue.flatten()
            y4 = pred_tm.flatten()
            a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
            a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
            pred_confident = r2_score(y3, y4)

            err_rat = error_ratio(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix)

            error = np.expand_dims(np.array([a_nmae, a_nmse, pred_confident, err_rat]), axis=0)

            errors = np.concatenate([errors, error], axis=0)

            visualize_results_by_timeslot(y_true=ytrue,
                                          y_pred=pred_tm,
                                          measured_matrix=measured_matrix,
                                          description='DeepRNN' + '_nlayers_%i' % n_layers + '_nhidden_%i' % hidden_dim)

            visualize_retsult_by_flows(y_true=ytrue,
                                       y_pred=pred_tm,
                                       sampling_itvl=5,
                                       description='DeepRNN' + '_nlayers_%i' % n_layers + '_nhidden_%i' % hidden_dim,
                                       measured_matrix=measured_matrix)

        plot_errors(x_axis=[0.2, 0.3, 0.4],
                    xlabel='sampling_ratio',
                    errors=errors,
                    filename='deepRNN_nlayer_%i_hidden_unit_%i.png' % (n_layers, hidden_dim),
                    saving_path='/home/hong/TM_estimation_figures/DeepRNN/n_layers_%i_hidden_%i/' % (
                        n_layers, hidden_dim))


def run_test_bidirect_rnn(raw_data, hidden_dim=300, dataset_name='Abilene24s'):
    test_name = 'test_bidirect_rnn'
    splitting_ratio = [0.6, 0.4]
    errors = np.empty((0, 3))
    a_lookback = range(25, 30, 1)

    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    for look_back in a_lookback:
        train_set, test_set = prepare_train_test_set(data=raw_data,
                                                     sampling_itvl=5,
                                                     splitting_ratio=splitting_ratio)

        ################################################################################################################
        #                                         For testing Flows Clustering                                         #

        seperated_train_set, centers_train_set = mean_std_flows_clustering(train_set)
        training_set, train_scalers, train_cluster_lens = different_flows_scaling(seperated_train_set[1:],
                                                                                  centers_train_set[1:])

        seperated_test_set, centers_test_set = mean_std_flows_clustering(test_set)
        testing_set, test_scalers, test_cluster_lens = different_flows_scaling(seperated_test_set[1:],
                                                                               centers_test_set[1:])

        ################################################################################################################

        # Create XY set using 4 features (xc, xp, tc, dw)
        trainX_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                      '/TrainX_bidirectional_' + dataset_name + '_0.7_look_back_%i.npy' % look_back
        trainY_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                      '/TrainY_bidirectional_' + dataset_name + '_0.7_look_back_%i.npy' % look_back
        testX_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                     '/TestX_bidirectional_' + dataset_name + '_0.3_look_back_%i.npy' % look_back
        testY_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                     '/TestY_bidirectional_' + dataset_name + '_0.3_look_back_%i.npy' % look_back

        trainX, trainY = [], []
        if not os.path.isfile(trainX_path):

            print('--- Create TrainX, TrainY ---')
            trainX, trainY = parallel_create_xy_set_spatial_temporal(training_set,
                                                                     look_back=look_back,
                                                                     sampling_ivtl=5,
                                                                     nproc=8,
                                                                     rnn_type='bidirectional_rnn')
            print('--- Save TrainX, TrainY to %s ---' % trainX_path)
            np.save(trainX_path, trainX)
            np.save(trainY_path, trainY)
        else:
            print('--- Load TrainX, TrainY from file %s --- ' % trainX_path)
            trainX = np.load(trainX_path)
            trainY = np.load(trainY_path)

        print(trainX.shape)
        trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))

        rnn = RNN(
            saving_path=model_recorded_path + 'hidden_%i_lookback_%i_epoch_%i/' % (hidden_dim, look_back, N_EPOCH),
            raw_data=raw_data,
            look_back=look_back,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=hidden_dim)

        if os.path.isfile(path=rnn.saving_path + 'model.json'):
            rnn.load_model_from_disk(model_json_file='model.json',
                                     model_weight_file='model.h5')
        else:
            print('--- Compile model for test_inputshape %s --- ' % (rnn.saving_path))
            rnn.bidirectional_model_construction(input_shape=(trainX.shape[1], trainX.shape[2]), output_dim=look_back)
            history = rnn.model.fit(trainX,
                                    trainY,
                                    epochs=rnn.n_epoch,
                                    batch_size=rnn.batch_size,
                                    validation_split=0.05,
                                    callbacks=rnn.callbacks_list,
                                    verbose=1)
            rnn.save_model_to_disk()
            # rnn.plot_model_history(history)

        print(rnn.model.summary())

        errors = np.empty((0, 4))
        sampling_ratioes = [0.2, 0.3, 0.4, 0.5]

        figures_saving_path = '/home/anle/TM_estimation_figures/' + dataset_name \
                              + '/bidirectional_rnn/test_hidden_%i_look_back_%i/' % (hidden_dim, look_back)

        if not os.path.exists(figures_saving_path):
            os.makedirs(figures_saving_path)

        for sampling_ratio in sampling_ratioes:
            pred_tm, measured_matrix = predict_with_loss_bidirectional_rnn(testing_set,
                                                                           look_back=look_back,
                                                                           rnn_model=rnn.model,
                                                                           sampling_ratio=sampling_ratio)

            ############################################################################################################
            #                                         For testing Flows Clustering

            pred_tm = different_flows_invert_scaling(pred_tm, scalers=test_scalers, cluster_lens=test_cluster_lens)
            pred_tm[pred_tm < 0] = 0
            ytrue = different_flows_invert_scaling(data=testing_set, scalers=test_scalers,
                                                   cluster_lens=test_cluster_lens)
            ############################################################################################################

            errors_by_day = calculate_error_ratio_by_day(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix,
                                                         sampling_itvl=5)
            mean_abs_error_by_day = mean_absolute_errors_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5,
                                                                measured_matrix=measured_matrix)

            y3 = ytrue.flatten()
            y4 = pred_tm.flatten()
            a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
            a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
            pred_confident = r2_score(y3, y4)

            err_rat = error_ratio(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix)

            error = np.expand_dims(np.array([a_nmae, a_nmse, pred_confident, err_rat]), axis=0)

            errors = np.concatenate([errors, error], axis=0)

            # visualize_results_by_timeslot(y_true=ytrue,
            #                               y_pred=pred_tm,
            #                               measured_matrix=measured_matrix,
            #                               description='diff_scalers_testRNN_input' + '_sampling_%f' % sampling_ratio)

            # visualize_retsult_by_flows(y_true=ytrue,
            #                            y_pred=pred_tm,
            #                            sampling_itvl=5,
            #                            description='bidirectional_rnn_hidden_%i_sampling_%f' % (
            #                                hidden_dim, sampling_ratio),
            #                            measured_matrix=measured_matrix,
            #                            saving_path='/home/anle/TM_estimation_figures/' + dataset_name + '/')

            print('--- Sampling ratio: %.2f - Means abs errors by day ---' % sampling_ratio)
            print(mean_abs_error_by_day)

            plt.title('Means abs errors by day\nSampling: %.2f' % sampling_ratio)
            plt.plot(range(len(mean_abs_error_by_day)), mean_abs_error_by_day)
            plt.xlabel('Day')
            plt.savefig(figures_saving_path + 'Means_abs_errors_by_day_sampling_%.2f.png' % sampling_ratio)
            plt.close()

        plot_errors(x_axis=sampling_ratioes,
                    xlabel='sampling_ratio',
                    errors=errors,
                    filename='bidirectional_rnn.png',
                    saving_path=figures_saving_path)

    return


def prepare_input_online_predict(pred_tm, look_back, timeslot, day_in_week, time_in_day, sampling_itvl=5):
    k = 4
    day_size = 24 * (60 / sampling_itvl)

    dataX = np.empty((0, look_back, k))

    for j in xrange(pred_tm.shape[1]):
        sample = []

        # Get x_c for all look_back
        xc = pred_tm[timeslot:(timeslot + look_back), j]
        sample.append(xc)

        # Get x_p: x_p = x_c in the first day in dataset
        if timeslot - day_size + 1 < 0:
            xp = pred_tm[timeslot:(timeslot + look_back), j]
            sample.append(xp)
        else:
            xp = pred_tm[(timeslot + 1 - day_size):(timeslot + look_back - day_size + 1), j]
            sample.append(xp)

        # Get the current timeslot
        tc = time_in_day[timeslot:(timeslot + look_back)]
        sample.append(tc)

        # Get the current day in week
        dw = day_in_week[timeslot:(timeslot + look_back)]
        sample.append(dw)

        # Stack the feature into a sample and reshape it into the input shape of RNN: (1, timestep, features)
        a_sample = np.reshape(np.array(sample).T, (1, look_back, k))

        # Concatenate the samples into a dataX
        dataX = np.concatenate([dataX, a_sample], axis=0)
    return dataX


def predict_with_loss_test_inputshpae(test_set, look_back, rnn_model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)
    n_days = int(test_set.shape[0] / day_size) if (test_set.shape[0] % day_size) == 0 \
        else int(test_set.shape[0] / day_size) + 1

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :].T
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    day_in_week = time_scaler(range(1, 8, 1) * day_size * n_days, feature_range=(0, 1))
    time_in_day = time_scaler(range(day_size) * n_days, feature_range=(0, 1))

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - look_back, 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input = prepare_input_online_predict(ret_tm.T, look_back, tslot, day_in_week, time_in_day, sampling_itvl=5)

        # Get the TM prediction of next time slot
        predictX = rnn_model.predict(rnn_input)

        #####################################################################
        # # For testing: if mean flow < 0.5, => next time slot = 0
        # _means = np.mean(rnn_input, axis=1)
        # _low_mean_indice = np.argwhere(_means < 0.05)
        # predictX[_low_mean_indice] = 0.05
        #####################################################################

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf,
                                                   size=(test_set.shape[1]),
                                                   p=[sampling_ratio, 1 - sampling_ratio]), axis=0)
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = predictX.T * inv_sampling
        measured_input = test_set[tslot + look_back, :] * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input.T + measured_input.T
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=1)

    return ret_tm.T, measured_matrix


def update_predicted_data(pred_X, pred_tm, current_ts, look_back, measured_matrix):
    sampling_measured_matrix = measured_matrix[(current_ts + 1):(current_ts + look_back), :]
    inv_sampling_measured_matrix = np.invert(sampling_measured_matrix)
    bidirect_rnn_pred_value = pred_X[0:-1, :] * inv_sampling_measured_matrix

    if current_ts >=0:
        for ts in range(look_back-1):
            plt.plot(pred_tm[current_ts + ts + 1, :], label='Old_TM')
            plt.plot(pred_X[ts, :], label='Updated_TM')
            plt.legend()
            plt.show()

    pred_tm[(current_ts + 1):(current_ts + look_back), :] = pred_tm[(current_ts + 1):(current_ts + look_back), :] * \
                                                            sampling_measured_matrix + bidirect_rnn_pred_value
    return pred_tm.T


def predict_with_loss_bidirectional_rnn(test_set, look_back, rnn_model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)
    n_days = int(test_set.shape[0] / day_size) if (test_set.shape[0] % day_size) == 0 \
        else int(test_set.shape[0] / day_size) + 1

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :].T
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    day_in_week = time_scaler(range(1, 8, 1) * day_size * n_days, feature_range=(0, 1))
    time_in_day = time_scaler(range(day_size) * n_days, feature_range=(0, 1))

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - look_back, 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input = prepare_input_online_predict(ret_tm.T, look_back, tslot, day_in_week, time_in_day,
                                                 sampling_itvl=5)

        # Get the TM prediction of next time slot
        predictX = rnn_model.predict(rnn_input)
        predictX = np.reshape(predictX, (predictX.shape[0], predictX.shape[1]))

        #####################################################################
        # # For testing: if mean flow < 0.5, => next time slot = 0
        # _means = np.mean(rnn_input, axis=1)
        # _low_mean_indice = np.argwhere(_means < 0.05)
        # predictX[_low_mean_indice] = 0.05
        #####################################################################

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf,
                                                   size=(test_set.shape[1]),
                                                   p=[sampling_ratio, 1 - sampling_ratio]), axis=0)
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = predictX.T[-1, :] * inv_sampling
        measured_input = test_set[tslot + look_back, :] * sampling

        # Update the predicted historical data
        ret_tm = update_predicted_data(pred_X=predictX.T, pred_tm=ret_tm.T, current_ts=tslot, look_back=look_back,
                                       measured_matrix=measured_matrix)

        # Merge value from pred_input and measured_input
        new_input = pred_input.T + measured_input.T
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=1)

    return ret_tm.T, measured_matrix


def predict_without_loss(test_set, look_back, rnn_model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)
    n_days = int(test_set.shape[0] / day_size) if (test_set.shape[0] % day_size) == 0 \
        else int(test_set.shape[0] / day_size) + 1

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :].T
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    day_in_week = time_scaler(range(1, 8, 1) * day_size * n_days, feature_range=(0, 1))
    time_in_day = time_scaler(range(day_size) * n_days, feature_range=(0, 1))

    prediction_tm = np.empty((0, test_set.shape[1]))
    prediction_tm = np.concatenate([prediction_tm, rnn_input.T], axis=0)

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - look_back, 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input = prepare_input_online_predict(ret_tm.T, look_back, tslot, day_in_week, time_in_day, sampling_itvl=5)

        # Get the TM prediction of next time slot
        predictX = rnn_model.predict(rnn_input)
        prediction_tm = np.concatenate([prediction_tm, predictX.T], axis=0)

        measured_input = test_set[tslot + look_back, :]

        # Merge value from pred_input and measured_input
        new_input = np.expand_dims(measured_input.T, axis=1)
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=1)

    return prediction_tm, measured_matrix


def predict_all_consecutive_loss(test_set, look_back, rnn_model, sampling_ratio):
    """
    Predict the TM in case of a consecutive loss (50 min)
    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """
    day_size = 24 * (60 / 5)
    n_days = int(test_set.shape[0] / day_size) if (test_set.shape[0] % day_size) == 0 \
        else int(test_set.shape[0] / day_size) + 1

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :].T
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    day_in_week = time_scaler(range(1, 8, 1) * day_size * n_days, feature_range=(0, 1))
    time_in_day = time_scaler(range(day_size) * n_days, feature_range=(0, 1))

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - look_back, 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input = prepare_input_online_predict(ret_tm.T, look_back, tslot, day_in_week, time_in_day, sampling_itvl=5)

        # Get the TM prediction of next time slot
        predictX = rnn_model.predict(rnn_input)

        #####################################################################
        # # For testing: if mean flow < 0.5, => next time slot = 0
        # _means = np.mean(rnn_input, axis=1)
        # _low_mean_indice = np.argwhere(_means < 0.05)
        # predictX[_low_mean_indice] = 0.05
        #####################################################################

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        date = int(tslot / day_size)
        sampling = []
        if (tslot + look_back >= (date * 288 + 120)) & (tslot + look_back < (date * 288 + 130)):

            sampling = np.expand_dims(np.random.choice(tf,
                                                       size=(test_set.shape[1]),
                                                       p=[0, 1]), axis=0)
        else:
            sampling = np.expand_dims(np.random.choice(tf,
                                                       size=(test_set.shape[1]),
                                                       p=[sampling_ratio, 1 - sampling_ratio]), axis=0)

        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = predictX.T * inv_sampling
        measured_input = test_set[tslot + look_back, :] * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input.T + measured_input.T
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=1)

    return ret_tm.T, measured_matrix


def test_rnn_inputshape(raw_data):
    splitting_ratio = [0.6, 0.4]
    a_lookback = range(25, 30, 1)
    errors = np.empty((0, 3))
    dataset_name = 'Abilene24s'

    model_recorded_path = './Model_Recorded/' + dataset_name + '/Diff_scaled/'

    for look_back in a_lookback:
        train_set, test_set = prepare_train_test_set(data=raw_data,
                                                     sampling_itvl=5,
                                                     splitting_ratio=splitting_ratio)

        ################################################################################################################
        #                                         For testing Flows Clustering                                         #

        seperated_train_set, centers_train_set = mean_std_flows_clustering(train_set)
        training_set, train_scalers, train_cluster_lens = different_flows_scaling(seperated_train_set[1:],
                                                                                  centers_train_set[1:])

        seperated_test_set, centers_test_set = mean_std_flows_clustering(test_set)
        testing_set, test_scalers, test_cluster_lens = different_flows_scaling(seperated_test_set[1:],
                                                                               centers_test_set[1:])

        ################################################################################################################

        ################################################################################################################
        # Scale the training set using the MinMaxScaler (only for testing).
        # The approximate scaling will be defined later
        # train_scaler = MinMaxScaler().fit(train_set)
        # train_set = train_scaler.transform(train_set)
        #
        # test_scaler = MinMaxScaler().fit(test_set)
        # test_set = test_scaler.transform(test_set)
        ################################################################################################################

        # # Create XY set using only 1 feature (history traffic volume)
        # trainX, trainY = create_xy_set(train_set, look_back=look_back)
        # testX, testY = create_xy_set(test_set, look_back=look_back)

        # Create XY set using 4 features (xc, xp, tc, dw)
        trainX_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                      '/TrainX_diff_scaled_Abilene24s_0.7_look_back_%i.npy' % look_back
        trainY_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                      '/TrainY_diff_scaled_Abilene24s_0.7_look_back_%i.npy' % look_back
        testX_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                     '/TestX_diff_scaled_Abilene24s_0.3_look_back_%i.npy' % look_back
        testY_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                     '/TestY_diff_scaled_Abilene24s_0.3_look_back_%i.npy' % look_back

        trainX, trainY = [], []
        if not os.path.isfile(trainX_path):

            print('--- Create TrainX, TrainY ---')
            trainX, trainY = parallel_create_xy_set_spatial_temporal(training_set,
                                                                     look_back=look_back,
                                                                     sampling_ivtl=5,
                                                                     nproc=8)
            print('--- Save TrainX, TrainY to %s ---' % trainX_path)
            np.save(trainX_path, trainX)
            np.save(trainY_path, trainY)
        else:
            print('--- Load TrainX, TrainY from file %s --- ' % trainX_path)
            trainX = np.load(trainX_path)
            trainY = np.load(trainY_path)

        print(trainX.shape)
        trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))

        rnn = RNN(saving_path=model_recorded_path + 'lookback_%i/' % look_back,
                  raw_data=raw_data,
                  look_back=look_back,
                  n_epoch=100,
                  batch_size=BATCH_SIZE,
                  hidden_dim=HIDDEN_DIM)

        if os.path.isfile(path=rnn.saving_path + 'model.json'):
            rnn.load_model_from_disk(model_json_file='model.json',
                                     model_weight_file='model.h5')
        else:
            print('--- Compile model for test_inputshape %s --- ' % (rnn.saving_path))
            rnn.modelContruction(input_shape=(trainX.shape[1], trainX.shape[2]), output_dim=1)
            history = rnn.model.fit(trainX,
                                    trainY,
                                    epochs=rnn.n_epoch,
                                    batch_size=rnn.batch_size,
                                    validation_split=0.05,
                                    callbacks=rnn.callbacks_list,
                                    verbose=1)
            rnn.save_model_to_disk()
            # rnn.plot_model_history(history)

        errors = np.empty((0, 4))
        sampling_ratioes = [0.2, 0.3, 0.4, 0.5]
        for sampling_ratio in sampling_ratioes:
            # pred_tm, measured_matrix = predict_with_loss_test_inputshpae(testing_set,
            #                                                              look_back=look_back,
            #                                                              rnn_model=rnn.model,
            #                                                              sampling_ratio=sampling_ratio)

            # pred_tm, measured_matrix = predict_without_loss(testing_set,
            #                                                              look_back=look_back,
            #                                                              rnn_model=rnn.model,
            #                                                              sampling_ratio=sampling_ratio)

            pred_tm, measured_matrix = predict_all_consecutive_loss(testing_set,
                                                                    look_back=look_back,
                                                                    rnn_model=rnn.model,
                                                                    sampling_ratio=sampling_ratio)

            ############################################################################################################
            #                                         For testing Flows Clustering

            pred_tm = different_flows_invert_scaling(pred_tm, scalers=test_scalers, cluster_lens=test_cluster_lens)
            pred_tm[pred_tm < 0] = 0
            ytrue = different_flows_invert_scaling(data=testing_set, scalers=test_scalers,
                                                   cluster_lens=test_cluster_lens)
            ############################################################################################################

            # y3 = ytrue.flatten()
            # y4 = pred_tm.flatten()
            # a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
            # a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
            # pred_confident = r2_score(y3, y4)
            #
            # err_rat = error_ratio(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix)
            #
            # error = np.expand_dims(np.array([a_nmae, a_nmse, pred_confident, err_rat]), axis=0)
            #
            # errors = np.concatenate([errors, error], axis=0)
            #
            # # visualize_results_by_timeslot(y_true=ytrue,
            # #                               y_pred=pred_tm,
            # #                               measured_matrix=measured_matrix,
            # #                               description='diff_scalers_testRNN_input' + '_sampling_%f' % sampling_ratio)
            #
            # visualize_retsult_by_flows(y_true=ytrue,
            #                            y_pred=pred_tm,
            #                            sampling_itvl=5,
            #                            description='diff_scalers_testRNN_input' + '_sampling_%f' % sampling_ratio,
            #                            measured_matrix=measured_matrix,
            #                            saving_path='/home/anle/TM_estimation_figures/Abilene24s/')

            ############################################################################################################
            #                                         Consecutive loss testing                                         #
            pred_consec_loss = np.empty((0, ytrue.shape[1]))
            y_true_consec_loss = pred_consec_loss
            day_size = 24 * (60 / 5)
            for tslot in range(pred_tm.shape[0]):
                date = int(tslot / day_size) if tslot % day_size == 0 else int(tslot / day_size) + 1
                if (tslot >= (date * 288 + 120)) & (tslot < (date * 288 + 130)):
                    pred_consec_loss = np.append([pred_consec_loss, np.expand_dims(pred_tm[tslot, :], axis=0)], axis=0)
                    y_true_consec_loss = np.append([y_true_consec_loss, np.expand_dims(ytrue[tslot, :], axis=0)],
                                                   axis=0)

            y3 = y_true_consec_loss.flatten()
            y4 = pred_consec_loss.flatten()
            a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
            a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
            pred_confident = r2_score(y3, y4)

            err_rat = error_ratio(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix)

            error = np.expand_dims(np.array([a_nmae, a_nmse, pred_confident, err_rat]), axis=0)

            errors = np.concatenate([errors, error], axis=0)
            ############################################################################################################

        plot_errors(x_axis=sampling_ratioes,
                    xlabel='sampling_ratio',
                    errors=errors,
                    filename='test_input.png',
                    saving_path='/home/anle/TM_estimation_figures/' + dataset_name + '/test_RNN_input/diff_scalers'
                                                                                     '/look_back_%i/' % look_back)
    return


def testing_multiple_rnn(raw_data, dataset_name):
    test_name = 'multiple_rnn'
    sampling_ratioes = [0.2, 0.3, 0.4, 0.5]
    splitting_ratio = [0.6, 0.4]
    look_back = 15
    errors = np.empty((0, 3))

    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'
    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    ################################################################################################################
    #                                         For testing Flows Clustering                                         #

    seperated_train_set, centers_train_set = mean_std_flows_clustering(train_set)
    training_sets, train_scalers, train_cluster_lens = different_flows_scaling_without_join(seperated_train_set[1:],
                                                                                            centers_train_set[1:])

    seperated_test_set, centers_test_set = mean_std_flows_clustering(test_set)
    testing_sets, test_scalers, test_cluster_lens = different_flows_scaling_without_join(seperated_test_set[1:],
                                                                                         centers_test_set[1:])

    ################################################################################################################

    ################################################################################################################
    # Scale the training set using the MinMaxScaler (only for testing).
    # The approximate scaling will be defined later
    # train_scaler = MinMaxScaler().fit(train_set)
    # train_set = train_scaler.transform(train_set)
    #
    # test_scaler = MinMaxScaler().fit(test_set)
    # test_set = test_scaler.transform(test_set)
    ################################################################################################################

    # # Create XY set using only 1 feature (history traffic volume)
    # trainX, trainY = create_xy_set(train_set, look_back=look_back)
    # testX, testY = create_xy_set(test_set, look_back=look_back)

    # Create XY set using 4 features (xc, xp, tc, dw)
    trainXs, trainYs = [], []
    testXs, testYs = [], []
    rnns = []
    predXs = []
    y_trues = []
    measured_matrices = []
    for i in range(len(training_sets)):
        trainX_path = '/home/anle/TM_estimation_dataset/TrainX_%i_MultiRNN_Abilene24s_0.7.npy' % i
        trainY_path = '/home/anle/TM_estimation_dataset/TrainY_%i_MultiRNN_Abilene24s_0.7.npy' % i

        if not os.path.isfile(trainX_path):
            print('--- Create TrainX, TrainY ---')
            trainX, trainY = parallel_create_xy_set_spatial_temporal(training_sets[i],
                                                                     look_back=look_back,
                                                                     sampling_ivtl=5,
                                                                     nproc=8)
            print('--- Save TrainX, TrainY to %s ---' % trainX_path)
            np.save(trainX_path, trainX)
            np.save(trainY_path, trainY)
            trainXs.append(trainX)
            trainYs.append(trainY)
        else:
            print('--- Load TrainX, TrainY from file %s --- ' % trainX_path)
            trainX = np.load(trainX_path)
            trainY = np.load(trainY_path)
            trainXs.append(trainX)
            trainYs.append(trainY)

        rnns.append(RNN(saving_path=model_recorded_path + 'multiRNNs/lookback_%i/cluster_%i/' % (look_back, i),
                        raw_data=raw_data,
                        look_back=look_back,
                        n_epoch=100,
                        batch_size=BATCH_SIZE,
                        hidden_dim=HIDDEN_DIM))

        if os.path.isfile(path=rnns[i].saving_path + 'model.json'):
            rnns[i].load_model_from_disk(model_json_file='model.json',
                                         model_weight_file='model.h5')
        else:
            print('--- Compile model for test_inputshape ---')
            rnns[i].modelContruction(input_shape=(trainXs[i].shape[1], trainXs[i].shape[2]), output_dim=1)
            history = rnns[i].model.fit(trainXs[i],
                                        trainYs[i],
                                        epochs=rnns[i].n_epoch,
                                        batch_size=rnns[i].batch_size,
                                        validation_split=0.05,
                                        callbacks=rnns[i].callbacks_list,
                                        verbose=0)
            rnns[i].save_model_to_disk()
            # rnns[i].plot_model_history(history)

        for sampling_ratio in sampling_ratioes:
            pred_tm, measured_matrix = predict_with_loss_test_inputshpae(testing_sets[i],
                                                                         look_back=look_back,
                                                                         rnn_model=rnns[i].model,
                                                                         sampling_ratio=sampling_ratio)

            pred_tm = test_scalers[i].inverse_transform(pred_tm)
            pred_tm[pred_tm < 0] = 0
            ytrue = test_scalers[i].inverse_transform(testing_sets[i])

            ############################################################################################################
            #                                         For testing Flows Clustering

            # pred_tm = different_flows_invert_scaling(pred_tm, scalers=test_scalers, cluster_lens=test_cluster_lens)
            # pred_tm[pred_tm < 0] = 0
            # ytrue = different_flows_invert_scaling(data=testing_set, scalers=test_scalers,
            #                                        cluster_lens=test_cluster_lens)
            ############################################################################################################

            predXs.append(pred_tm)
            y_trues.append(ytrue)
            measured_matrices.append(measured_matrix)

    errors = np.empty((0, 4))

    for sampling_idx in range(len(sampling_ratioes)):

        tm_prediction = np.empty((test_set.shape[0], 0))
        measured_matrix = np.empty((test_set.shape[0], 0))
        ytrue = tm_prediction
        for clusterid in range(len(training_sets)):
            tm_prediction = np.concatenate(
                [tm_prediction, predXs[sampling_idx + clusterid * len(sampling_ratioes)]], axis=1)
            ytrue = np.concatenate([ytrue, y_trues[sampling_idx + clusterid * len(sampling_ratioes)]], axis=1)
            measured_matrix = np.concatenate(
                [measured_matrix, measured_matrices[sampling_idx + clusterid * len(sampling_ratioes)]], axis=1)

        y3 = ytrue.flatten()
        y4 = tm_prediction.flatten()
        a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
        a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
        pred_confident = r2_score(y3, y4)

        err_rat = error_ratio(y_true=ytrue, y_pred=tm_prediction, measured_matrix=measured_matrix)

        error = np.expand_dims(np.array([a_nmae, a_nmse, pred_confident, err_rat]), axis=0)

        errors = np.concatenate([errors, error], axis=0)

        # visualize_results_by_timeslot(y_true=ytrue,
        #                               y_pred=pred_tm,
        #                               measured_matrix=measured_matrix,
        #                               description='diff_scalers_testRNN_input' + '_sampling_%f' % sampling_ratio)

        visualize_retsult_by_flows(y_true=ytrue,
                                   y_pred=tm_prediction,
                                   sampling_itvl=5,
                                   description='multiRNNs' + '_sampling_%f' % sampling_ratioes[sampling_idx],
                                   measured_matrix=measured_matrix)

    plot_errors(x_axis=sampling_ratioes,
                xlabel='sampling_ratio',
                errors=errors,
                filename='multiRNNs.png',
                saving_path='/home/anle/TM_estimation_figures/MultiRNNs/')

    return


if __name__ == "__main__":
    np.random.seed(10)

    # convert_abilene_24()
    Abilene24_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24_original.csv')
    plot_flow_acf(Abilene24_data)
    # Abilene_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene.csv')
    # Abilene1_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene1.csv')
    # Abilene3_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene3.csv')

    # run_test_deep_rnn(raw_data=Abilene24s_data, hidden_dim=50)
    # test_rnn_inputshape(Abilene24s_data)
    # running_test_lookback(Abilene24s_data, train_mode=LOOK_BACK_MODE)
    # testing_multiple_rnn(Abilene24s_data)
    # run_test_hidden_layer(raw_data=Abilene24s_data)
    # run_test_bidirect_rnn(raw_data=Abilene24s_data, hidden_dim=300, dataset_name='Abilene3')

    # # Test Flows Clustering
    # sample_flows = Abilene24s_data
    #
    # # n_cluster = predict_k(pairwise_distances(sample_flows.T))
    # # print('ncluster: %i' % n_cluster)
    # # spatial_cluster = SpatialClustering(sample_flows)
    # # sp = spatial_cluster.spectral_clustering(n_clusters=3)
    # # print(sp.labels_)
    # # labels = sp.labels_
    #
    # means = np.mean(sample_flows, axis=0)
    # means = np.expand_dims(means, axis=1)
    # stds = np.expand_dims(np.std(sample_flows, axis=0), axis=1)
    # mean_stds = np.concatenate([means, stds], axis=1)
    # print(mean_stds)
    #
    # spatial_cluster = SpatialClustering(mean_stds)
    # db, n_clusters = spatial_cluster.flows_DBSCAN(eps=0.4, min_samples=int(0.05*sample_flows.shape[1]))
    # labels = db.labels_
    #
    # print(labels)
    #
    # labels = labels + 1
    #
    # for label in range(n_clusters):
    #     if not os.path.exists('/home/anle/TM_estimation_figures/FlowClustering/Class_%i/'%label):
    #         os.makedirs('/home/anle/TM_estimation_figures/FlowClustering/Class_%i/'%label)
    #
    #     label_flows = np.argwhere(labels==label)
    #     for flowID in np.nditer(label_flows):
    #         plt.title('Flow %i - Class %i'%(flowID, labels[flowID]))
    #         plt.plot(sample_flows[:,flowID])
    #         plt.xlabel('Time')
    #         plt.ylabel('Mbps')
    #         plt.savefig('/home/anle/TM_estimation_figures/FlowClustering/Class_%i/Flow_%i.png'%(label, flowID))
    #         plt.close()
