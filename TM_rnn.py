from RNN import *
from Utils.DataHelper import *
from Utils.DataPreprocessing import *
import datetime

# PATH CONFIGURATION
FIGURE_DIR = './figures/'
MODEL_RECORDED = './Model_Recorded/'
HOME = os.path.expanduser('~')

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
HIDDEN_DIM = 300
LOOK_BACK = 26
N_EPOCH = 200
BATCH_SIZE = 1024 * 2

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

    if current_ts >= 0:
        for ts in range(look_back - 1):
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


def test_rnn(raw_data, dataset_name):
    test_name='normal_rnn'
    splitting_ratio = [0.7, 0.3]
    a_lookback = range(26, 27, 1)
    errors = np.empty((0, 3))

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
        trainX_path = HOME + '/TM_estimation_dataset/' + dataset_name + \
                      '/TrainX_'+test_name+'_Abilene24s_'+str(splitting_ratio[0])+'_look_back_%i.npy' % look_back
        trainY_path = HOME + '/TM_estimation_dataset/' + dataset_name + \
                      '/TrainY_'+test_name+'_Abilene24s_'+str(splitting_ratio[0])+'_look_back_%i.npy' % look_back
        trainX, trainY = [], []
        if not os.path.isfile(trainX_path):
            print('--- Can not find dataset at: ' + trainX_path)
            print('--- Create TrainX, TrainY ---')
            trainX, trainY = parallel_create_xy_set_spatial_temporal(training_set,
                                                                     look_back=look_back,
                                                                     sampling_ivtl=5,
                                                                     nproc=8,
                                                                     rnn_type='TM_rnn.py')
            print('--- Save TrainX, TrainY to %s ---' % trainX_path)
            np.save(trainX_path, trainX)
            np.save(trainY_path, trainY)
        else:
            print('--- Load TrainX, TrainY from file %s --- ' % trainX_path)
            trainX = np.load(trainX_path)
            trainY = np.load(trainY_path)

        print(trainX.shape)
        # trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))

        rnn = RNN(
            saving_path=model_recorded_path + 'hidden_%i_lookback_%i_epoch_%i/' % (HIDDEN_DIM, look_back, N_EPOCH),
            raw_data=raw_data,
            look_back=look_back,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM)

        if os.path.isfile(path=rnn.saving_path + 'model.json'):
            rnn.load_model_from_disk(model_json_file='model.json',
                                     model_weight_file='model.h5')
        else:
            print('--- Compile model for test_inputshape %s --- ' % (rnn.saving_path))
            rnn.modelContruction(input_shape=(trainX.shape[1], trainX.shape[2]), output_dim=1)
            rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
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

        figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                              + '/'+test_name+'/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, look_back)
        if not os.path.exists(figures_saving_path):
            os.makedirs(figures_saving_path)

        for sampling_ratio in sampling_ratioes:
            pred_tm, measured_matrix = predict_with_loss_test_inputshpae(testing_set,
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

            visualize_retsult_by_flows(y_true=ytrue,
                                       y_pred=pred_tm,
                                       sampling_itvl=5,
                                       description=test_name + '_sampling_%f' % sampling_ratio,
                                       measured_matrix=measured_matrix,
                                       saving_path=HOME + '/TM_estimation_figures/Abilene24s/')

            print('--- Sampling ratio: %.2f - Means abs errors by day ---' % sampling_ratio)
            print(mean_abs_error_by_day)

            plt.title('Means abs errors by day\nSampling: %.2f' % sampling_ratio)
            plt.plot(range(len(mean_abs_error_by_day)), mean_abs_error_by_day)
            plt.xlabel('Day')
            plt.savefig(figures_saving_path +
                        'Means_abs_errors_by_day_sampling_%.2f_lookback_%i.png' % (sampling_ratio, look_back))
            plt.close()

            plt.title('Error ratio by day\nSampling: %.2f' % sampling_ratio)
            plt.plot(range(len(errors_by_day)), errors_by_day)
            plt.xlabel('Day')
            plt.savefig(
                figures_saving_path + 'Error_ratio_by_day_sampling_%.2f_lookback_%i.png' % (sampling_ratio, look_back))
            plt.close()

        plot_errors(x_axis=sampling_ratioes,
                    xlabel='sampling_ratio',
                    errors=errors,
                    filename='%s.png'%test_name,
                    saving_path=figures_saving_path)
    return


if __name__ == "__main__":
    np.random.seed(10)

    Abilene24s_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24s.csv')
    # Abilene_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene.csv')
    # Abilene1_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene1.csv')
    # Abilene3_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene3.csv')

    # run_test_deep_rnn(raw_data=Abilene24s_data, hidden_dim=50)
    # test_rnn_inputshape(Abilene24s_data)
    # running_test_lookback(Abilene24s_data, train_mode=LOOK_BACK_MODE)
    # testing_multiple_rnn(Abilene24s_data)
    # run_test_hidden_layer(raw_data=Abilene24s_data)
    # run_test_bidirect_rnn(raw_data=Abilene24s_data, hidden_dim=300, dataset_name='Abilene3')
    Abilene24s_data = remove_zero_flow(Abilene24s_data, eps=1)
    print(Abilene24s_data.shape)
    test_rnn(raw_data=Abilene24s_data, dataset_name='Abilene24s')
