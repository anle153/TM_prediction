from RNN import *
from Utils.DataHelper import *
from Utils.DataPreprocessing import *
from sklearn.metrics import r2_score
import datetime

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
# RNN CONFIGURATION
INPUT_DIM = 100
HIDDEN_DIM = 300
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


def set_measured_flow(rnn_input, forward_pred, backward_pred, measured_matrix, sampling_ratio):
    """
    :param rnn_input:
    :param forward_pred:
    :param backward_pred:
    :param updated_rnn:
    :param measured_matrix:
    :return:
    """

    w = calculate_measured_weights(rnn_input=rnn_input,
                                   forward_pred=forward_pred,
                                   backward_pred=backward_pred,
                                   measured_matrix=measured_matrix)

    sampling = np.zeros(shape=(rnn_input.shape[0]))
    m = int(sampling_ratio * rnn_input.shape[0])
    sorted_idx_w = np.argpartition(w, m)
    sampling[sorted_idx_w[:m]] = 1

    sampling = np.expand_dims(sampling, axis=0)

    return sampling.astype(bool)


def initialized_training_pharse(rnn, initial_training_data, look_back):
    labels = np.ones((look_back * 5, initial_training_data.shape[1]))
    initial_trainX, initial_trainY = create_xy_labeled(raw_data=initial_training_data,
                                                       data=initial_training_data,
                                                       look_back=look_back,
                                                       labels=labels)

    history = rnn.model.fit(initial_trainX,
                            initial_trainY,
                            epochs=rnn.n_epoch,
                            batch_size=rnn.batch_size,
                            validation_split=0.1,
                            callbacks=rnn.callbacks_list,
                            verbose=1)

    return rnn


def online_training(rnn, training_set, look_back, sampling_ratio):
    """
    In the online training function, a part of the predictions are used as input for the next training data based on
    the sampling ratio.
    :param rnn: the rnn instance
    :param training_set: array-like - shape = (timeslot x OD): the training set.
    :param look_back: int - the number of previous data used for prediction
    :param sampling_ratio: float
    :return:
    """
    tf = np.array([True, False])

    # Initialize the processed matrix - shape = (lookback x OD)
    pred_tm = training_set[0:look_back, :]
    # The initialized labels matrix - All the initialized data points have been marked as label 1
    labels = np.ones((look_back, pred_tm.shape[1]))

    # The training will go through the training set along the timeslot axis from 0 to (T  - look_back)
    for ts in range(training_set.shape[0] - look_back):
        # Rnn input is the last block of data in processed matrix with shape = (look_back x OD)
        rnn_input = pred_tm[ts:(ts + look_back), :]
        trainX, trainY = create_xy_labeled(raw_data=training_set[ts:(ts + look_back + 1), :],
                                           data=rnn_input,
                                           look_back=look_back,
                                           labels=labels[ts:(ts + look_back), :])

        rnn.model.fit(trainX,
                      trainY,
                      epochs=1,
                      batch_size=rnn.batch_size,
                      verbose=1)

        _pred = rnn.model.predict(trainX)
        _pred = np.reshape(_pred, (_pred.shape[0], _pred.shape[1]))

        sampling = np.expand_dims(
            np.random.choice(tf, size=training_set.shape[1], p=(sampling_ratio, 1 - sampling_ratio)), axis=0)

        inv_sampling = np.invert(sampling)
        pred_input = _pred.T[-1, :] * inv_sampling
        measured_input = training_set[ts + look_back, :] * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        pred_tm = np.concatenate([pred_tm, new_input], axis=0)

        label = np.ones((1, training_set.shape[1])) * sampling
        labels = np.concatenate([labels, label], axis=0)


def run_test_bidirect_rnn(raw_data, dataset_name='Abilene24s'):
    test_name = 'bidirectional'
    splitting_ratio = [0.7, 0.3]
    errors = np.empty((0, 3))
    a_lookback = range(26, 31, 1)

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
                      '/TrainX_' + test_name + '_' + dataset_name + '_' + str(
            splitting_ratio[0]) + '_look_back_%i.npy' % look_back
        trainY_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                      '/TrainY_' + test_name + '_' + dataset_name + '_' + str(
            splitting_ratio[0]) + '_look_back_%i.npy' % look_back
        trainX, trainY = [], []
        if not os.path.isfile(trainX_path):
            print('--- Can not find dataset at: ' + trainX_path)
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
            rnn.bidirectional_model_construction(input_shape=(trainX.shape[1], trainX.shape[2]), output_dim=look_back)
            history = rnn.model.fit(trainX,
                                    trainY,
                                    epochs=rnn.n_epoch,
                                    batch_size=rnn.batch_size,
                                    validation_split=0.1,
                                    callbacks=rnn.callbacks_list,
                                    verbose=1)
            rnn.save_model_to_disk()
            # rnn.plot_model_history(history)

        print(rnn.model.summary())

        errors = np.empty((0, 4))
        sampling_ratioes = [0.2, 0.3, 0.4, 0.5]

        figures_saving_path = '/home/anle/TM_estimation_figures/' + dataset_name \
                              + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, look_back)

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
                    filename='%s.png' % test_name,
                    saving_path=figures_saving_path)

    return


def bidirect_rnn_one_features(raw_data, dataset_name='Abilene24s'):
    test_name = 'bidirectional_one_feature'
    splitting_ratio = [0.7, 0.3]
    errors = np.empty((0, 3))
    a_lookback = range(288, 289, 1)

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
                      '/TrainX_' + test_name + '_' + dataset_name + '_' + str(
            splitting_ratio[0]) + '_look_back_%i.npy' % look_back
        trainY_path = '/home/anle/TM_estimation_dataset/' + dataset_name + \
                      '/TrainY_' + test_name + '_' + dataset_name + '_' + str(
            splitting_ratio[0]) + '_look_back_%i.npy' % look_back
        trainX, trainY = [], []
        if not os.path.isfile(trainX_path):
            print('--- Can not find dataset at: ' + trainX_path)
            print('--- Create TrainX, TrainY ---')
            trainX, trainY = create_xy_set_seq2seq(training_set, look_back)
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
            trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))

            print('--- Save TrainX, TrainY to %s ---' % trainX_path)
            np.save(trainX_path, trainX)
            np.save(trainY_path, trainY)
        else:
            print('--- Load TrainX, TrainY from file %s --- ' % trainX_path)
            trainX = np.load(trainX_path)
            trainY = np.load(trainY_path)

        print(trainX.shape)

        rnn = RNN(
            saving_path=model_recorded_path + 'hidden_%i_lookback_%i_epoch_%i/' % (HIDDEN_DIM, look_back, N_EPOCH),
            raw_data=raw_data,
            look_back=look_back,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            check_point=True)

        if os.path.isfile(path=rnn.saving_path + 'model.json'):
            rnn.load_model_from_disk(model_json_file='model.json',
                                     model_weight_file='model.h5')
        else:
            print('--- Compile model for %s %s --- ' % (test_name, rnn.saving_path))

            rnn.bidirectional_model_construction(input_shape=(trainX.shape[1], trainX.shape[2]), output_dim=look_back)

            from_epoch = rnn.load_model_from_check_point()
            if from_epoch > 0:
                rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                history = rnn.model.fit(trainX,
                                        trainY,
                                        epochs=rnn.n_epoch,
                                        initial_epoch=from_epoch,
                                        batch_size=rnn.batch_size,
                                        validation_split=0.1,
                                        callbacks=rnn.callbacks_list,
                                        verbose=1)
                rnn.plot_model_history(history)


            else:
                rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                history = rnn.model.fit(trainX,
                                        trainY,
                                        epochs=rnn.n_epoch,
                                        batch_size=rnn.batch_size,
                                        validation_split=0.1,
                                        callbacks=rnn.callbacks_list,
                                        verbose=1)
                rnn.plot_model_history(history)

            rnn.save_model_to_disk()

        print(rnn.model.summary())

        errors = np.empty((0, 4))
        sampling_ratioes = [0.2, 0.3, 0.4, 0.5]

        figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                              + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, look_back)

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
                    filename='%s.png' % test_name,
                    saving_path=figures_saving_path)

    return


def bidirect_rnn_label_features(raw_data, dataset_name='Abilene24s'):
    test_name = 'bidirect_rnn_label_features'
    splitting_ratio = [0.7, 0.3]
    errors = np.empty((0, 3))
    look_back = 26

    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'
    errors = np.empty((0, 4))
    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, look_back)
    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    sampling_ratioes = [0.2, 0.3, 0.4, 0.5]
    for sampling_ratio in sampling_ratioes:
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

        rnn = RNN(
            saving_path=model_recorded_path + 'hidden_%i_lookback_%i_epoch_%i_sampling_ratio_%.2f/' % (
                HIDDEN_DIM, look_back, N_EPOCH, sampling_ratio),
            raw_data=raw_data,
            look_back=look_back,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            check_point=False)

        if os.path.isfile(path=rnn.saving_path + 'model.json'):
            rnn.load_model_from_disk(model_json_file='model.json',
                                     model_weight_file='model.h5')
        else:
            print('--- Compile model for %s %s --- ' % (test_name, rnn.saving_path))

            rnn.seq_seq_tempotal_model_construction(n_timesteps=look_back, n_features=2, drop_out=0.2)

            from_epoch = rnn.load_model_from_check_point()
            if from_epoch > 0:
                rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                online_training(rnn=rnn,
                                training_set=training_set,
                                look_back=look_back,
                                sampling_ratio=sampling_ratio)


            else:
                rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                initial_training_data = training_set[0:look_back * 5, :]
                rnn = initialized_training_pharse(rnn=rnn, initial_training_data=initial_training_data,
                                                  look_back=look_back)
                for epoch in range(rnn.n_epoch):
                    online_training(rnn=rnn,
                                    training_set=training_set,
                                    look_back=look_back,
                                    sampling_ratio=sampling_ratio)
                    if epoch % 10 == 0:
                        rnn.save_model_to_disk(model_json_filename='model-%i.json' % epoch,
                                               model_weight_filename='model-%i.h5' % epoch)

            rnn.save_model_to_disk()

        print(rnn.model.summary())

        pred_tm, measured_matrix = predict_with_loss_labels_feature(testing_set,
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
        mean_abs_error_by_day = mean_absolute_errors_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)

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
                                   description=test_name + '_hidden_%i_sampling_%f' % (
                                       HIDDEN_DIM, sampling_ratio),
                                   measured_matrix=measured_matrix,
                                   saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/')

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
                filename='%s.png' % test_name,
                saving_path=figures_saving_path)

    return


def forward_backward_rnn_one_features(raw_data, dataset_name='Abilene24s'):
    test_name = 'forward_backward_rnn_labed_features'
    splitting_ratio = [0.7, 0.3]
    errors = np.empty((0, 3))
    look_back = 26
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'
    errors = np.empty((0, 4))
    sampling_ratioes = [0.2, 0.3, 0.4, 0.5]

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, look_back)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    for sampling_ratio in sampling_ratioes:
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
        train_path = '/home/anle/TM_estimation_dataset/' + dataset_name
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        trainX_forward_file_name = '/TrainX_forward_' + test_name + '_' + dataset_name + '_' + str(
            splitting_ratio[0]) + '_look_back_%i.npy' % look_back
        trainY_forward_file_name = '/TrainY_forward_' + test_name + '_' + dataset_name + '_' + str(
            splitting_ratio[0]) + '_look_back_%i.npy' % look_back

        trainX_backward_file_name = '/TrainX_backward_' + test_name + '_' + dataset_name + '_' + str(
            splitting_ratio[0]) + '_look_back_%i.npy' % look_back
        trainY_backward_file_name = '/TrainY_backward_' + test_name + '_' + dataset_name + '_' + str(
            splitting_ratio[0]) + '_look_back_%i.npy' % look_back

        if not os.path.isfile(train_path + trainX_forward_file_name):
            print('--- Can not find dataset at: ' + train_path + trainX_forward_file_name)
            print('--- Create TrainX, TrainY ---')
            trainX_forward, trainY_forward = create_xy_set_seq2seq(training_set, look_back)
            trainX_forward = np.reshape(trainX_forward, (trainX_forward.shape[0], trainX_forward.shape[1], 1))
            trainY_forward = np.reshape(trainY_forward, (trainY_forward.shape[0], trainY_forward.shape[1], 1))

            trainX_backward, trainY_backward = create_xy_set_backward(training_set, look_back)
            trainX_backward = np.reshape(trainX_backward, (trainX_backward.shape[0], trainX_backward.shape[1], 1))
            trainY_backward = np.reshape(trainY_backward, (trainY_backward.shape[0], trainY_backward.shape[1], 1))

            print('--- Save TrainX, TrainY to %s ---' % (train_path + trainX_forward_file_name))
            np.save((train_path + trainX_forward_file_name), trainX_forward)
            np.save((train_path + trainY_forward_file_name), trainY_forward)

            np.save((train_path + trainX_backward_file_name), trainX_backward)
            np.save((train_path + trainY_backward_file_name), trainY_backward)

        else:
            print('--- Load TrainX, TrainY from file %s --- ' % (train_path + trainX_forward_file_name))
            trainX_forward = np.load((train_path + trainX_forward_file_name))
            trainY_forward = np.load((train_path + trainY_forward_file_name))

            trainX_backward = np.load((train_path + trainX_backward_file_name))
            trainY_backward = np.load((train_path + trainY_backward_file_name))

        print(trainX_forward.shape)

        rnn_forward = RNN(
            saving_path=model_recorded_path + 'rnn_forward/hidden_%i_lookback_%i_epoch_%i_sampling_ratio_%.2f/' % (
                HIDDEN_DIM, look_back, N_EPOCH, sampling_ratio),
            raw_data=raw_data,
            look_back=look_back,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            check_point=True)

        rnn_backward = RNN(
            saving_path=model_recorded_path + 'rnn_backward/hidden_%i_lookback_%i_epoch_%i_sampling_ratio_%.2f/' % (
                HIDDEN_DIM, look_back, N_EPOCH, sampling_ratio),
            raw_data=raw_data,
            look_back=look_back,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            check_point=True)

        if os.path.isfile(path=rnn_forward.saving_path + 'model.json'):
            rnn_forward.load_model_from_disk(model_json_file='model.json',
                                             model_weight_file='model.h5')

            rnn_backward.load_model_from_disk(model_json_file='model.json',
                                              model_weight_file='model.h5')


        else:
            print('--- Compile model for test_inputshape %s --- ' % (rnn_forward.saving_path))

            rnn_forward.bidirectional_model_construction(input_shape=(trainX_forward.shape[1], trainX_forward.shape[2]))

            from_epoch = rnn_forward.load_model_from_check_point()
            if from_epoch > 0:
                rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                forward_history = rnn_forward.model.fit(trainX_forward,
                                                        trainY_forward,
                                                        epochs=rnn_forward.n_epoch,
                                                        initial_epoch=from_epoch,
                                                        batch_size=rnn_forward.batch_size,
                                                        validation_split=0.1,
                                                        callbacks=rnn_forward.callbacks_list,
                                                        verbose=1)
                rnn_forward.save_model_to_disk()

            else:

                rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                forward_history = rnn_forward.model.fit(trainX_forward,
                                                        trainY_forward,
                                                        epochs=rnn_forward.n_epoch,
                                                        batch_size=rnn_forward.batch_size,
                                                        validation_split=0.1,
                                                        callbacks=rnn_forward.callbacks_list,
                                                        verbose=1)
                rnn_forward.save_model_to_disk()

            rnn_backward.bidirectional_model_construction(
                input_shape=(trainX_backward.shape[1], trainX_backward.shape[2]))
            from_epoch_backward = rnn_backward.load_model_from_check_point()
            if from_epoch_backward > 0:
                rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                backward_history = rnn_backward.model.fit(trainX_backward,
                                                          trainY_backward,
                                                          epochs=rnn_backward.n_epoch,
                                                          initial_epoch=from_epoch_backward,
                                                          batch_size=rnn_backward.batch_size,
                                                          validation_split=0.1,
                                                          callbacks=rnn_backward.callbacks_list,
                                                          verbose=1)
                rnn_backward.save_model_to_disk()
            else:
                rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                backward_history = rnn_backward.model.fit(trainX_backward,
                                                          trainY_backward,
                                                          epochs=rnn_backward.n_epoch,
                                                          batch_size=rnn_backward.batch_size,
                                                          validation_split=0.1,
                                                          callbacks=rnn_backward.callbacks_list,
                                                          verbose=1)
                rnn_backward.save_model_to_disk()

            rnn_forward.plot_model_history(forward_history)
            rnn_backward.plot_model_history(backward_history)

        print(rnn_forward.model.summary())
        print(rnn_backward.model.summary())

        pred_tm, measured_matrix = predict_with_loss_forward_backward_rnn(test_set=testing_set,
                                                                          look_back=look_back,
                                                                          forward_model=rnn_forward.model,
                                                                          backward_model=rnn_backward.model,
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
        mean_abs_error_by_day = mean_absolute_errors_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)

        rmse_by_day = root_means_squared_error_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)

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
                                   saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/')

        print('--- Sampling ratio: %.2f - Means abs errors by day ---' % sampling_ratio)
        print(mean_abs_error_by_day)
        print('--- Sampling ratio: %.2f - RMSE by day ---' % sampling_ratio)
        print(rmse_by_day)

        plt.title('Means abs errors by day\nSampling: %.2f' % sampling_ratio)
        plt.plot(range(len(mean_abs_error_by_day)), mean_abs_error_by_day)
        plt.xlabel('Day')
        plt.savefig(figures_saving_path + 'Means_abs_errors_by_day_sampling_%.2f.png' % sampling_ratio)
        plt.close()

        plt.title('RMSE by day\nSampling: %.2f' % sampling_ratio)
        plt.plot(range(len(rmse_by_day)), rmse_by_day)
        plt.xlabel('Day')
        plt.savefig(figures_saving_path + 'RMSE_by_day_sampling_%.2f.png' % sampling_ratio)
        plt.close()

    plot_errors(x_axis=sampling_ratioes,
                xlabel='sampling_ratio',
                errors=errors,
                filename='%s.png' % test_name,
                saving_path=figures_saving_path)

    return


def rnn_labeled_feature(raw_data, dataset_name="Abilene24s"):
    test_name = 'rnn_labeled_feature'
    splitting_ratio = [0.7, 0.3]
    look_back = 26
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    errors = np.empty((0, 4))
    sampling_ratioes = [0.3]

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, look_back)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    for sampling_ratio in sampling_ratioes:
        train_set, test_set = prepare_train_test_set(data=raw_data,
                                                     sampling_itvl=5,
                                                     splitting_ratio=splitting_ratio)

        seperated_train_set, centers_train_set = mean_std_flows_clustering(train_set)
        training_set, train_scalers, train_cluster_lens = different_flows_scaling(seperated_train_set[1:],
                                                                                  centers_train_set[1:])

        seperated_test_set, centers_test_set = mean_std_flows_clustering(test_set)
        testing_set, test_scalers, test_cluster_lens = different_flows_scaling(seperated_test_set[1:],
                                                                               centers_test_set[1:])

        rnn = RNN(
            saving_path=model_recorded_path + 'rnn_labeled/hidden_%i_lookback_%i_sampling_ratio_%.2f/' % (
                HIDDEN_DIM, look_back, sampling_ratio),
            raw_data=raw_data,
            look_back=look_back,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            check_point=False)

        if os.path.isfile(path=rnn.saving_path + 'model.json'):
            rnn.load_model_from_disk(model_json_file='model.json',
                                             model_weight_file='model.h5')

        else:
            print('[%s]---Compile model. Saving path %s --- ' % (test_name, rnn.saving_path))

            rnn.bidirectional_model_construction(input_shape=(look_back, 2))

            from_epoch = rnn.load_model_from_check_point()
            if from_epoch > 0:
                rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                print('[%s]--- Continue training forward model from epoch %i --- ' % (test_name, from_epoch))
                for epoch in range(from_epoch, rnn.n_epoch + 1, 1):
                    online_training(rnn=rnn,
                                    training_set=training_set,
                                    look_back=look_back,
                                    sampling_ratio=sampling_ratio)
                    rnn.save_model_to_disk(model_json_filename='model-%i-.json' % epoch,
                                                   model_weight_filename='model-%i-.h5' % epoch)
                    rnn.save_model_to_disk()

            else:

                rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                initial_training_data = training_set[0:look_back * 5, :]
                rnn = initialized_training_pharse(rnn=rnn, initial_training_data=initial_training_data,
                                                          look_back=look_back)
                for epoch in range(1, rnn.n_epoch + 1, 1):
                    online_training(rnn=rnn,
                                    training_set=training_set,
                                    look_back=look_back,
                                    sampling_ratio=sampling_ratio)
                    rnn.save_model_to_disk(model_json_filename='model-%i-.json' % epoch,
                                                   model_weight_filename='model-%i-.h5' % epoch)
                    rnn.save_model_to_disk()

        print(rnn.model.summary())

        pred_tm, measured_matrix = predict_with_loss_forward_backward_labeled(test_set=testing_set,
                                                                              look_back=look_back,
                                                                              forward_model=rnn_forward.model,
                                                                              backward_model=rnn_backward.model,
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
        mean_abs_error_by_day = mean_absolute_errors_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)

        rmse_by_day = root_means_squared_error_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)

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
        #                               description=test_name + '_sampling_%f' % sampling_ratio,
        #                               saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/',
        #                               ts_plot=288*3)
        #
        # visualize_retsult_by_flows(y_true=ytrue,
        #                            y_pred=pred_tm,
        #                            sampling_itvl=5,
        #                            description=test_name + '_sampling_%f' % sampling_ratio,
        #                            measured_matrix=measured_matrix,
        #                            saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/',
        #                            visualized_day=-1)

        print('--- Sampling ratio: %.2f - Means abs errors by day ---' % sampling_ratio)
        print(mean_abs_error_by_day)
        print('--- Sampling ratio: %.2f - RMSE by day ---' % sampling_ratio)
        print(rmse_by_day)
        print('--- Sampling ratio: %.2f - Error ratio by day ---' % sampling_ratio)
        print(errors_by_day)

        plt.title('Means abs errors by day\nSampling: %.2f' % sampling_ratio)
        plt.plot(range(len(mean_abs_error_by_day)), mean_abs_error_by_day)
        plt.xlabel('Day')
        plt.savefig(figures_saving_path + 'Means_abs_errors_by_day_sampling_%.2f.png' % sampling_ratio)
        plt.close()

        plt.title('RMSE by day\nSampling: %.2f' % sampling_ratio)
        plt.plot(range(len(rmse_by_day)), rmse_by_day)
        plt.xlabel('Day')
        plt.savefig(figures_saving_path + 'RMSE_by_day_sampling_%.2f.png' % sampling_ratio)
        plt.close()
        print('ERROR of testing at %.2f sampling' % sampling_ratio)
        print(errors)

    plot_errors(x_axis=sampling_ratioes,
                xlabel='sampling_ratio',
                errors=errors,
                filename='%s.png' % test_name,
                saving_path=figures_saving_path)


def forward_backward_rnn_labeled_features(raw_data, dataset_name='Abilene24s'):
    test_name = 'forward_backward_rnn_labeled_features'
    splitting_ratio = [0.7, 0.3]
    look_back = 26
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    errors = np.empty((0, 4))
    sampling_ratioes = [0.3]

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, look_back)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    for sampling_ratio in sampling_ratioes:
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

        rnn_forward = RNN(
            saving_path=model_recorded_path + 'rnn_forward/hidden_%i_lookback_%i_sampling_ratio_%.2f/' % (
                HIDDEN_DIM, look_back, sampling_ratio),
            raw_data=raw_data,
            look_back=look_back,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            check_point=False)

        rnn_backward = RNN(
            saving_path=model_recorded_path + 'rnn_backward/hidden_%i_lookback_%i_sampling_ratio_%.2f/' % (
                HIDDEN_DIM, look_back, sampling_ratio),
            raw_data=raw_data,
            look_back=look_back,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            check_point=False)

        if os.path.isfile(path=rnn_forward.saving_path + 'model.json'):
            rnn_forward.load_model_from_disk(model_json_file='model.json',
                                             model_weight_file='model.h5')

        else:
            print('[%s]---Compile model. Saving path %s --- ' % (test_name, rnn_forward.saving_path))

            rnn_forward.bidirectional_model_construction(input_shape=(look_back, 2))

            from_epoch = rnn_forward.load_model_from_check_point()
            if from_epoch > 0:
                rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                print('[%s]--- Continue training forward model from epoch %i --- ' % (test_name, from_epoch))
                for epoch in range(from_epoch, rnn_forward.n_epoch + 1, 1):
                    online_training(rnn=rnn_forward,
                                    training_set=training_set,
                                    look_back=look_back,
                                    sampling_ratio=sampling_ratio)
                    rnn_forward.save_model_to_disk(model_json_filename='model-%i-.json' % epoch,
                                                   model_weight_filename='model-%i-.h5' % epoch)
                rnn_forward.save_model_to_disk()

            else:

                rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                initial_training_data = training_set[0:look_back * 5, :]
                rnn_forward = initialized_training_pharse(rnn=rnn_forward, initial_training_data=initial_training_data,
                                                          look_back=look_back)
                for epoch in range(1, rnn_forward.n_epoch + 1, 1):
                    online_training(rnn=rnn_forward,
                                    training_set=training_set,
                                    look_back=look_back,
                                    sampling_ratio=sampling_ratio)
                    rnn_forward.save_model_to_disk(model_json_filename='model-%i-.json' % epoch,
                                                   model_weight_filename='model-%i-.h5' % epoch)
                rnn_forward.save_model_to_disk()

        if os.path.isfile(path=rnn_backward.saving_path + 'model.json'):
            rnn_backward.load_model_from_disk(model_json_file='model.json',
                                              model_weight_file='model.h5')

        else:
            print('[%s]---Compile model. Saving path %s --- ' % (test_name, rnn_backward.saving_path))

            # Flip the training_set for the backward training process
            # Then using the same initial_training_data and online_training functions as the forwarding process
            training_set_backward = np.flip(training_set, axis=0)
            rnn_backward.bidirectional_model_construction(input_shape=(look_back, 2))
            from_epoch_backward = rnn_backward.load_model_from_check_point()
            if from_epoch_backward > 0:

                rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

                for epoch in range(from_epoch_backward, rnn_backward.n_epoch + 1, 1):
                    print('[%s]--- Continue training backward model from epoch %i --- ' % (
                        test_name, from_epoch_backward))

                    online_training(rnn=rnn_backward,
                                    training_set=training_set_backward,
                                    look_back=look_back,
                                    sampling_ratio=sampling_ratio)
                    rnn_backward.save_model_to_disk(model_json_filename='model-%i-.json' % epoch,
                                                    model_weight_filename='model-%i-.h5' % epoch)
                rnn_backward.save_model_to_disk()


            else:
                rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                initial_training_data = training_set_backward[0:look_back * 5, :]
                rnn_backward = initialized_training_pharse(rnn=rnn_backward,
                                                           initial_training_data=initial_training_data,
                                                           look_back=look_back)
                for epoch in range(1, rnn_backward.n_epoch + 1, 1):
                    online_training(rnn=rnn_backward,
                                    training_set=training_set_backward,
                                    look_back=look_back,
                                    sampling_ratio=sampling_ratio)
                    rnn_backward.save_model_to_disk(model_json_filename='model-%i-.json' % epoch,
                                                    model_weight_filename='model-%i-.h5' % epoch)
                rnn_backward.save_model_to_disk()

        print(rnn_forward.model.summary())
        print(rnn_backward.model.summary())

        pred_tm, measured_matrix = predict_with_loss_forward_backward_labeled(test_set=testing_set,
                                                                              look_back=look_back,
                                                                              forward_model=rnn_forward.model,
                                                                              backward_model=rnn_backward.model,
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
        mean_abs_error_by_day = mean_absolute_errors_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)

        rmse_by_day = root_means_squared_error_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)

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
        #                               description=test_name + '_sampling_%f' % sampling_ratio,
        #                               saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/',
        #                               ts_plot=288*3)
        #
        # visualize_retsult_by_flows(y_true=ytrue,
        #                            y_pred=pred_tm,
        #                            sampling_itvl=5,
        #                            description=test_name + '_sampling_%f' % sampling_ratio,
        #                            measured_matrix=measured_matrix,
        #                            saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/',
        #                            visualized_day=-1)

        print('--- Sampling ratio: %.2f - Means abs errors by day ---' % sampling_ratio)
        print(mean_abs_error_by_day)
        print('--- Sampling ratio: %.2f - RMSE by day ---' % sampling_ratio)
        print(rmse_by_day)
        print('--- Sampling ratio: %.2f - Error ratio by day ---' % sampling_ratio)
        print(errors_by_day)

        plt.title('Means abs errors by day\nSampling: %.2f' % sampling_ratio)
        plt.plot(range(len(mean_abs_error_by_day)), mean_abs_error_by_day)
        plt.xlabel('Day')
        plt.savefig(figures_saving_path + 'Means_abs_errors_by_day_sampling_%.2f.png' % sampling_ratio)
        plt.close()

        plt.title('RMSE by day\nSampling: %.2f' % sampling_ratio)
        plt.plot(range(len(rmse_by_day)), rmse_by_day)
        plt.xlabel('Day')
        plt.savefig(figures_saving_path + 'RMSE_by_day_sampling_%.2f.png' % sampling_ratio)
        plt.close()
        print('ERROR of testing at %.2f sampling' % sampling_ratio)
        print(errors)

    plot_errors(x_axis=sampling_ratioes,
                xlabel='sampling_ratio',
                errors=errors,
                filename='%s.png' % test_name,
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


def update_predicted_data(pred_X, pred_tm, current_ts, look_back, measured_matrix):
    sampling_measured_matrix = measured_matrix[(current_ts + 1):(current_ts + look_back), :]
    inv_sampling_measured_matrix = np.invert(sampling_measured_matrix)
    bidirect_rnn_pred_value = pred_X[0:-1, :] * inv_sampling_measured_matrix

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

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :].T
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - look_back, 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input = ret_tm[:, tslot:]
        rnn_input = np.reshape(rnn_input, (rnn_input.shape[0], rnn_input.shape[1], 1))

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


def predict_with_loss_labels_feature(test_set, look_back, rnn_model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """
    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :]
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.ones((look_back, test_set.shape[1]), dtype=bool)
    labels = np.ones((look_back, test_set.shape[1]))

    # Predict the TM from time slot look_back
    for ts in range(0, test_set.shape[0] - look_back, 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input = ret_tm[ts:(ts + look_back), :]
        trainX, trainY = create_xy_labeled(raw_data=test_set[ts:(ts + look_back + 1), :],
                                           data=rnn_input,
                                           look_back=look_back,
                                           labels=labels[ts:(ts + look_back), :])

        # Get the TM prediction of next time slot
        predictX = rnn_model.predict(trainX)
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
        measured_input = test_set[ts + look_back, :] * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=0)

        label = np.ones((1, test_set.shape[1])) * sampling
        labels = np.concatenate([labels, label], axis=0)

    return ret_tm, measured_matrix


def calculate_updated_weights(current_ts, look_back, measured_block, weight_rnn_input=0.2):
    """
    Calculate the weights for the updating rnn input based on the number of measured data used for predicting
    In this function, all data have shape = (od x timeslot)
    :param current_ts:
    :param look_back:
    :param measured_matrix: array-like shape = (od x timeslot)
    :return:
    """

    labels = measured_block.astype(int)

    forward_weight_matrix = np.empty((measured_block.shape[0], 0))
    backward_weight_matrix = np.empty((measured_block.shape[0], 0))
    input_weight_matrix = np.empty((measured_block.shape[0], 0))
    for input_idx in range(look_back):
        _left = labels[:, 0:input_idx]
        _right = labels[:, (input_idx + 1):]
        _count_measured_left = np.expand_dims(np.count_nonzero(_left, axis=1), axis=1).astype(dtype='float64')
        _count_measured_right = np.expand_dims(np.count_nonzero(_right, axis=1), axis=1).astype(dtype='float64')

        _count_sum = _count_measured_left + _count_measured_right
        _count_sum[_count_sum == 0] = 1

        _proba_forward = (_count_measured_left / (_count_sum)) * (
                1 - weight_rnn_input)
        _proba_backward = (_count_measured_right / (_count_sum)) * (
                1 - weight_rnn_input)

        _proba_forward[_proba_forward == 0] = 0.1
        _proba_backward[_proba_backward == 0] = 0.1

        _proba_input = 1 - _proba_forward - _proba_backward

        forward_weight_matrix = np.concatenate([forward_weight_matrix, _proba_forward], axis=1)
        backward_weight_matrix = np.concatenate([backward_weight_matrix, _proba_backward], axis=1)
        input_weight_matrix = np.concatenate([input_weight_matrix, _proba_input], axis=1)

    return forward_weight_matrix, backward_weight_matrix, input_weight_matrix


def updating_historical_data(pred_tm, pred_forward, pred_backward, rnn_input, current_ts, look_back, measured_matrix):
    measured_block = measured_matrix.T[:, current_ts:(current_ts + look_back)]

    forward_weights, backward_weights, input_weights = calculate_updated_weights(current_ts=current_ts,
                                                                                 look_back=look_back,
                                                                                 measured_block=measured_block,
                                                                                 weight_rnn_input=0.3)
    forward_weights = forward_weights[:, 1:-1]
    backward_weights = backward_weights[:, 1:-1]
    input_weights = input_weights[:, 1:-1]

    considered_forward = pred_forward[:, 0:-2]
    considered_backward = np.flip(pred_backward[:, 0:-2], axis=1)
    considered_rnn_input = rnn_input[:, 1:-1]
    considered_rnn_input = np.reshape(considered_rnn_input,
                                      (considered_rnn_input.shape[0], considered_rnn_input.shape[1]))

    # if current_ts > 5:

    # plt.title('Checking Update Bidirectional RNN of flow %i at ts %i' % (4, current_ts))
    # plt.plot(considered_forward[4,:], label='Forward-RNN')
    # plt.plot(considered_backward[4,:], label='Backward-RNN')
    # plt.plot(considered_rnn_input[4, :], label = 'Input')
    # plt.plot(pred_tm[4, (current_ts + 1):(current_ts + look_back - 1)], label='Raw data')
    # plt.plot()
    # plt.legend()
    # # plt.savefig(HOME+'/TM_estimation_figures/Abilene24s/CheckingFBRNN/Timeslot_%i.png'%current_ts)
    # plt.show()
    # plt.close()

    updated_rnn_input = considered_backward * forward_weights + \
                        considered_forward * backward_weights + \
                        considered_rnn_input * input_weights

    sampling_measured_matrix = measured_matrix.T[:, (current_ts + 1):(current_ts + look_back - 1)]
    inv_sampling_measured_matrix = np.invert(sampling_measured_matrix)
    bidirect_rnn_pred_value = updated_rnn_input * inv_sampling_measured_matrix

    pred_tm[:, (current_ts + 1):(current_ts + look_back - 1)] = \
        pred_tm[:, (current_ts + 1):(current_ts + look_back - 1)] * sampling_measured_matrix + bidirect_rnn_pred_value

    return pred_tm.T


def predict_with_loss_forward_backward_rnn(test_set, look_back, forward_model, backward_model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :].T
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - look_back, 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input = ret_tm[:, tslot:]
        rnn_input_backward = np.flip(rnn_input, axis=0)

        rnn_input = np.reshape(rnn_input, (rnn_input.shape[0], rnn_input.shape[1], 1))
        rnn_input_backward = np.reshape(rnn_input_backward,
                                        (rnn_input_backward.shape[0], rnn_input_backward.shape[1], 1))

        # Get the TM prediction of next time slot
        forward_predictX = forward_model.predict(rnn_input)
        forward_predictX = np.reshape(forward_predictX, (forward_predictX.shape[0], forward_predictX.shape[1]))

        backward_predictX = backward_model.predict(rnn_input_backward)
        backward_predictX = np.reshape(backward_predictX, (backward_predictX.shape[0], backward_predictX.shape[1]))

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

        # todo: update historical data
        ret_tm = updating_historical_data(pred_tm=ret_tm,
                                          pred_forward=forward_predictX,
                                          pred_backward=backward_predictX,
                                          rnn_input=rnn_input,
                                          current_ts=tslot,
                                          look_back=look_back,
                                          measured_matrix=measured_matrix)

        # ret_tm = update_predicted_data(pred_X=predictX.T, pred_tm=ret_tm.T, current_ts=tslot, look_back=look_back,
        #                                measured_matrix=measured_matrix)

        pred_input = forward_predictX.T[-1, :] * inv_sampling
        measured_input = test_set[tslot + look_back, :] * sampling

        # Update the predicted historical data

        # Merge value from pred_input and measured_input
        new_input = pred_input.T + measured_input.T
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=1)

    return ret_tm.T, measured_matrix


def predict_with_loss_forward_backward_labeled(test_set, look_back, forward_model, backward_model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :]  # rnn input shape = (timeslot x OD)
    # Results TM
    ret_tm = rnn_input  # ret_rm shape = (time slot x OD)
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.ones((look_back, test_set.shape[1]), dtype=bool)
    labels = np.ones((look_back, test_set.shape[1]))

    # Predict the TM from time slot look_back
    for ts in range(0, test_set.shape[0] - look_back, 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input_forward = ret_tm[ts:(ts + look_back), :]

        testX_forward, testY_forward = create_xy_labeled(raw_data=test_set[ts:(ts + look_back + 1), :],
                                                         data=rnn_input_forward,
                                                         look_back=look_back,
                                                         labels=labels[ts:(ts + look_back), :])

        rnn_input_backward = np.flip(ret_tm[ts:(ts + look_back), :], axis=0)
        testX_backward, testY_backward = create_xy_labeled(raw_data=test_set[ts:(ts + look_back + 1), :],
                                                           data=rnn_input_backward,
                                                           look_back=look_back,
                                                           labels=np.flip(labels[ts:(ts + look_back), :], axis=1))

        # Get the TM prediction of next time slot
        forward_predictX = forward_model.predict(testX_forward)
        forward_predictX = np.reshape(forward_predictX, (forward_predictX.shape[0], forward_predictX.shape[1]))

        backward_predictX = backward_model.predict(testX_backward)
        backward_predictX = np.reshape(backward_predictX, (backward_predictX.shape[0], backward_predictX.shape[1]))

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        ret_tm = updating_historical_data(pred_tm=ret_tm.T,
                                          pred_forward=forward_predictX,
                                          pred_backward=backward_predictX,
                                          rnn_input=rnn_input_forward.T,
                                          current_ts=ts,
                                          look_back=look_back,
                                          measured_matrix=measured_matrix)

        # boolean array(1 x n_flows):for choosing value from predicted data
        # sampling = np.expand_dims(np.random.choice(tf,
        #                                            size=(test_set.shape[1]),
        #                                            p=[sampling_ratio, 1 - sampling_ratio]), axis=0)

        sampling = set_measured_flow(rnn_input=rnn_input_forward.T,
                                     forward_pred=forward_predictX,
                                     backward_pred=np.flip(backward_predictX, axis=1),
                                     measured_matrix=measured_matrix[ts:(ts + look_back)].T,
                                     sampling_ratio=sampling_ratio)

        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        # ret_tm = update_predicted_data(pred_X=predictX.T, pred_tm=ret_tm.T, current_ts=tslot, look_back=look_back,
        #                                measured_matrix=measured_matrix)

        pred_input = forward_predictX.T[-1, :] * inv_sampling
        measured_input = test_set[ts + look_back, :] * sampling

        # Update the predicted historical data

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=0)
        label = np.ones((1, test_set.shape[1])) * sampling
        labels = np.concatenate([labels, label], axis=0)

    return ret_tm, measured_matrix


if __name__ == "__main__":
    np.random.seed(10)

    Abilene24s_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24s.csv')
    # Abilene_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene.csv')
    # Abilene1_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene1.csv')
    # Abilene3_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene3.csv')
    Abilene24s_data = remove_zero_flow(Abilene24s_data, eps=1)
    # forward_backward_rnn_one_features(raw_data=Abilene24s_data, dataset_name='Abilene24s')
    # bidirect_rnn_one_features(raw_data=Abilene24s_data, dataset_name='Abilene24s')
    # bidirect_rnn_label_features(raw_data=Abilene24s_data, dataset_name='Abilene24s')
    forward_backward_rnn_labeled_features(raw_data=Abilene24s_data, dataset_name='Abilene24s')
