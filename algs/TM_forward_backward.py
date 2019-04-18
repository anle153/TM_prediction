import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from Models.RNN_LSTM import *
from common.DataHelper import *
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
# RNN CONFIGURATION
INPUT_DIM = 100
HIDDEN_DIM_FW = 100
HIDDEN_DIM_BW = 100
N_EPOCH_FW = 50
N_EPOCH_BW = 50
BATCH_SIZE = 512

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


def set_measured_flow(rnn_input, forward_pred, backward_pred, measured_matrix, sampling_ratio, hyperparams):
    """
    :param rnn_input:
    :param forward_pred:
    :param backward_pred:
    :param updated_rnn:
    :param measured_matrix: shape = (od x timeslot)
    :return:
    """

    w = calculate_measured_weights(rnn_input=rnn_input,
                                   forward_pred=forward_pred,
                                   backward_pred=backward_pred,
                                   measured_matrix=measured_matrix,
                                   hyperparams=hyperparams)

    sampling = np.zeros(shape=(rnn_input.shape[0]))
    m = int(sampling_ratio * rnn_input.shape[0])
    sorted_idx_w = np.argpartition(w, m)
    sampling[sorted_idx_w[:m]] = 1

    sampling = np.expand_dims(sampling, axis=0)

    return sampling.astype(bool)


def calculate_updated_weights(look_back, measured_block, forward_loss, backward_loss):
    """
    Calculate the weights for the updating rnn input based on the number of measured data used for predicting
    (The confident of the input, forward and backward data)
    In this function, all data have shape = (od x timeslot)
    :param weight_based_rl:
    :param adjust_loss:
    :param weight_rnn_input:
    :param current_ts:
    :param look_back:
    :param measured_block: array-like shape = (od x timeslot)
    :return: weight shape = (od x timeslot)
    """
    eps = 0.0001

    labels = measured_block.astype(int)

    measured_count = np.sum(labels, axis=1).astype(float)
    _eta = measured_count / look_back

    _eta[_eta == 0] = eps

    alpha = 1 - _eta
    alpha = np.tile(np.expand_dims(alpha, axis=1), (1, look_back))

    # Calculate p
    rho = np.empty((measured_block.shape[0], 0))
    mu = np.empty((measured_block.shape[0], 0))
    for j in range(0, look_back):
        _rho = np.expand_dims((np.sum(measured_block[:, j:], axis=1)) / float(look_back - j), axis=1)
        _mu = np.expand_dims((np.sum(measured_block[:, :(j + 1)], axis=1)) / float(j + 1), axis=1)
        rho = np.concatenate([rho, _rho], axis=1)
        mu = np.concatenate([mu, _mu], axis=1)

    forward_loss = np.tile(np.expand_dims(forward_loss, axis=1), (1, look_back))
    backward_loss = np.tile(np.expand_dims(backward_loss, axis=1), (1, look_back))

    beta = (backward_loss + mu) * (1 - alpha) / (forward_loss + backward_loss + mu + rho)

    gamma = (forward_loss + rho) * (1 - alpha) / (forward_loss + backward_loss + mu + rho)

    return alpha, beta, gamma


def calculate_forward_backward_loss(measured_block, pred_forward, pred_backward, rnn_input):
    """

    :param measured_block: shape = (od, lookback)
    :param pred_forward: shape = (od, lookback)
    :param pred_backward: shape = (od, lookback)
    :param rnn_input: shape = (od, lookback)
    :return: shape = (od, )
    """
    eps = 10e-8

    rnn_first_input_updated = np.expand_dims(pred_backward[:, 1], axis=1)
    rnn_last_input_updated = np.expand_dims(pred_forward[:, -2], axis=1)
    rnn_updated_input_forward = np.concatenate([rnn_first_input_updated, pred_forward[:, 0:-2], rnn_last_input_updated],
                                               axis=1)
    rnn_updated_input_backward = np.concatenate([rnn_first_input_updated, pred_backward[:, 2:], rnn_last_input_updated],
                                                axis=1)
    rl_forward = recovery_loss(rnn_input=rnn_input, rnn_updated=rnn_updated_input_forward,
                               measured_matrix=measured_block)
    rl_forward[rl_forward == 0] = eps

    rl_backward = recovery_loss(rnn_input=rnn_input, rnn_updated=rnn_updated_input_backward,
                                measured_matrix=measured_block)
    rl_backward[rl_backward == 0] = eps

    return rl_forward, rl_backward


def updating_historical_data(pred_tm, pred_forward, pred_backward, rnn_input, current_ts, look_back,
                             measured_matrix, raw_data):
    """

    :param pred_tm:    (timeslot x OD)
    :param pred_forward:
    :param pred_backward:
    :param rnn_input:
    :param current_ts:
    :param look_back:
    :param measured_matrix:
    :param raw_data:
    :return:
    """
    # Calculate loss before update
    _er = error_ratio(y_true=raw_data, y_pred=pred_tm[-look_back:, :], measured_matrix=measured_matrix[-look_back:, :])

    pred_tm = pred_tm.T

    # measure_block shape = (od x timeslot)
    measured_block = measured_matrix.T[:, current_ts:(current_ts + look_back)]

    forward_loss, backward_loss = calculate_forward_backward_loss(measured_block=measured_block,
                                                                  pred_forward=pred_forward,
                                                                  pred_backward=pred_backward,
                                                                  rnn_input=rnn_input)

    alpha, beta, gamma = calculate_updated_weights(look_back=look_back,
                                                   measured_block=measured_block,
                                                   forward_loss=forward_loss,
                                                   backward_loss=backward_loss)

    considered_forward = pred_forward[:, 0:-2]
    considered_backward = pred_backward[:, 2:]
    considered_rnn_input = rnn_input[:, 1:-1]

    alpha = alpha[:, 1:-1]
    beta = beta[:, 1:-1]
    gamma = gamma[:, 1:-1]

    updated_rnn_input = considered_backward * gamma + considered_forward * beta + considered_rnn_input * alpha

    sampling_measured_matrix = measured_matrix.T[:, (current_ts + 1):(current_ts + look_back - 1)]
    inv_sampling_measured_matrix = np.invert(sampling_measured_matrix)
    bidirect_rnn_pred_value = updated_rnn_input * inv_sampling_measured_matrix

    pred_tm[:, (current_ts + 1):(current_ts + look_back - 1)] = \
        pred_tm[:, (current_ts + 1):(current_ts + look_back - 1)] * sampling_measured_matrix + bidirect_rnn_pred_value

    # Calculate loss after update
    er_ = error_ratio(y_true=raw_data, y_pred=pred_tm.T[-look_back:, :],
                      measured_matrix=measured_matrix[-look_back:, :])

    # if er_ > _er:
    #     print('Correcting Fail')

    return pred_tm.T


def predict_forward_backward_rnn(test_set, n_timesteps, forward_model, backward_model, sampling_ratio,
                                 hyperparams=[2.71, 1, 4.83, 1.09]):
    """

    :param test_set: array-like, shape=(timeslot, od)
    :param look_back: int, default 26
    :param forward_model: rnn model for forward lstm
    :param backward_model: model for backward lstm
    :param sampling_ratio: sampling ratio at each time slot
    :param hyper_parameters:
    [adjust_loss, recovery_loss_weight, dfa_weight, forward_weight, backward_weight, consecutive_loss_weight]

    :return: prediction traffic matrix shape = (timeslot, od)
    """

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:n_timesteps, :]  # rnn input shape = (timeslot x OD)
    # Results TM
    ret_tm = rnn_input  # ret_rm shape = (time slot x OD)
    # The TF array for random choosing the measured flows
    measured_matrix = np.ones((n_timesteps, test_set.shape[1]), dtype=bool)
    labels = np.ones((n_timesteps, test_set.shape[1]))
    tf = np.array([True, False])

    day_size = 24 * (60 / 5)

    # Predict the TM from time slot look_back
    for ts in range(0, test_set.shape[0] - n_timesteps, 1):
        date = int(ts / day_size)
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input_forward = ret_tm[ts:(ts + n_timesteps), :]

        testX_forward, testY_forward = create_xy_labeled(raw_data=test_set[(ts + 1):(ts + n_timesteps + 1), :],
                                                         data=rnn_input_forward,
                                                         look_back=n_timesteps,
                                                         labels=labels[ts:(ts + n_timesteps), :])

        rnn_input_backward = np.flip(ret_tm[ts:(ts + n_timesteps), :], axis=0)
        test_set_backward = test_set[(ts - 1):(ts + n_timesteps - 1), :]
        test_set_backward_flipped = np.flip(test_set_backward, axis=0)

        testX_backward, testY_backward = create_xy_labeled(raw_data=test_set_backward_flipped,
                                                           data=rnn_input_backward,
                                                           look_back=n_timesteps,
                                                           labels=np.flip(labels[ts:(ts + n_timesteps), :], axis=0))

        # Get the TM prediction of next time slot
        forward_predictX = forward_model.predict(testX_forward)
        forward_predictX = np.reshape(forward_predictX, (forward_predictX.shape[0], forward_predictX.shape[1]))

        backward_predictX = backward_model.predict(testX_backward)
        backward_predictX = np.reshape(backward_predictX, (backward_predictX.shape[0], backward_predictX.shape[1]))

        backward_predictX = np.flip(backward_predictX, axis=1)

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        ret_tm = updating_historical_data(pred_tm=ret_tm,
                                          pred_forward=forward_predictX,
                                          pred_backward=backward_predictX,
                                          rnn_input=rnn_input_forward.T,
                                          current_ts=ts,
                                          look_back=n_timesteps,
                                          measured_matrix=measured_matrix,
                                          raw_data=test_set[ts:(ts + n_timesteps), :])

        # boolean array(1 x n_flows):for choosing value from predicted data
        # sampling = np.expand_dims(np.random.choice(tf,
        #                                            size=(test_set.shape[1]),
        #                                            p=[sampling_ratio, 1 - sampling_ratio]), axis=0)

        sampling = set_measured_flow(rnn_input=rnn_input_forward.T,
                                     forward_pred=forward_predictX,
                                     backward_pred=backward_predictX,
                                     measured_matrix=measured_matrix[ts:(ts + n_timesteps)].T,
                                     sampling_ratio=sampling_ratio,
                                     hyperparams=hyperparams)

        # For Consecutive loss
        # if (ts > (date * day_size + 120)) and ( ts <= (date * day_size + 120 + 12 * 4)):
        #     print('Consecutive_loss at date %i' %date)
        #     sampling = np.zeros(shape=(1, measured_matrix.shape[1]), dtype=bool)
        # else:
        #     sampling = set_measured_flow(rnn_input=rnn_input_forward.T,
        #                                  forward_pred=forward_predictX,
        #                                  backward_pred=backward_predictX,
        #                                  measured_matrix=measured_matrix[ts:(ts + look_back)].T,
        #                                  sampling_ratio=sampling_ratio,
        #                                  hyperparams=hyperparams)

        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        # ret_tm = update_predicted_data(pred_X=predictX.T, pred_tm=ret_tm.T, current_ts=tslot, look_back=look_back,
        #                                measured_matrix=measured_matrix)

        pred_input = forward_predictX.T[-1, :] * inv_sampling
        measured_input = test_set[ts + n_timesteps, :] * sampling

        # Update the predicted historical data

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=0)
        label = sampling.astype(int)
        labels = np.concatenate([labels, label], axis=0)

    return ret_tm, measured_matrix


# def predict_traffic(raw_data, dataset_name='Abilene24s', hyperparams=[]):
#     print('------ predict_traffic ------')
#     test_name = 'forward_backward_rnn_labeled_features'
#     splitting_ratio = [0.7, 0.3]
#     look_back = 26
#     model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'
#
#     errors = np.empty((0, 5))
#     sampling_ratio = 0.3
#
#     figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
#                           + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, look_back)
#
#     if not os.path.exists(figures_saving_path):
#         os.makedirs(figures_saving_path)
#
#     train_set, test_set = prepare_train_test_set(data=raw_data,
#                                                  sampling_itvl=5,
#                                                  splitting_ratio=splitting_ratio)
#
#     ################################################################################################################
#     #                                         For testing Flows Clustering                                         #
#
#     seperated_train_set, centers_train_set = mean_std_flows_clustering(train_set)
#     training_set, train_scalers, train_cluster_lens = different_flows_scaling(seperated_train_set[1:],
#                                                                               centers_train_set[1:])
#
#     seperated_test_set, centers_test_set = mean_std_flows_clustering(test_set)
#     testing_set, test_scalers, test_cluster_lens = different_flows_scaling(seperated_test_set[1:],
#                                                                            centers_test_set[1:])
#
#     copy_testing_set = np.copy(testing_set)
#     ################################################################################################################
#
#     rnn_forward = RNN(
#         saving_path=model_recorded_path + 'rnn_forward/hidden_%i_lookback_%i_sampling_ratio_%.2f/' % (
#             HIDDEN_DIM, look_back, sampling_ratio),
#         raw_data=raw_data,
#         look_back=look_back,
#         n_epoch=N_EPOCH,
#         batch_size=BATCH_SIZE,
#         hidden_dim=HIDDEN_DIM,
#         check_point=False)
#
#     rnn_backward = RNN(
#         saving_path=model_recorded_path + 'rnn_backward/hidden_%i_lookback_%i_sampling_ratio_%.2f/' % (
#             HIDDEN_DIM, look_back, sampling_ratio),
#         raw_data=raw_data,
#         look_back=look_back,
#         n_epoch=N_EPOCH,
#         batch_size=BATCH_SIZE,
#         hidden_dim=HIDDEN_DIM,
#         check_point=False)
#
#     list_weights_files_rnn_forward = fnmatch.filter(os.listdir(rnn_forward.saving_path), '*.h5')
#
#     list_weights_files_rnn_backward = fnmatch.filter(os.listdir(rnn_backward.saving_path), '*.h5')
#
#     if len(list_weights_files_rnn_forward) == 0 or len(list_weights_files_rnn_backward) == 0:
#         print('----> [RNN-load_model_from_check_point] --- Found no weights file at %s---' % rnn_forward.saving_path)
#         return -1
#
#     list_weights_files_rnn_forward = sorted(list_weights_files_rnn_forward, key=lambda x: int(x.split('-')[1]))
#     list_weights_files_rnn_backward = sorted(list_weights_files_rnn_backward, key=lambda x: int(x.split('-')[1]))
#
#     _max_epoch_rnn_forward = int(list_weights_files_rnn_forward[-1].split('-')[1])
#     _max_epoch_rnn_backward = int(list_weights_files_rnn_backward[-1].split('-')[1])
#
#     range_epoch = _max_epoch_rnn_forward if _max_epoch_rnn_forward < _max_epoch_rnn_backward else _max_epoch_rnn_backward
#
#     for epoch in range(1, range_epoch + 1, 1):
#         rnn_forward.load_model_from_check_point(_from_epoch=epoch)
#         rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#
#         rnn_backward.load_model_from_check_point(_from_epoch=epoch)
#         rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#
#         print(rnn_forward.model.summary())
#         print(rnn_backward.model.summary())
#
#         pred_tm, measured_matrix = predict_with_loss_forward_backward_labeled(test_set=testing_set,
#                                                                               look_back=look_back,
#                                                                               forward_model=rnn_forward.model,
#                                                                               backward_model=rnn_backward.model,
#                                                                               sampling_ratio=sampling_ratio,
#                                                                               hyperparams=hyperparams)
#         ############################################################################################################
#         #                                         For testing Flows Clustering
#
#         pred_tm = different_flows_invert_scaling(pred_tm, scalers=test_scalers, cluster_lens=test_cluster_lens)
#         pred_tm[pred_tm < 0] = 0
#         ytrue = different_flows_invert_scaling(data=testing_set, scalers=test_scalers,
#                                                cluster_lens=test_cluster_lens)
#         ############################################################################################################
#
#         errors_by_day = calculate_error_ratio_by_day(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix,
#                                                      sampling_itvl=5)
#         mean_abs_error_by_day = mean_absolute_errors_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)
#
#         rmse_by_day = root_means_squared_error_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)
#
#         y3 = ytrue.flatten()
#         y4 = pred_tm.flatten()
#         a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
#         a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
#         pred_confident = r2_score(y3, y4)
#
#         err_rat = error_ratio(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix)
#
#         error = np.expand_dims(np.array([epoch, a_nmae, a_nmse, pred_confident, err_rat]), axis=0)
#
#         errors = np.concatenate([errors, error], axis=0)
#
#         # visualize_results_by_timeslot(y_true=ytrue,
#         #                               y_pred=pred_tm,
#         #                               measured_matrix=measured_matrix,
#         #                               description=test_name + '_sampling_%f' % sampling_ratio,
#         #                               saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/',
#         #                               ts_plot=288*3)
#         #
#         # visualize_retsult_by_flows(y_true=ytrue,
#         #                            y_pred=pred_tm,
#         #                            sampling_itvl=5,
#         #                            description=test_name + '_sampling_%f' % sampling_ratio,
#         #                            measured_matrix=measured_matrix,
#         #                            saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/',
#         #                            visualized_day=-1)
#
#         print('--- Sampling ratio: %.2f - Means abs errors by day ---' % sampling_ratio)
#         print(mean_abs_error_by_day)
#         print('--- Sampling ratio: %.2f - RMSE by day ---' % sampling_ratio)
#         print(rmse_by_day)
#         print('--- Sampling ratio: %.2f - Error ratio by day ---' % sampling_ratio)
#         print(errors_by_day)
#
#         plt.title('Means abs errors by day\nSampling: %.2f' % sampling_ratio)
#         plt.plot(range(len(mean_abs_error_by_day)), mean_abs_error_by_day)
#         plt.xlabel('Day')
#         plt.savefig(figures_saving_path + 'Means_abs_errors_by_day_sampling_%.2f.png' % sampling_ratio)
#         plt.close()
#
#         plt.title('RMSE by day\nSampling: %.2f' % sampling_ratio)
#         plt.plot(range(len(rmse_by_day)), rmse_by_day)
#         plt.xlabel('Day')
#         plt.savefig(figures_saving_path + 'RMSE_by_day_sampling_%.2f.png' % sampling_ratio)
#         plt.close()
#         print('ERROR of testing at %.2f sampling' % sampling_ratio)
#         print(errors)
#
#     np.savetxt('./Errors/[%s][hidden_%i_lookback_%i_sampling_ratio_%.2f]Errors_by_epoch.csv' % (
#         test_name, HIDDEN_DIM, look_back, sampling_ratio), errors, delimiter=',')


# def try_hyper_parameter(raw_data, dataset_name='Abilene24s', hyperparams=[], errors=[]):
#     _best_epoch = 115
#     test_name = 'forward_backward_rnn_labeled_features'
#     splitting_ratio = [0.7, 0.3]
#     look_back = 26
#     model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'
#
#     sampling_ratio = 0.3
#
#     figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
#                           + '/' + test_name + '/test_hidden_%i_look_back_%i/' % (HIDDEN_DIM, look_back)
#
#     if not os.path.exists(figures_saving_path):
#         os.makedirs(figures_saving_path)
#
#     train_set, test_set = prepare_train_test_set(data=raw_data,
#                                                  sampling_itvl=5,
#                                                  splitting_ratio=splitting_ratio)
#
#     ################################################################################################################
#     #                                         For testing Flows Clustering                                         #
#
#     seperated_train_set, centers_train_set = mean_std_flows_clustering(train_set)
#     training_set, train_scalers, train_cluster_lens = different_flows_scaling(seperated_train_set[1:],
#                                                                               centers_train_set[1:])
#
#     seperated_test_set, centers_test_set = mean_std_flows_clustering(test_set)
#     testing_set, test_scalers, test_cluster_lens = different_flows_scaling(seperated_test_set[1:],
#                                                                            centers_test_set[1:])
#
#     ################################################################################################################
#
#     rnn_forward = RNN(
#         saving_path=model_recorded_path + 'rnn_forward/hidden_%i_lookback_%i_sampling_ratio_%.2f/' % (
#             HIDDEN_DIM, look_back, sampling_ratio),
#         raw_data=raw_data,
#         look_back=look_back,
#         n_epoch=N_EPOCH,
#         batch_size=BATCH_SIZE,
#         hidden_dim=HIDDEN_DIM,
#         check_point=False)
#
#     rnn_backward = RNN(
#         saving_path=model_recorded_path + 'rnn_backward/hidden_%i_lookback_%i_sampling_ratio_%.2f/' % (
#             HIDDEN_DIM, look_back, sampling_ratio),
#         raw_data=raw_data,
#         look_back=look_back,
#         n_epoch=N_EPOCH,
#         batch_size=BATCH_SIZE,
#         hidden_dim=HIDDEN_DIM,
#         check_point=False)
#
#     copy_testing_set = np.copy(testing_set)
#
#     print('Hyperparams:' + str(hyperparams))
#
#     rnn_forward.load_model_from_check_point(_from_epoch=_best_epoch)
#     rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#
#     rnn_backward.load_model_from_check_point(_from_epoch=_best_epoch)
#     rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#
#     print(rnn_forward.model.summary())
#     print(rnn_backward.model.summary())
#
#     pred_tm, measured_matrix = predict_with_loss_forward_backward_labeled(test_set=copy_testing_set,
#                                                                           look_back=look_back,
#                                                                           forward_model=rnn_forward.model,
#                                                                           backward_model=rnn_backward.model,
#                                                                           sampling_ratio=sampling_ratio,
#                                                                           hyperparams=hyperparams)
#     ############################################################################################################
#     #                                         For testing Flows Clustering
#
#     pred_tm = different_flows_invert_scaling(pred_tm, scalers=test_scalers, cluster_lens=test_cluster_lens)
#     pred_tm[pred_tm < 0] = 0
#     ytrue = different_flows_invert_scaling(data=copy_testing_set, scalers=test_scalers,
#                                            cluster_lens=test_cluster_lens)
#     ############################################################################################################
#
#     errors_by_day = calculate_error_ratio_by_day(y_true=ytrue, y_pred=pred_tm,
#                                                  measured_matrix=measured_matrix,
#                                                  sampling_itvl=5)
#     mean_abs_error_by_day = mean_absolute_errors_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)
#
#     rmse_by_day = root_means_squared_error_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)
#
#     y3 = ytrue.flatten()
#     y4 = pred_tm.flatten()
#     a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
#     a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
#     mae = mean_abs_error(y_true=y3, y_pred=y4)
#     pred_confident = r2_score(y3, y4)
#
#     err_rat = error_ratio(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix)
#
#     # errors format: [adjust_loss, dfa, forward, backward, consecutive, nmae, nmse, r2, er]
#
#     error = np.expand_dims(np.array(
#         [hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3], a_nmae, a_nmse, pred_confident,
#          err_rat, mae]), axis=0)
#
#     errors = np.concatenate([errors, error], axis=0)
#
#     # Saving results
#     results_path = './Results/%s/' % dataset_name
#     if not os.path.exists(results_path):
#         os.makedirs(results_path)
#
#     # np.savetxt(
#     #     results_path + '[%s][sampling_rate_%.2f][look_back_%.2f][Consecutive_Loss_4]Observation.csv' % (test_name,
#     #                                                                                 sampling_ratio,
#     #                                                                                 look_back),
#     #     ytrue, delimiter=',')
#     # np.savetxt(
#     #     results_path + '[%s][sampling_rate_%.2f][look_back_%.2f][Consecutive_Loss_4]Prediction.csv' % (test_name,
#     #                                                                                sampling_ratio,
#     #                                                                                look_back),
#     #     pred_tm, delimiter=',')
#     #
#     # np.savetxt(
#     #     results_path + '[%s][sampling_rate_%.2f][look_back_%.2f][Consecutive_Loss_4]MeasurementMatrix.csv' % (test_name,
#     #                                                                                       sampling_ratio,
#     #                                                                                       look_back),
#     #     measured_matrix, delimiter=',')
#
#     print('--- Sampling ratio: %.2f - Means abs errors by day ---' % sampling_ratio)
#     print(mean_abs_error_by_day)
#     print('--- Sampling ratio: %.2f - RMSE by day ---' % sampling_ratio)
#     print(rmse_by_day)
#     print('--- Sampling ratio: %.2f - Error ratio by day ---' % sampling_ratio)
#     print(errors_by_day)
#
#     plt.title('Means abs errors by day\nSampling: %.2f' % sampling_ratio)
#     plt.plot(range(len(mean_abs_error_by_day)), mean_abs_error_by_day)
#     plt.xlabel('Day')
#     plt.savefig(figures_saving_path + 'Means_abs_errors_by_day_sampling_%.2f.png' % sampling_ratio)
#     plt.close()
#
#     plt.title('RMSE by day\nSampling: %.2f' % sampling_ratio)
#     plt.plot(range(len(rmse_by_day)), rmse_by_day)
#     plt.xlabel('Day')
#     plt.savefig(figures_saving_path + 'RMSE_by_day_sampling_%.2f.png' % sampling_ratio)
#     plt.close()
#     print('ERROR of testing at %.2f sampling' % sampling_ratio)
#
#     return errors


def forward_backward_rnn(raw_data, dataset_name='Abilene24s', n_timesteps=26):
    test_name = 'forward_backward_rnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    sampling_ratioes = [0.3]
    random_eps = 1

    for sampling_ratio in sampling_ratioes:

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
                                                              8)

            np.save(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX.npy',
                trainX)
            np.save(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY.npy',
                trainY)
        else:

            print("|---  Load xy set from " + HOME + '/TM_estimation_dataset/' + dataset_name + '/' + dataset_name)

            trainX = np.load(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX.npy')
            trainY = np.load(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY.npy')

        print('|--- Creating XY backward set')
        training_set_backward = np.flip(training_set, axis=0)

        if not os.path.isfile(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX_backward.npy'):

            if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps):
                os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps)

            trainX_backward, trainY_backward = parallel_create_xy_set_by_random(training_set_backward, n_timesteps,
                                                                                sampling_ratio, random_eps,
                                                                                8)

            np.save(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX_backward.npy',
                trainX_backward)
            np.save(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY_backward.npy',
                trainY_backward)
        else:

            print(
                    "|---  Load xy backward set from " + HOME + '/TM_estimation_dataset/' + dataset_name + '/' + dataset_name)

            trainX_backward = np.load(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX_backward.npy')
            trainY_backward = np.load(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY_backward.npy')

        print("|--- Create RNN forward-backward model")

        forward_model_name = 'Sampling_%.2f_timesteps_%d/Forward_hidden_%i' % (sampling_ratio,
                                                                               n_timesteps,
                                                                               HIDDEN_DIM_FW)
        backward_model_name = 'Sampling_%.2f_timesteps_%d/Backward_hidden_%i' % (sampling_ratio,
                                                                                 n_timesteps,
                                                                                 HIDDEN_DIM_BW)

        model_name = 'FWBW_RNN_%s_%s' % (forward_model_name, backward_model_name)

        rnn_forward = RNN(
            saving_path=model_recorded_path + '%s/' % forward_model_name,
            raw_data=raw_data,
            look_back=n_timesteps,
            n_epoch=N_EPOCH_FW,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM_FW,
            check_point=True)

        rnn_backward = RNN(
            saving_path=model_recorded_path + '%s/' % backward_model_name,
            raw_data=raw_data,
            look_back=n_timesteps,
            n_epoch=N_EPOCH_BW,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM_BW,
            check_point=True)

        if os.path.isfile(path=rnn_forward.saving_path + 'weights-%i-0.00.hdf5' % N_EPOCH_FW):
            print('|--- Forward model exist!')
            rnn_forward.load_model_from_check_point(_from_epoch=N_EPOCH_FW, weights_file_type='hdf5')

        else:
            print('[%s]---Compile model. Saving path %s --- ' % (test_name, rnn_forward.saving_path))

            rnn_forward.seq2seq_model_construction(n_timesteps=n_timesteps, n_features=2)

            from_epoch = rnn_forward.load_model_from_check_point(weights_file_type='hdf5')
            if from_epoch > 0:
                rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam',
                                          metrics=['mse', 'mae', 'accuracy'])
                print('[%s]--- Continue training forward model from epoch %i --- ' % (test_name, from_epoch))

                forward_training_history = rnn_forward.model.fit(trainX,
                                                                 trainY,
                                                                 batch_size=BATCH_SIZE,
                                                                 initial_epoch=from_epoch,
                                                                 epochs=N_EPOCH_FW,
                                                                 validation_split=0.25,
                                                                 callbacks=rnn_forward.callbacks_list)

                rnn_forward.plot_model_metrics(forward_training_history,
                                               plot_prefix_name='Metrics')

            else:

                rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam',
                                          metrics=['mse', 'mae', 'accuracy'])

                forward_training_history = rnn_forward.model.fit(trainX,
                                                                 trainY,
                                                                 batch_size=BATCH_SIZE,
                                                                 epochs=N_EPOCH_FW,
                                                                 validation_split=0.25,
                                                                 callbacks=rnn_forward.callbacks_list)
                rnn_forward.plot_model_metrics(forward_training_history,
                                               plot_prefix_name='Metrics')

        # Training backward model
        if os.path.isfile(path=rnn_backward.saving_path + 'weights-%i-0.00.hdf5' % N_EPOCH_BW):
            print('|--- Forward model exist!')
            rnn_backward.load_model_from_check_point(_from_epoch=N_EPOCH_BW, weights_file_type='hdf5')

        else:
            print('[%s]---Compile model. Saving path %s --- ' % (test_name, rnn_backward.saving_path))

            rnn_backward.seq2seq_model_construction(n_timesteps=n_timesteps, n_features=2)
            from_epoch_backward = rnn_backward.load_model_from_check_point(weights_file_type='hdf5')
            if from_epoch_backward > 0:

                rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam',
                                           metrics=['mse', 'mae', 'accuracy'])

                print('[%s]--- Continue training backward model from epoch %i --- ' % (
                    test_name, from_epoch_backward))

                backward_training_history = rnn_backward.model.fit(trainX_backward,
                                                                   trainY_backward,
                                                                   batch_size=BATCH_SIZE,
                                                                   initial_epoch=from_epoch_backward,
                                                                   epochs=N_EPOCH_BW,
                                                                   validation_split=0.25,
                                                                   callbacks=rnn_backward.callbacks_list)
                rnn_backward.plot_model_metrics(backward_training_history,
                                                plot_prefix_name='Metrics')


            else:

                rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam',
                                           metrics=['mse', 'mae', 'accuracy'])

                backward_training_history = rnn_backward.model.fit(trainX_backward,
                                                                   trainY_backward,
                                                                   batch_size=BATCH_SIZE,
                                                                   epochs=N_EPOCH_BW,
                                                                   validation_split=0.25,
                                                                   callbacks=rnn_backward.callbacks_list)
                rnn_backward.plot_model_metrics(backward_training_history,
                                                plot_prefix_name='Metrics')

        print(rnn_forward.model.summary())
        print(rnn_backward.model.summary())

    return


def forward_backward_test_loop(test_set, testing_set, rnn_forward, rnn_backward,
                               epoch_fw, epoch_bw, n_timesteps, sampling_ratio,
                               std_train, mean_train, hyperparams, n_running_time=10):
    err_ratio_temp = []

    for running_time in range(n_running_time):
        print('|--- Epoch_fw %d - Epoch_bw %d  - Running time: %d' % (epoch_fw, epoch_bw, running_time))

        rnn_forward.load_model_from_check_point(_from_epoch=epoch_fw, weights_file_type='hdf5')
        rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

        rnn_backward.load_model_from_check_point(_from_epoch=epoch_bw, weights_file_type='hdf5')
        rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

        _testing_set = np.copy(testing_set)
        _test_set = np.copy(test_set)

        # print(cnn_brnn_model_forward.model.summary())
        # print(cnn_brnn_model_backward.model.summary())

        pred_tm, measured_matrix = predict_forward_backward_rnn(test_set=_testing_set,
                                                                n_timesteps=n_timesteps,
                                                                forward_model=rnn_forward.model,
                                                                backward_model=rnn_backward.model,
                                                                sampling_ratio=sampling_ratio,
                                                                hyperparams=hyperparams)

        pred_tm = pred_tm * std_train + mean_train

        err_ratio_temp.append(error_ratio(y_true=_test_set, y_pred=pred_tm, measured_matrix=measured_matrix))
        print('|--- error: %.3f' % error_ratio(y_true=_test_set, y_pred=pred_tm, measured_matrix=measured_matrix))

    return err_ratio_temp


def forward_backward_rnn_test(raw_data, n_timesteps, dataset_name,
                              hyperparams,
                              epoch_fw=0,
                              epoch_bw=0):
    print('------ predict_traffic ------')
    test_name = 'forward_backward_rnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    errors = np.empty((0, 2))
    sampling_ratio = 0.3

    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    copy_test_set = np.copy(test_set)

    testing_set = np.copy(test_set)

    mean_train = np.mean(train_set)
    std_train = np.std(train_set)
    testing_set = (testing_set - mean_train) / std_train

    copy_testing_set = np.copy(testing_set)

    print("|--- Create RNN forward-backward model")

    sampling_timesteps = 'Sampling_%.2f_timesteps_%d' % (sampling_ratio, n_timesteps)

    forward_model_name = 'Forward_hidden_%i' % (HIDDEN_DIM_FW)
    backward_model_name = 'Backward_hidden_%i' % (HIDDEN_DIM_BW)

    model_name = 'FWBW_RNN_%s_%s' % (forward_model_name, backward_model_name)

    rnn_forward = RNN(
        saving_path=model_recorded_path + '%s/%s/' % (sampling_timesteps,forward_model_name),
        raw_data=raw_data,
        look_back=n_timesteps,
        n_epoch=N_EPOCH_FW,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM_FW,
        check_point=True)

    rnn_backward = RNN(
        saving_path=model_recorded_path + '%s/%s/' % (sampling_timesteps, backward_model_name),
        raw_data=raw_data,
        look_back=n_timesteps,
        n_epoch=N_EPOCH_BW,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM_BW,
        check_point=True)

    rnn_forward.seq2seq_model_construction(n_timesteps=n_timesteps, n_features=2)
    rnn_backward.seq2seq_model_construction(n_timesteps=n_timesteps, n_features=2)


    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/' % \
                  (dataset_name, test_name, sampling_timesteps, model_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if epoch_fw != 0 and epoch_bw != 0:
        n_running_time = 1
        err_ratio_temp = forward_backward_test_loop(test_set=copy_test_set,
                                                    testing_set=copy_testing_set,
                                                    rnn_forward=rnn_forward,
                                                    rnn_backward=rnn_backward,
                                                    epoch_fw=epoch_fw,
                                                    epoch_bw=epoch_bw,
                                                    n_timesteps=n_timesteps,
                                                    sampling_ratio=sampling_ratio,
                                                    std_train=std_train,
                                                    mean_train=mean_train,
                                                    hyperparams=hyperparams,
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
        print('|--- Results have been saved at %s' %
              (result_path + 'Epoch_fw_%i_Epoch_bw_%i_n_running_time_%i.csv' %
               (epoch_fw, epoch_bw, n_running_time)))
        np.savetxt(fname=result_path + 'Epoch_fw_%i_Epoch_bw_%i_n_running_time_%i.csv' %
                         (epoch_fw, epoch_bw, n_running_time),
                   X=results, delimiter=',')
    else:

        list_weights_files_rnn_forward = fnmatch.filter(os.listdir(rnn_forward.saving_path), '*.hdf5')

        list_weights_files_rnn_backward = fnmatch.filter(os.listdir(rnn_backward.saving_path), '*.hdf5')

        if len(list_weights_files_rnn_forward) == 0 or len(list_weights_files_rnn_backward) == 0:
            print(
                    '----> [RNN-load_model_from_check_point] --- Found no weights file at %s---' %
                    rnn_forward.saving_path)
            return -1

        list_weights_files_rnn_forward = sorted(list_weights_files_rnn_forward, key=lambda x: int(x.split('-')[1]))
        list_weights_files_rnn_backward = sorted(list_weights_files_rnn_backward, key=lambda x: int(x.split('-')[1]))

        _max_epoch_rnn_forward = int(list_weights_files_rnn_forward[-1].split('-')[1])
        _max_epoch_rnn_backward = int(list_weights_files_rnn_backward[-1].split('-')[1])

        range_epoch = _max_epoch_rnn_forward if _max_epoch_rnn_forward < _max_epoch_rnn_backward else _max_epoch_rnn_backward

        for epoch in range(1, range_epoch + 1, 1):
            rnn_forward.load_model_from_check_point(_from_epoch=epoch, weights_file_type='hdf5')
            rnn_forward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

            rnn_backward.load_model_from_check_point(_from_epoch=epoch, weights_file_type='hdf5')
            rnn_backward.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

            # print(rnn_forward.model.summary())
            # print(rnn_backward.model.summary())

            testing_set = np.copy(copy_testing_set)
            test_set = np.copy(copy_test_set)

            pred_tm, measured_matrix = predict_forward_backward_rnn(test_set=testing_set,
                                                                    n_timesteps=n_timesteps,
                                                                    forward_model=rnn_forward.model,
                                                                    backward_model=rnn_backward.model,
                                                                    sampling_ratio=sampling_ratio,
                                                                    hyperparams=hyperparams)
            ############################################################################################################
            #                                         For testing Flows Clustering

            pred_tm = pred_tm * std_train + mean_train
            ############################################################################################################

            err_ratio = error_ratio(y_true=test_set, y_pred=pred_tm, measured_matrix=measured_matrix)
            # y3 = test_set.flatten()
            # y4 = pred_tm.flatten()
            # a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
            # a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
            # pred_confident = r2_score(y3, y4)

            # err_rat = error_ratio(y_true=test_set, y_pred=pred_tm, measured_matrix=measured_matrix)

            # error = np.expand_dims(np.array([epoch, a_nmae, a_nmse, pred_confident, err_rat]), axis=0)

            # errors = np.concatenate([errors, error], axis=0)
            error = np.expand_dims(np.array([epoch, err_ratio]), axis=0)
            errors = np.concatenate([errors, error], axis=0)
            print('|--- Errors by epoches ---')
            print(errors)
            print('|-------------------------------------------------------')


if __name__ == "__main__":
    np.random.seed(10)
    Abilene24 = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24.csv')

    Abilene24s = Abilene24[0:288 * 7 * 4, :]

    # Get data in 1 week

    n_timesteps_range = [20]

    for n_timesteps in n_timesteps_range:
        with tf.device('/device:GPU:0'):
            # forward_backward_rnn(raw_data=Abilene24, dataset_name='Abilene24', n_timesteps=n_timesteps)
            hyperparams = [2.72, 1, 5.8, 0.4]
            forward_backward_rnn_test(raw_data=Abilene24,
                                      n_timesteps=n_timesteps,
                                      dataset_name='Abilene24',
                                      hyperparams=hyperparams,
                                      epoch_fw=50,
                                      epoch_bw=50)
