import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from common.DataHelper import *
from Utils.DataPreprocessing import *
# from Models.CnnLSTM_model import *
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
# RNN CONFIGURATION
INPUT_DIM = 100
HIDDEN_DIM = 100
LOOK_BACK = 26
N_EPOCH_FW = 50
N_EPOCH_BW = 100
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


# def set_measured_flow_3d(rnn_input_labels, forward_pred, backward_pred, sampling_ratio, hyperparams):
#     """
#
#     :param rnn_input_labels:  shape=(time, od, od, 2)
#     :param forward_pred: shape = (time, od, od)
#     :param backward_pred: shape = (time, od, od)
#     :param sampling_ratio:
#     :param hyperparams:
#     :return:
#     """
#
#     rnn_input = rnn_input_labels[:, :, :, 0]
#     measured_matrix = rnn_input_labels[:, :, :, 1]
#
#     rl_forward, rl_backward = calculate_forward_backward_loss_3d(measured_block=measured_matrix,
#                                                                  pred_forward=forward_pred,
#                                                                  pred_backward=backward_pred,
#                                                                  rnn_input=rnn_input)
#
#     w = calculate_flows_weights_3d(rnn_input=rnn_input,
#                                    rl_forward=rl_forward,
#                                    rl_backward=rl_backward,
#                                    measured_matrix=measured_matrix,
#                                    hyperparams=hyperparams)
#
#     sampling = np.zeros(shape=(rnn_input.shape[1] * rnn_input.shape[2]))
#     m = int(sampling_ratio * rnn_input.shape[1] * rnn_input.shape[2])
#
#     w = w.flatten()
#     sorted_idx_w = np.argpartition(w, m)
#     sampling[sorted_idx_w[:m]] = 1
#
#     sampling = np.expand_dims(sampling, axis=0)
#
#     sampling = np.reshape(sampling, newshape=(rnn_input.shape[1], rnn_input.shape[2]))
#
#     return sampling.astype(bool)


def calculate_updated_weights_3d(n_timesteps, measured_block, forward_loss, backward_loss):
    eps = 0.00001

    labels = measured_block.astype(int)

    measured_count = np.sum(labels, axis=0).astype(float)
    _eta = measured_count / n_timesteps

    # _eta[_eta == 0.0] = eps

    alpha = 1 - _eta  # shape = (od, od)
    alpha = np.tile(np.expand_dims(alpha, axis=0), (n_timesteps, 1, 1))

    # Calculate rho
    rho = np.empty((0, measured_block.shape[1], measured_block.shape[1]))
    mu = np.empty((0, measured_block.shape[1], measured_block.shape[1]))
    for j in range(0, n_timesteps):
        _mu = np.expand_dims((np.sum(measured_block[:(j + 1), :, :], axis=0)) / float(j + 1), axis=0)
        mu = np.concatenate([mu, _mu], axis=0)

        _rho = np.expand_dims((np.sum(measured_block[j:, :, :], axis=0)) / float(n_timesteps - j), axis=0)
        rho = np.concatenate([rho, _rho], axis=0)

    forward_loss = np.tile(np.expand_dims(forward_loss, axis=0), (n_timesteps, 1, 1))
    backward_loss = np.tile(np.expand_dims(backward_loss, axis=0), (n_timesteps, 1, 1))

    forward_loss = forward_loss[0:-2, :, :]
    backward_loss = backward_loss[0:-2, :, :]

    mu = mu[0:-2, :, :]
    rho = rho[2:, :, :]

    alpha = alpha[:-2, :, :]

    beta = (backward_loss + mu) * (1 - alpha) / (forward_loss + backward_loss + mu + rho)

    gamma = (forward_loss + rho) * (1 - alpha) / (forward_loss + backward_loss + mu + rho)

    return alpha, beta, gamma


def calculate_forward_backward_loss_3d(measured_block, pred_forward, pred_backward, rnn_input):
    """

    :param measured_block: shape= (time x od x od)
    :param pred_forward: shape = (time x od x od)
    :param pred_backward: shape = (time x od x od)
    :param rnn_input: = (time x od x od)
    :return:
    """
    eps = 10e-8

    rnn_first_input_updated = np.expand_dims(pred_backward[1, :, :], axis=0)
    rnn_last_input_updated = np.expand_dims(pred_forward[-2, :, :], axis=0)
    rnn_updated_input_forward = np.concatenate(
        [rnn_first_input_updated, pred_forward[0:-2, :, :], rnn_last_input_updated],
        axis=0)
    rnn_updated_input_backward = np.concatenate(
        [rnn_first_input_updated, pred_backward[2:, :, :], rnn_last_input_updated],
        axis=0)
    rl_forward = recovery_loss_3d(rnn_input=rnn_input, rnn_updated=rnn_updated_input_forward,
                                  measured_matrix=measured_block)
    rl_forward[rl_forward == 0] = eps

    rl_backward = recovery_loss_3d(rnn_input=rnn_input, rnn_updated=rnn_updated_input_backward,
                                   measured_matrix=measured_block)
    rl_backward[rl_backward == 0] = eps

    return rl_forward, rl_backward


def updating_historical_data_3d(tm_labels, pred_forward, pred_backward, rnn_input_labels, n_timesteps):
    """

    :param tm_labels: shape = (timesteps, od, od, 2)
    :param pred_forward: shape = (timesteps, od, od)
    :param pred_backward: shape = (timesteps, od, od)
    :param rnn_input_labels: shape = (timesteps, od, od, 2)
    :param n_timesteps: int
    :param raw_data: (timesteps, od, od)
    :return:
    """
    rnn_input = rnn_input_labels[:, :, :, 0]
    measured_block = rnn_input_labels[:, :, :, 1]

    # copy_data_before_update = np.copy(tm_labels[-ntimesteps:, :, :, 0])

    # Calculate loss before update
    # _er = error_ratio(y_true=raw_data[-n_timesteps:, :, :],
    #                   y_pred=tm_labels[-ntimesteps:, :, :, 0],
    #                   measured_matrix=measured_block)

    forward_loss, backward_loss = calculate_forward_backward_loss_3d(measured_block=measured_block,
                                                                     pred_forward=pred_forward,
                                                                     pred_backward=pred_backward,
                                                                     rnn_input=rnn_input)

    alpha, beta, gamma = calculate_updated_weights_3d(n_timesteps=n_timesteps,
                                                      measured_block=measured_block,
                                                      forward_loss=forward_loss,
                                                      backward_loss=backward_loss)

    considered_rnn_input = rnn_input[1:-1, :, :]
    considered_forward = pred_forward[0:-2, :, :]
    considered_backward = pred_backward[2:, :, :]

    updated_rnn_input = considered_rnn_input * alpha + considered_forward * beta + considered_backward * gamma

    sampling_measured_matrix = measured_block.astype(bool)
    sampling_measured_matrix = sampling_measured_matrix[1:-1, :, :]
    inv_sampling_measured_matrix = np.invert(sampling_measured_matrix)

    bidirect_rnn_pred_value = updated_rnn_input * inv_sampling_measured_matrix

    tm_labels[(-n_timesteps + 1):-1, :, :, 0] = \
        tm_labels[(-n_timesteps + 1):-1, :, :, 0] * sampling_measured_matrix + bidirect_rnn_pred_value

    # copy_data_after_update = np.copy(tm_labels[-ntimesteps:, :, :, 0])

    # print(np.array_equal(copy_data_before_update, copy_data_after_update))

    # Calculate loss after update
    # er_ = error_ratio(y_true=raw_data[-ntimesteps:, :, :], y_pred=tm_labels[-ntimesteps:, :, :, 0],
    #                   measured_matrix=measured_block)

    # if er_ > _er:
    #     print('|--- Correcting Fail: Error 1: %.3f - Error 2: %.3f' % (_er, er_))

    return tm_labels


def cnn_brnn_iterated_multi_step_tm_prediction(tm_labels, forward_model, backward_model,
                                               n_timesteps,
                                               prediction_steps,
                                               iterated_multi_steps_tm):
    multi_steps_tm = np.copy(tm_labels[-n_timesteps:, :, :, :])

    for ts_ahead in range(prediction_steps):
        rnn_input = np.copy(multi_steps_tm[-n_timesteps:, :, :, :])  # shape(timesteps, od, od , 2)

        rnn_input_forward = np.expand_dims(rnn_input, axis=0)  # shape(1, timesteps, od, od , 2)

        rnn_input_backward = np.flip(rnn_input, axis=0)
        rnn_input_backward = np.expand_dims(rnn_input_backward, axis=0)  # shape(1, timesteps, od, od , 2)

        # Prediction results from forward network
        predictX = forward_model.predict(rnn_input_forward)  # shape(1, timesteps, od, od , 1)

        predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, od, od , 1)
        predictX = np.squeeze(predictX, axis=3)  # shape(timesteps, od, od)

        # Prediction results from backward network
        predictX_backward = backward_model.predict(rnn_input_backward)  # shape(1, timesteps, od, od , 1)

        predictX_backward = np.squeeze(predictX_backward, axis=0)  # shape(timesteps, od, od , 1)
        predictX_backward = np.squeeze(predictX_backward, axis=3)  # shape(timesteps, od, od)

        # Flipping the backward prediction
        predictX_backward = np.flip(predictX_backward, axis=0)

        # Correcting the imprecise input data
        updating_historical_data_3d(tm_labels=multi_steps_tm, pred_forward=predictX, pred_backward=predictX_backward,
                                    rnn_input_labels=rnn_input, n_timesteps=n_timesteps)

        predict_tm = predictX[-1, :, :]

        sampling = np.zeros(shape=(12, 12, 1))

        # Calculating the true value for the TM
        new_input = predict_tm

        # Concaternating the new tm to the final results
        # Shape = (12, 12, 2)
        new_input = np.concatenate([np.expand_dims(new_input, axis=2), sampling], axis=2)
        new_input = np.expand_dims(new_input, axis=0)  # Shape = (1, 12, 12, 2)
        multi_steps_tm = np.concatenate([multi_steps_tm, new_input], axis=0)  # Shape = (timestep, 12, 12, 2)

    multi_steps_tm = multi_steps_tm[n_timesteps:, :, :, 0]
    multi_steps_tm = np.expand_dims(multi_steps_tm, axis=0)

    iterated_multi_steps_tm = np.concatenate([iterated_multi_steps_tm, multi_steps_tm], axis=0)

    return iterated_multi_steps_tm


def predict_cnn_brnn(test_set, n_timesteps, forward_model, backward_model):
    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = np.copy(test_set[0:n_timesteps, :, :])  # rnn input shape = (timeslot, od, od)
    # Results TM
    # The TF array for random choosing the measured flows
    labels = np.ones((rnn_input.shape[0], rnn_input.shape[1], rnn_input.shape[2]))
    tf = np.array([True, False])

    tm_labels = np.concatenate([np.expand_dims(rnn_input, axis=3), np.expand_dims(labels, axis=3)], axis=3)

    # Predict the TM from time slot look_back
    for ts in range(0, test_set.shape[0] - n_timesteps, 1):
        # print ('--- Predict at timeslot %i ---' % ts)

        rnn_input = np.copy(tm_labels[ts:(ts + n_timesteps), :, :, :])  # shape(timesteps, od, od , 2)

        rnn_input_forward = np.expand_dims(rnn_input, axis=0)  # shape(1, timesteps, od, od , 2)

        rnn_input_backward = np.flip(rnn_input, axis=0)
        rnn_input_backward = np.expand_dims(rnn_input_backward, axis=0)  # shape(1, timesteps, od, od , 2)

        # Prediction results from forward network
        predictX = forward_model.predict(rnn_input_forward)  # shape(1, timesteps, od, od , 1)

        predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, od, od , 1)
        predictX = np.squeeze(predictX, axis=3)  # shape(timesteps, od, od)

        # Prediction results from backward network
        predictX_backward = backward_model.predict(rnn_input_backward)  # shape(1, timesteps, od, od , 1)

        predictX_backward = np.squeeze(predictX_backward, axis=0)  # shape(timesteps, od, od , 1)
        predictX_backward = np.squeeze(predictX_backward, axis=3)  # shape(timesteps, od, od)

        # Flipping the backward prediction
        predictX_backward = np.flip(predictX_backward, axis=0)

        # Correcting the imprecise input data
        updating_historical_data_3d(tm_labels=tm_labels, pred_forward=predictX, pred_backward=predictX_backward,
                                    rnn_input_labels=rnn_input, n_timesteps=n_timesteps)

        predict_tm = predictX[-1, :, :]

        # Selecting next monitored flows randomly

        sampling_ratio = random.uniform(0.1, 0.4)

        sampling = np.random.choice(tf, size=(12, 12), p=(sampling_ratio, 1 - sampling_ratio))
        inv_sampling = np.invert(sampling)

        pred_tm = predict_tm * inv_sampling
        corrected_data = np.copy(test_set[ts + n_timesteps, :, :])
        ground_truth = corrected_data * sampling

        # Calculating the true value for the TM
        new_input = pred_tm + ground_truth

        # Concaternating the new tm to the final results
        # Shape = (12, 12, 2)
        new_input = np.concatenate([np.expand_dims(new_input, axis=2), np.expand_dims(sampling, axis=2)], axis=2)
        new_input = np.expand_dims(new_input, axis=0)  # Shape = (1, 12, 12, 2)
        tm_labels = np.concatenate([tm_labels, new_input], axis=0)  # Shape = (timestep, 12, 12, 2)

    return tm_labels


def cnn_brnn(raw_data, dataset_name='Abilene24_3d', n_timesteps=26):
    test_name = 'cnn_brnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    print('|--- Splitting train-test set.')
    train_set, test_set = prepare_train_test_set_3d(data=raw_data,
                                                    sampling_itvl=5,
                                                    splitting_ratio=splitting_ratio)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_set)
    std_train = np.std(train_set)
    training_set = (train_set - mean_train) / std_train

    if not os.path.isfile(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                    n_timesteps) + dataset_name + '_trainX.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps)):
            os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps))

        print("|--- Create XY sets.")

        train_x, train_y = create_xy_set_3d_dynamic_sampling_ratio(raw_data=training_set,
                                                                   n_timesteps=n_timesteps,
                                                                   random_eps=1,
                                                                   low_ratio=0.1,
                                                                   high_ratio=0.4)

        # Save xy sets to file
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps) + dataset_name + '_trainX.npy',
            train_x)
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps) + dataset_name + '_trainY.npy',
            train_y)

    else:  # Load xy sets from file

        print(
            "|---  Load xy set from " + HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps))

        train_x = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps) + dataset_name + '_trainX.npy')
        train_y = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps) + dataset_name + '_trainY.npy')

    # Create the xy backward set
    # Flip the training set in order to create the xy backward sets
    training_set_backward = np.flip(training_set, axis=0)

    if not os.path.isfile(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                    n_timesteps) + dataset_name + '_trainX_backward.npy'):

        train_x_backward, train_y_backward = create_xy_set_3d_dynamic_sampling_ratio(raw_data=training_set_backward,
                                                                                     n_timesteps=n_timesteps,
                                                                                     random_eps=1,
                                                                                     low_ratio=0.1,
                                                                                     high_ratio=0.4)

        # Save xy backward sets to file

        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps) + dataset_name + '_trainX_backward.npy',
            train_x_backward)
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps) + dataset_name + '_trainY_backward.npy',
            train_y_backward)

    else:  # Load xy backward sets from file

        print("|---  Load xy bw set from " + HOME + '/TM_estimation_dataset/' +
              dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                  n_timesteps))
        train_x_backward = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps) + dataset_name + '_trainX_backward.npy')
        train_y_backward = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Dynamic_Sampling_timesteps_%i/' % (
                n_timesteps) + dataset_name + '_trainY_backward.npy')

    print("|--- Create CNN_BRNN model.")

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

    forward_model_name = 'Forward_CNN_layers_%i_filters%skernels%sdropouts%srnn_dropouts%s' % \
                         (cnn_layers, filters_2_str, kernel_2_str, dropouts_2_str, rnn_dropouts_2_str)

    backward_model_name = 'Backward_CNN_layers_%i_filters%skernels%sdropouts%srnn_dropouts%s' % \
                          (cnn_layers_backward, filters_2_str_backward, kernel_2_str_backward, dropouts_2_str_backward,
                           rnn_dropouts_2_str_backward)

    model_name = 'BRNN_%s_%s' % (forward_model_name, backward_model_name)

    # CNN_BRNN forward model
    cnn_brnn_model_forward = ConvLSTM(n_timsteps=n_timesteps, height=12, weight=12, depth=2,
                                      cnn_layers=cnn_layers, a_filters=filters, a_strides=strides, dropouts=dropouts,
                                      kernel_sizes=kernel_sizes,
                                      rnn_dropouts=rnn_dropouts,
                                      check_point=True,
                                      saving_path=model_recorded_path + 'Dynamic_Sampling_timesteps_%i/%s/'
                                                  % (n_timesteps, forward_model_name))

    # CNN_BRNN backward model
    cnn_brnn_model_backward = ConvLSTM(n_timsteps=n_timesteps, height=12, weight=12, depth=2,
                                       cnn_layers=cnn_layers_backward, a_filters=filters_backward,
                                       a_strides=strides_backward, dropouts=dropouts_backward,
                                       kernel_sizes=kernel_sizes_backward,
                                       rnn_dropouts=rnn_dropouts_backward,
                                       check_point=True,
                                       saving_path=model_recorded_path + 'Dynamic_Sampling_timesteps_%i/%s/'
                                                   % (n_timesteps, backward_model_name))

    if os.path.isfile(path=cnn_brnn_model_forward.saving_path + 'weights-%i-0.00.hdf5' % N_EPOCH_FW):
        print('|--- Forward model exist!')
        cnn_brnn_model_forward.load_model_from_check_point(_from_epoch=N_EPOCH_FW, weights_file_type='hdf5')
    else:
        print('|--- Compile model. Saving path %s --- ' % cnn_brnn_model_forward.saving_path)

        # Load model check point
        from_epoch = cnn_brnn_model_forward.load_model_from_check_point()
        if from_epoch > 0:
            cnn_brnn_model_forward.model.compile(loss='mean_squared_error', optimizer='adam',
                                                 metrics=['mse', 'mae', 'accuracy'])
            print('|--- Continue training forward model from epoch %i --- ' % from_epoch)
            training_forward_history = cnn_brnn_model_forward.model.fit(train_x,
                                                                        train_y,
                                                                        batch_size=BATCH_SIZE,
                                                                        epochs=N_EPOCH_FW,
                                                                        initial_epoch=from_epoch,
                                                                        validation_split=0.25,
                                                                        callbacks=cnn_brnn_model_forward.callbacks_list)
            # Plot the training history
            cnn_brnn_model_forward.plot_model_metrics(training_forward_history,
                                                      plot_prefix_name='Metrics')

        else:
            print('|--- Training new forward model.')

            cnn_brnn_model_forward.model.compile(loss='mean_squared_error', optimizer='adam',
                                                 metrics=['mse', 'mae', 'accuracy'])

            training_forward_history = cnn_brnn_model_forward.model.fit(train_x,
                                                                        train_y,
                                                                        batch_size=BATCH_SIZE,
                                                                        epochs=N_EPOCH_FW,
                                                                        validation_split=0.25,
                                                                        callbacks=cnn_brnn_model_forward.callbacks_list)

            # Plot the training history
            cnn_brnn_model_forward.plot_model_metrics(training_forward_history,
                                                      plot_prefix_name='Metrics')

    # Training cnn_brnn backward model
    if os.path.isfile(path=cnn_brnn_model_backward.saving_path + 'weights-%i-0.00.hdf5' % N_EPOCH_BW):
        print('|--- Backward model exist!')
        cnn_brnn_model_backward.load_model_from_check_point(_from_epoch=N_EPOCH_BW, weights_file_type='hdf5')
    else:
        print('|---Compile model. Saving path: %s' % cnn_brnn_model_backward.saving_path)
        # Load model from check point
        from_epoch_backward = cnn_brnn_model_backward.load_model_from_check_point()
        if from_epoch_backward > 0:

            cnn_brnn_model_backward.model.compile(loss='mean_squared_error', optimizer='adam',
                                                  metrics=['mse', 'mae', 'accuracy'])

            training_backward_history = cnn_brnn_model_backward.model.fit(train_x_backward,
                                                                          train_y_backward,
                                                                          batch_size=BATCH_SIZE,
                                                                          epochs=N_EPOCH_BW,
                                                                          initial_epoch=from_epoch_backward,
                                                                          validation_split=0.25,
                                                                          callbacks=cnn_brnn_model_backward.callbacks_list)
            cnn_brnn_model_backward.plot_model_metrics(training_backward_history,
                                                       plot_prefix_name='Metrics')

        else:
            print('|--- Training new backward model.')

            cnn_brnn_model_backward.model.compile(loss='mean_squared_error', optimizer='adam',
                                                  metrics=['mse', 'mae', 'accuracy'])

            training_backward_history = cnn_brnn_model_backward.model.fit(train_x_backward,
                                                                          train_y_backward,
                                                                          batch_size=BATCH_SIZE,
                                                                          epochs=N_EPOCH_BW,
                                                                          validation_split=0.25,
                                                                          callbacks=cnn_brnn_model_backward.callbacks_list)
            cnn_brnn_model_backward.plot_model_metrics(training_backward_history,
                                                       plot_prefix_name='Metrics')

    # print(cnn_brnn_model_forward.model.summary())
    # print(cnn_brnn_model_backward.model.summary())

    return


def cnn_brnn_test_loop(test_set, testing_set, cnn_brnn_model_forward, cnn_brnn_model_backward,
                       epoch_fw, epoch_bw, n_timesteps, std_train, mean_train, n_running_time,
                       results_path):
    err_ratio_temp = []
    err_ratio_ims_temp = []
    for running_time in range(n_running_time):
        print('|--- Epoch_fw %d - Epoch_bw %d  - Running time: %d' % (epoch_fw, epoch_bw, running_time))

        cnn_brnn_model_forward.load_model_from_check_point(_from_epoch=epoch_fw, weights_file_type='hdf5')
        cnn_brnn_model_forward.model.compile(loss='mean_squared_error', optimizer='adam',
                                             metrics=['mse', 'mae', 'accuracy'])

        cnn_brnn_model_backward.load_model_from_check_point(_from_epoch=epoch_bw, weights_file_type='hdf5')
        cnn_brnn_model_backward.model.compile(loss='mean_squared_error', optimizer='adam',
                                              metrics=['mse', 'mae', 'accuracy'])
        _testing_set = np.copy(testing_set)
        _test_set = np.copy(test_set)

        tm_labels = predict_cnn_brnn(test_set=_testing_set,
                                     n_timesteps=n_timesteps,
                                     forward_model=cnn_brnn_model_forward.model,
                                     backward_model=cnn_brnn_model_backward.model,
                                     )
        pred_tm = tm_labels[:, :, :, 0]
        measured_matrix = tm_labels[:, :, :, 1]

        pred_tm = pred_tm * std_train + mean_train

        post_checking(y_true=np.copy(test_set), y_pred=np.copy(pred_tm), measured_matrix=measured_matrix)

        err = error_ratio(y_true=np.copy(test_set), y_pred=np.copy(pred_tm), measured_matrix=measured_matrix)
        r2_score = calculate_r2_score(y_true=np.copy(test_set), y_pred=np.copy(pred_tm))
        rmse = rmse_tm_prediction(y_true=np.copy(test_set), y_pred=np.copy(pred_tm))

        err_ratio_temp.append(err)

        np.save(file=results_path + 'Predicted_tm_running_time_%d' % running_time,
                arr=np.copy(pred_tm))
        np.save(file=results_path + 'Predicted_measured_matrix_running_time_%d' % running_time,
                arr=measured_matrix)
        np.save(file=results_path + 'Ground_truth_tm_running_time_%d' % running_time,
                arr=np.copy(test_set))

        print('|--- err: %.3f --- rmse: %.3f --- r2: %.3f' % (err, rmse, r2_score))
        print('|--- Sampling ratio: %f' % (np.sum(measured_matrix) / np.size(measured_matrix)))

    return err_ratio_temp


def post_checking(y_true, y_pred, measured_matrix):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    measured_matrix = measured_matrix.flatten()
    observated_indice = np.where(measured_matrix == True)
    print('Post checking:')
    print(np.array_equal(y_true[observated_indice], y_pred[observated_indice]))


def calculate_lstm_iterated_multi_step_tm_prediction_errors(iterated_multi_step_pred_tm, test_set, n_timesteps,
                                                            prediction_steps):
    iterated_multi_step_test_set = np.empty(shape=(0, prediction_steps, test_set.shape[1], test_set.shape[2]))

    for ts in range(test_set.shape[0] - n_timesteps - prediction_steps):
        multi_step_test_set = np.copy(test_set[(ts + n_timesteps): (ts + n_timesteps + prediction_steps), :, :])
        multi_step_test_set = np.expand_dims(multi_step_test_set, axis=0)
        iterated_multi_step_test_set = np.concatenate([iterated_multi_step_test_set, multi_step_test_set], axis=0)

    # measured_matrix = np.zeros(shape=iterated_multi_step_test_set.shape)
    #
    # err_ratio = error_ratio(y_pred=iterated_multi_step_pred_tm,
    #                         y_true=iterated_multi_step_test_set,
    #                         measured_matrix=measured_matrix)

    return iterated_multi_step_test_set


def cnn_brnn_test(raw_data, dataset_name, n_timesteps, with_epoch_fw=0, with_epoch_bw=0):
    test_name = 'cnn_brnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    print('|--- Splitting train-test set.')
    train_set, test_set = prepare_train_test_set_3d(data=raw_data,
                                                    sampling_itvl=5,
                                                    splitting_ratio=splitting_ratio)

    test_set = test_set[0:-864, :, :]

    copy_test_set = np.copy(test_set)

    testing_set = np.copy(test_set)

    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_set)
    std_train = np.std(train_set)
    testing_set = (testing_set - mean_train) / std_train
    copy_testing_set = np.copy(testing_set)

    print("|--- Create CNN_BRNN model.")

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

    forward_model_name = 'Forward_CNN_layers_%i_filters%skernels%sdropouts%srnn_dropouts%s' % \
                         (cnn_layers, filters_2_str, kernel_2_str, dropouts_2_str, rnn_dropouts_2_str)

    backward_model_name = 'Backward_CNN_layers_%i_filters%skernels%sdropouts%srnn_dropouts%s' % \
                          (cnn_layers_backward, filters_2_str_backward, kernel_2_str_backward, dropouts_2_str_backward,
                           rnn_dropouts_2_str_backward)

    model_name = 'BRNN_%s_%s' % (forward_model_name, backward_model_name)

    # CNN_BRNN forward model
    cnn_brnn_model_forward = ConvLSTM(n_timsteps=n_timesteps, height=12, weight=12, depth=2,
                                      cnn_layers=cnn_layers, a_filters=filters, a_strides=strides, dropouts=dropouts,
                                      kernel_sizes=kernel_sizes,
                                      rnn_dropouts=rnn_dropouts,
                                      check_point=True,
                                      saving_path=model_recorded_path + 'Dynamic_Sampling_timesteps_%i/%s/' %
                                                  (n_timesteps, forward_model_name))

    # CNN_BRNN backward model
    cnn_brnn_model_backward = ConvLSTM(n_timsteps=n_timesteps, height=12, weight=12, depth=2,
                                       cnn_layers=cnn_layers_backward, a_filters=filters_backward,
                                       a_strides=strides_backward, dropouts=dropouts_backward,
                                       kernel_sizes=kernel_sizes_backward,
                                       rnn_dropouts=rnn_dropouts_backward,
                                       check_point=True,
                                       saving_path=model_recorded_path + 'Dynamic_Sampling_timesteps_%i/%s/' %
                                                   (n_timesteps, backward_model_name))

    sampling_timesteps = 'Dynamic_Sampling_timesteps_%d' % (n_timesteps)

    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/' % \
                  (dataset_name, test_name, sampling_timesteps, model_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if with_epoch_fw != 0 or with_epoch_bw != 0:
        n_running_time = 3

        err_ratio_temp = cnn_brnn_test_loop(
            test_set=copy_test_set,
            testing_set=copy_testing_set,
            cnn_brnn_model_forward=cnn_brnn_model_forward,
            cnn_brnn_model_backward=cnn_brnn_model_backward,
            epoch_fw=with_epoch_fw,
            epoch_bw=with_epoch_bw,
            n_timesteps=n_timesteps,
            std_train=std_train,
            mean_train=mean_train,
            n_running_time=n_running_time,
            results_path=result_path)

        err_ratio_temp = np.array(err_ratio_temp)
        err_ratio_temp = np.reshape(err_ratio_temp, newshape=(n_running_time, 1))
        err_ratio = np.mean(err_ratio_temp)
        err_ratio_std = np.std(err_ratio_temp)

        print('Hyperparam: ' + str(hyperparams))
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
               (with_epoch_fw, with_epoch_bw, n_running_time)))
        np.savetxt(fname=result_path + 'Epoch_fw_%i_Epoch_bw_%i_n_running_time_%i.csv' %
                         (with_epoch_fw, with_epoch_bw, n_running_time),
                   X=results, delimiter=',')
    return


if __name__ == "__main__":
    np.random.seed(10)

    if not os.path.isfile(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/Abilene24_3d/'):
            os.makedirs(HOME + '/TM_estimation_dataset/Abilene24_3d/')
        load_abilene_3d()

    Abilene24_3d = np.load(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d.npy')

    ntimesteps_range = [26]
    hyperparams = [2.0, 1.0, 5.0, 0.4]
    print('Hyperparams')
    print(hyperparams)
    with tf.device('/device:GPU:1'):
        # cnn_brnn(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=26)
        cnn_brnn_test(raw_data=Abilene24_3d, dataset_name='Abilene24_3d', n_timesteps=26,
                      with_epoch_fw=50, with_epoch_bw=100)
