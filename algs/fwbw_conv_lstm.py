import pandas as pd
import tensorflow as tf

import common.convlstm_config as ConvlstmConfig
from Models.ConvLSTM_model import *
from common.DataHelper import *
from common.DataPreprocessing import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

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


def calculate_flows_weights_3d(rnn_input, rl_forward, rl_backward, measured_matrix):
    eps = 10e-5

    cl = calculate_consecutive_loss_3d(measured_matrix).astype(float)

    flows_stds = np.std(rnn_input, axis=0)

    cl_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(cl)
    flows_stds_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(flows_stds)
    rl_forward_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(rl_forward)
    rl_backward_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(rl_backward)

    w = 1 / (rl_forward_scaled * ConvlstmConfig.HYPERPARAMS[0] +
             rl_backward_scaled * ConvlstmConfig.HYPERPARAMS[1] +
             cl_scaled * ConvlstmConfig.HYPERPARAMS[2] +
             flows_stds_scaled * ConvlstmConfig.HYPERPARAMS[3])

    return w


def set_measured_flow_3d(rnn_input_labels, forward_pred, backward_pred):
    rnn_input = rnn_input_labels[:, :, :, 0]
    measured_matrix = rnn_input_labels[:, :, :, 1]

    rl_forward, rl_backward = calculate_forward_backward_loss_3d(measured_block=measured_matrix,
                                                                 pred_forward=forward_pred,
                                                                 pred_backward=backward_pred,
                                                                 rnn_input=rnn_input)

    w = calculate_flows_weights_3d(rnn_input=rnn_input,
                                   rl_forward=rl_forward,
                                   rl_backward=rl_backward,
                                   measured_matrix=measured_matrix)

    sampling = np.zeros(shape=(rnn_input.shape[1] * rnn_input.shape[2]))
    m = int(ConvlstmConfig.MON_RAIO * rnn_input.shape[1] * rnn_input.shape[2])

    w = w.flatten()
    sorted_idx_w = np.argpartition(w, m)
    sampling[sorted_idx_w[:m]] = 1

    sampling = np.expand_dims(sampling, axis=0)

    sampling = np.reshape(sampling, newshape=(rnn_input.shape[1], rnn_input.shape[2]))

    return sampling.astype(bool)


def calculate_updated_weights_3d(measured_block, forward_loss, backward_loss):
    labels = measured_block.astype(int)

    measured_count = np.sum(labels, axis=0).astype(float)
    _eta = measured_count / ConvlstmConfig.LSTM_STEP

    # _eta[_eta == 0.0] = eps

    alpha = 1 - _eta  # shape = (od, od)
    alpha = np.tile(np.expand_dims(alpha, axis=0), (ConvlstmConfig.LSTM_STEP, 1, 1))

    # Calculate rho
    rho = np.empty((0, measured_block.shape[1], measured_block.shape[1]))
    mu = np.empty((0, measured_block.shape[1], measured_block.shape[1]))
    for j in range(0, ConvlstmConfig.LSTM_STEP):
        _mu = np.expand_dims((np.sum(measured_block[:(j + 1), :, :], axis=0)) / float(j + 1), axis=0)
        mu = np.concatenate([mu, _mu], axis=0)

        _rho = np.expand_dims((np.sum(measured_block[j:, :, :], axis=0)) / float(ConvlstmConfig.LSTM_STEP - j), axis=0)
        rho = np.concatenate([rho, _rho], axis=0)

    forward_loss = np.tile(np.expand_dims(forward_loss, axis=0), (ConvlstmConfig.LSTM_STEP, 1, 1))
    backward_loss = np.tile(np.expand_dims(backward_loss, axis=0), (ConvlstmConfig.LSTM_STEP, 1, 1))

    forward_loss = forward_loss[0:-2, :, :]
    backward_loss = backward_loss[0:-2, :, :]

    mu = mu[0:-2, :, :]
    rho = rho[2:, :, :]

    alpha = alpha[:-2, :, :]

    beta = (backward_loss + mu) * (1 - alpha) / (forward_loss + backward_loss + mu + rho)

    gamma = (forward_loss + rho) * (1 - alpha) / (forward_loss + backward_loss + mu + rho)

    return alpha, beta, gamma


def calculate_forward_backward_loss_3d(measured_block, pred_forward, pred_backward, rnn_input):
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


def updating_historical_data_3d(tm_labels, pred_forward, pred_backward, rnn_input_labels):
    rnn_input = rnn_input_labels[:, :, :, 0]
    measured_block = rnn_input_labels[:, :, :, 1]

    forward_loss, backward_loss = calculate_forward_backward_loss_3d(measured_block=measured_block,
                                                                     pred_forward=pred_forward,
                                                                     pred_backward=pred_backward,
                                                                     rnn_input=rnn_input)

    alpha, beta, gamma = calculate_updated_weights_3d(measured_block=measured_block,
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

    tm_labels[(-ConvlstmConfig.LSTM_STEP + 1):-1, :, :, 0] = \
        tm_labels[(-ConvlstmConfig.LSTM_STEP + 1):-1, :, :, 0] * sampling_measured_matrix + bidirect_rnn_pred_value

    return tm_labels


def cnn_brnn_iterated_multi_step_tm_prediction(tm_labels, forward_model, backward_model,
                                               iterated_multi_steps_tm):
    multi_steps_tm = np.copy(tm_labels[-ConvlstmConfig.LSTM_STEP:, :, :, :])

    for ts_ahead in range(ConvlstmConfig.IMS_STEP):
        rnn_input = np.copy(multi_steps_tm[-ConvlstmConfig.LSTM_STEP:, :, :, :])  # shape(timesteps, od, od , 2)

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
                                    rnn_input_labels=rnn_input)

        predict_tm = predictX[-1, :, :]

        sampling = np.zeros(shape=(12, 12, 1))

        # Calculating the true value for the TM
        new_input = predict_tm

        # Concaternating the new tm to the final results
        # Shape = (12, 12, 2)
        new_input = np.concatenate([np.expand_dims(new_input, axis=2), sampling], axis=2)
        new_input = np.expand_dims(new_input, axis=0)  # Shape = (1, 12, 12, 2)
        multi_steps_tm = np.concatenate([multi_steps_tm, new_input], axis=0)  # Shape = (timestep, 12, 12, 2)

    multi_steps_tm = multi_steps_tm[ConvlstmConfig.LSTM_STEP:, :, :, 0]
    multi_steps_tm = np.expand_dims(multi_steps_tm, axis=0)

    iterated_multi_steps_tm = np.concatenate([iterated_multi_steps_tm, multi_steps_tm], axis=0)

    return iterated_multi_steps_tm


def predict_fwbw_conv_lstm(test_data, forward_model, backward_model):
    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = np.copy(test_data[0:ConvlstmConfig.LSTM_STEP, :, :])  # rnn input shape = (timeslot, od, od)
    # Results TM
    # The TF array for random choosing the measured flows
    labels = np.ones((rnn_input.shape[0], rnn_input.shape[1], rnn_input.shape[2]))
    tf = 7.2

    tm_labels = np.concatenate([np.expand_dims(rnn_input, axis=3), np.expand_dims(labels, axis=3)], axis=3)

    day_size = 24 * (60 / 5)
    iterated_multi_steps_tm = np.empty(shape=(0, ConvlstmConfig.IMS_STEP, 12, 12))

    # Predict the TM from time slot look_back
    for ts in range(0, test_data.shape[0] - ConvlstmConfig.LSTM_STEP, 1):
        date = int(ts / day_size)
        # print ('--- Predict at timeslot %i ---' % tslot)

        if ts < test_data.shape[0] - ConvlstmConfig.LSTM_STEP - ConvlstmConfig.IMS_STEP:
            iterated_multi_steps_tm = cnn_brnn_iterated_multi_step_tm_prediction(
                tm_labels=tm_labels,
                forward_model=forward_model,
                backward_model=backward_model,
                iterated_multi_steps_tm=iterated_multi_steps_tm)

        rnn_input = np.copy(tm_labels[ts:(ts + ConvlstmConfig.LSTM_STEP), :, :, :])  # shape(timesteps, od, od , 2)

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
                                    rnn_input_labels=rnn_input)

        predict_tm = predictX[-1, :, :]

        sampling = set_measured_flow_3d(rnn_input_labels=rnn_input,
                                        forward_pred=predictX,
                                        backward_pred=predictX_backward)

        # Selecting next monitored flows randomly
        # sampling = np.random.choice(tf, size=(12, 12), p=(sampling_ratio, 1 - sampling_ratio))
        inv_sampling = np.invert(sampling)

        pred_tm = predict_tm * inv_sampling
        corrected_data = np.copy(test_data[ts + ConvlstmConfig.LSTM_STEP, :, :])
        ground_truth = corrected_data * sampling

        # Calculating the true value for the TM
        new_input = pred_tm + ground_truth

        # Concaternating the new tm to the final results
        # Shape = (12, 12, 2)
        new_input = np.concatenate([np.expand_dims(new_input, axis=2), np.expand_dims(sampling, axis=2)], axis=2)
        new_input = np.expand_dims(new_input, axis=0)  # Shape = (1, 12, 12, 2)
        tm_labels = np.concatenate([tm_labels, new_input], axis=0)  # Shape = (timestep, 12, 12, 2)

    return tm_labels, iterated_multi_steps_tm


def build_model(args, input_shape):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    # CNN_BRNN forward model
    fw_net = ConvLSTM(input_shape=input_shape,
                      cnn_layers=ConvlstmConfig.CNN_LAYERS,
                      a_filters=ConvlstmConfig.FILTERS,
                      a_strides=ConvlstmConfig.STRIDES,
                      dropouts=ConvlstmConfig.DROPOUTS,
                      kernel_sizes=ConvlstmConfig.KERNEL_SIZE,
                      rnn_dropouts=ConvlstmConfig.RNN_DROPOUTS,
                      alg_name=alg_name,
                      tag=tag,
                      check_point=True,
                      saving_path=Config.MODEL_SAVE + '[fw]{}-{}-{}/'.format(data_name, alg_name, tag))

    # CNN_BRNN backward model
    bw_net = ConvLSTM(input_shape=input_shape,
                      cnn_layers=ConvlstmConfig.CNN_LAYERS,
                      a_filters=ConvlstmConfig.FILTERS,
                      a_strides=ConvlstmConfig.STRIDES,
                      dropouts=ConvlstmConfig.DROPOUTS,
                      kernel_sizes=ConvlstmConfig.KERNEL_SIZE,
                      rnn_dropouts=ConvlstmConfig.RNN_DROPOUTS,
                      alg_name=alg_name,
                      tag=tag,
                      check_point=True,
                      saving_path=Config.MODEL_SAVE + '[bw]{}-{}-{}/'.format(data_name, alg_name, tag))

    return fw_net, bw_net


def train_fwbw_conv_lstm(data, args):
    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_test_set_3d(data=data)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data = (train_data - mean_train) / std_train
    valid_data = (valid_data - mean_train) / std_train
    test_data = (test_data - mean_train) / std_train

    print("|--- Create FWBW_CONVLSTM model.")
    input_shape = (ConvlstmConfig.LSTM_STEP,
                   ConvlstmConfig.CNN_WIDE, ConvlstmConfig.CNN_HIGH, ConvlstmConfig.CNN_CHANNEL)

    fw_net, bw_net = build_model(args, input_shape)

    if ConvlstmConfig.MON_RAIO is not None:
        generator_train_data = generator_convlstm_train_data_fix_ratio
    else:
        generator_train_data = generator_convlstm_train_data

    if os.path.isfile(path=fw_net.saving_path + 'weights-%i-0.00.hdf5' % Config.N_EPOCH):
        print('|--- Forward model exist!')
        fw_net.load_model_from_check_point(_from_epoch=Config.N_EPOCH, weights_file_type='hdf5')
    else:
        print('|--- Compile model. Saving path %s --- ' % fw_net.saving_path)

        # Load model check point
        from_epoch = fw_net.load_model_from_check_point()
        if from_epoch > 0:
            fw_net.model.compile(loss='mean_squared_error', optimizer='adam',
                                 metrics=['mse', 'mae', 'accuracy'])
            print('|--- Continue training forward model from epoch %i --- ' % from_epoch)
            training_fw_history = fw_net.model.fit_generator(
                generator_train_data(train_data,
                                     input_shape,
                                     ConvlstmConfig.MON_RAIO,
                                     0.5,
                                     Config.BATCH_SIZE),
                epochs=Config.N_EPOCH,
                steps_per_epoch=Config.NUM_ITER,
                initial_epoch=from_epoch,
                validation_data=generator_convlstm_train_data(valid_data, input_shape, ConvlstmConfig.MON_RAIO,
                                                              0.5,
                                                              Config.BATCH_SIZE),
                validation_steps=int(Config.NUM_ITER * 0.2),
                callbacks=fw_net.callbacks_list,
                use_multiprocessing=True,
                workers=4,
                max_queue_size=1028)
        else:
            print('|--- Training new forward model.')

            fw_net.model.compile(loss='mean_squared_error', optimizer='adam',
                                 metrics=['mse', 'mae', 'accuracy'])

            training_fw_history = fw_net.model.fit_generator(
                generator_train_data(train_data,
                                     input_shape,
                                     None,
                                     0.5,
                                     Config.BATCH_SIZE),
                epochs=Config.N_EPOCH,
                steps_per_epoch=Config.NUM_ITER,
                validation_data=generator_convlstm_train_data(valid_data, input_shape, ConvlstmConfig.MON_RAIO,
                                                              0.5,
                                                              Config.BATCH_SIZE),
                validation_steps=int(Config.NUM_ITER * 0.2),
                callbacks=fw_net.callbacks_list,
                use_multiprocessing=True,
                workers=4,
                max_queue_size=1028)

        # Plot the training history
        if training_fw_history is not None:
            fw_net.plot_training_history(training_fw_history)

    train_data_bw = np.flip(train_data, axis=0)
    vallid_data_bw = np.flip(valid_data, axis=0)

    # Training cnn_brnn backward model
    if os.path.isfile(path=bw_net.saving_path + 'weights-%i-0.00.hdf5' % Config.N_EPOCH):
        print('|--- Backward model exist!')
        bw_net.load_model_from_check_point(_from_epoch=Config.N_EPOCH, weights_file_type='hdf5')
    else:
        print('|---Compile model. Saving path: %s' % bw_net.saving_path)
        # Load model from check point
        from_epoch_backward = bw_net.load_model_from_check_point()
        if from_epoch_backward > 0:

            bw_net.model.compile(loss='mean_squared_error', optimizer='adam',
                                 metrics=['mse', 'mae', 'accuracy'])

            training_bw_history = bw_net.model.fit(
                generator_train_data(train_data_bw,
                                     input_shape,
                                     ConvlstmConfig.MON_RAIO,
                                     0.5,
                                     Config.BATCH_SIZE),
                epochs=Config.N_EPOCH,
                steps_per_epoch=Config.NUM_ITER,
                initial_epoch=from_epoch_backward,
                validation_data=generator_convlstm_train_data(vallid_data_bw,
                                                              input_shape,
                                                              ConvlstmConfig.MON_RAIO,
                                                              0.5,
                                                              Config.BATCH_SIZE),
                validation_steps=int(Config.NUM_ITER * 0.2),
                callbacks=bw_net.callbacks_list,
                use_multiprocessing=True,
                workers=4,
                max_queue_size=1028)

        else:
            print('|--- Training new backward model.')

            bw_net.model.compile(loss='mean_squared_error', optimizer='adam',
                                 metrics=['mse', 'mae', 'accuracy'])

            training_bw_history = bw_net.model.fit_generator(
                generator_train_data(train_data_bw,
                                     input_shape,
                                     ConvlstmConfig.MON_RAIO,
                                     0.5,
                                     Config.BATCH_SIZE),
                epochs=Config.N_EPOCH,
                steps_per_epoch=Config.NUM_ITER,
                validation_data=generator_convlstm_train_data(vallid_data_bw,
                                                              input_shape,
                                                              ConvlstmConfig.MON_RAIO,
                                                              0.5,
                                                              Config.BATCH_SIZE),
                validation_steps=int(Config.NUM_ITER * 0.2),
                callbacks=bw_net.callbacks_list,
                use_multiprocessing=True,
                workers=4,
                max_queue_size=1028)
        if training_bw_history is not None:
            bw_net.plot_training_history(training_bw_history)

    print('---------------------------------FW_NET SUMMARY---------------------------------')
    print(fw_net.model.summary())
    print('---------------------------------BW_NET SUMMARY---------------------------------')
    print(bw_net.model.summary())

    return


def calculate_lstm_iterated_multi_step_tm_prediction_errors(test_set):
    iterated_multi_step_test_set = np.zeros(
        shape=(test_set.shape[0] - ConvlstmConfig.LSTM_STEP - ConvlstmConfig.IMS_STEP,
               ConvlstmConfig.IMS_STEP, test_set.shape[1],
               test_set.shape[2]))

    for ts in range(test_set.shape[0] - ConvlstmConfig.LSTM_STEP - ConvlstmConfig.IMS_STEP):
        multi_step_test_set = np.copy(
            test_set[(ts + ConvlstmConfig.LSTM_STEP): (ts + ConvlstmConfig.LSTM_STEP + ConvlstmConfig.IMS_STEP), :, :])
        iterated_multi_step_test_set[ts] = multi_step_test_set

    return iterated_multi_step_test_set


def test_fwbw_conv_lstm(data, args):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_test_set_3d(data=data)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data = (train_data - mean_train) / std_train
    valid_data = (valid_data - mean_train) / std_train
    test_data_normalized = (test_data - mean_train) / std_train

    print("|--- Create FWBW_CONVLSTM model.")
    input_shape = (ConvlstmConfig.LSTM_STEP,
                   ConvlstmConfig.CNN_WIDE, ConvlstmConfig.CNN_HIGH, ConvlstmConfig.CNN_CHANNEL)

    fw_net, bw_net = build_model(args, input_shape)

    results_summary = pd.read_csv(Config.RESULTS_PATH + '{}-{}-{}.csv'.format(data_name, alg_name, tag))

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    for i in range(Config.TESTING_TIME):
        tm_labels, iterated_multi_steps_tm = predict_fwbw_conv_lstm(test_data=test_data_normalized,
                                                                    forward_model=fw_net.model,
                                                                    backward_model=bw_net.model)

        pred_tm = tm_labels[:, :, :, 0]
        measured_matrix = tm_labels[:, :, :, 1]

        pred_tm = pred_tm * std_train + mean_train

        err.append(error_ratio(y_true=test_data_normalized, y_pred=np.copy(pred_tm), measured_matrix=measured_matrix))
        r2_score.append(calculate_r2_score(y_true=test_data_normalized, y_pred=np.copy(pred_tm)))
        rmse.append(rmse_tm_prediction(y_true=test_data_normalized, y_pred=np.copy(pred_tm)))

        iterated_multi_steps_tm = iterated_multi_steps_tm * std_train + mean_train

        iterated_multi_step_test_set = calculate_lstm_iterated_multi_step_tm_prediction_errors(test_set=test_data)

        measured_matrix = np.zeros(shape=iterated_multi_step_test_set.shape)
        err_ims.append(error_ratio(y_pred=iterated_multi_steps_tm,
                                   y_true=iterated_multi_step_test_set,
                                   measured_matrix=measured_matrix))

        r2_score_ims.append(calculate_r2_score(y_true=iterated_multi_step_test_set, y_pred=iterated_multi_steps_tm))
        rmse_ims.append(rmse_tm_prediction(y_true=iterated_multi_step_test_set, y_pred=iterated_multi_steps_tm))

    results_summary['running_time'] = range(Config.TESTING_TIME)
    results_summary['err'] = err
    results_summary['r2_score'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['err_ims'] = err_ims
    results_summary['r2_score_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    results_summary.to_csv(Config.RESULTS_PATH + '{}-{}-{}.csv'.format(data_name, alg_name, tag),
                           index=False)

    return


def try_hyperparams(raw_data, dataset_name, n_timesteps, hyperparams=[], with_epoch_fw=0, with_epoch_bw=0,
                    sampling_ratio=0.10):
    test_name = 'cnn_brnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/CNN_layers_%i_timesteps_%i/' % (3, n_timesteps)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

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
                                      saving_path=model_recorded_path + 'Sampling_%.2f_timesteps_%i/%s/' %
                                                  (sampling_ratio, n_timesteps, forward_model_name))

    # CNN_BRNN backward model
    cnn_brnn_model_backward = ConvLSTM(n_timsteps=n_timesteps, height=12, weight=12, depth=2,
                                       cnn_layers=cnn_layers_backward, a_filters=filters_backward,
                                       a_strides=strides_backward, dropouts=dropouts_backward,
                                       kernel_sizes=kernel_sizes_backward,
                                       rnn_dropouts=rnn_dropouts_backward,
                                       check_point=True,
                                       saving_path=model_recorded_path + 'Sampling_%.2f_timesteps_%i/%s/' %
                                                   (sampling_ratio, n_timesteps, backward_model_name))

    sampling_timesteps = 'Sampling_%.2f_timesteps_%d' % (sampling_ratio, n_timesteps)

    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/try_hyperparams/' % \
                  (dataset_name, test_name, sampling_timesteps, model_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if with_epoch_fw != 0 or with_epoch_bw != 0:

        errors_params = np.empty(shape=(0, 5))

        for p_1 in np.nditer(np.arange(2.18, 2.22, 0.01)):
            for p_2 in np.nditer(np.arange(5.08, 5.12, 0.01)):
                for p_3 in np.nditer(np.arange(0.28, 0.32, 0.01)):
                    hyperparams = [p_1, 1.0, p_2, p_3]
                    error_params = hyperparams

                    print('|--- Try params')
                    print(hyperparams)
                    n_running_time = 1
                    err_ratio_temp = cnn_brnn_test_loop(test_set=copy_test_set,
                                                        testing_set=copy_testing_set,
                                                        cnn_brnn_model_forward=cnn_brnn_model_forward,
                                                        cnn_brnn_model_backward=cnn_brnn_model_backward,
                                                        epoch_fw=with_epoch_fw,
                                                        epoch_bw=with_epoch_bw,
                                                        n_timesteps=n_timesteps,
                                                        sampling_ratio=sampling_ratio,
                                                        std_train=std_train,
                                                        mean_train=mean_train,
                                                        hyperparams=hyperparams,
                                                        n_running_time=n_running_time,
                                                        results_path=result_path)

                    error_params.append(err_ratio_temp[0])

                    error_params = np.array(error_params)
                    error_params = np.expand_dims(error_params, axis=0)
                    errors_params = np.concatenate([errors_params, error_params], axis=0)
                    print('|--- Errors-Hyperparams:')
                    print(errors_params)
                    print('--------------------------------------------------------------------------')
                    np.savetxt(fname=result_path + '[Errors_2.00-2.50]Epoch_fw_%i_Epoch_bw_%i_n_running_time_%i.csv' %
                                     (with_epoch_fw, with_epoch_bw, n_running_time),
                               X=errors_params,
                               delimiter=',')

    return
