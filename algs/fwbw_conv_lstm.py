import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from Models.ConvLSTM_model import ConvLSTM
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_3d, generator_convlstm_train_data, \
    generator_convlstm_train_data_fix_ratio
from common.error_utils import calculate_consecutive_loss_3d, recovery_loss_3d, error_ratio, calculate_r2_score, \
    calculate_rmse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def calculate_flows_weights_3d(rnn_input, rl_forward, rl_backward, measured_matrix):
    eps = 10e-5

    cl = calculate_consecutive_loss_3d(measured_matrix).astype(float)

    flows_stds = np.std(rnn_input, axis=0)

    cl_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(cl)
    flows_stds_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(flows_stds)
    rl_forward_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(rl_forward)
    rl_backward_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(rl_backward)

    w = 1 / (rl_forward_scaled * Config.HYPERPARAMS[0] +
             rl_backward_scaled * Config.HYPERPARAMS[1] +
             cl_scaled * Config.HYPERPARAMS[2] +
             flows_stds_scaled * Config.HYPERPARAMS[3])

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
    m = int(Config.MON_RAIO * rnn_input.shape[1] * rnn_input.shape[2])

    w = w.flatten()
    sorted_idx_w = np.argpartition(w, m)
    sampling[sorted_idx_w[:m]] = 1

    sampling = np.expand_dims(sampling, axis=0)

    sampling = np.reshape(sampling, newshape=(rnn_input.shape[1], rnn_input.shape[2]))

    return sampling.astype(bool)


def calculate_updated_weights_3d(measured_block, forward_loss, backward_loss):
    labels = measured_block.astype(int)

    measured_count = np.sum(labels, axis=0).astype(float)
    _eta = measured_count / Config.LSTM_STEP

    # _eta[_eta == 0.0] = eps

    alpha = 1 - _eta  # shape = (od, od)
    alpha = np.tile(np.expand_dims(alpha, axis=0), (Config.LSTM_STEP, 1, 1))

    # Calculate rho
    rho = np.empty((0, measured_block.shape[1], measured_block.shape[1]))
    mu = np.empty((0, measured_block.shape[1], measured_block.shape[1]))
    for j in range(0, Config.LSTM_STEP):
        _mu = np.expand_dims((np.sum(measured_block[:(j + 1), :, :], axis=0)) / float(j + 1), axis=0)
        mu = np.concatenate([mu, _mu], axis=0)

        _rho = np.expand_dims((np.sum(measured_block[j:, :, :], axis=0)) / float(Config.LSTM_STEP - j), axis=0)
        rho = np.concatenate([rho, _rho], axis=0)

    forward_loss = np.tile(np.expand_dims(forward_loss, axis=0), (Config.LSTM_STEP, 1, 1))
    backward_loss = np.tile(np.expand_dims(backward_loss, axis=0), (Config.LSTM_STEP, 1, 1))

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

    tm_labels[(-Config.LSTM_STEP + 1):-1, :, :, 0] = \
        tm_labels[(-Config.LSTM_STEP + 1):-1, :, :, 0] * sampling_measured_matrix + bidirect_rnn_pred_value

    return tm_labels


def ims_tm_prediction(tm_labels, forward_model, backward_model,
                      iterated_multi_steps_tm):
    multi_steps_tm = np.copy(tm_labels[-Config.LSTM_STEP:, :, :, :])

    for ts_ahead in range(Config.IMS_STEP):
        rnn_input = np.copy(multi_steps_tm[-Config.LSTM_STEP:, :, :, :])  # shape(timesteps, od, od , 2)

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

    multi_steps_tm = multi_steps_tm[Config.LSTM_STEP:, :, :, 0]
    multi_steps_tm = np.expand_dims(multi_steps_tm, axis=0)

    iterated_multi_steps_tm = np.concatenate([iterated_multi_steps_tm, multi_steps_tm], axis=0)

    return iterated_multi_steps_tm


def predict_fwbw_conv_lstm(test_data, forward_model, backward_model):
    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = np.copy(test_data[0:Config.LSTM_STEP, :, :])  # rnn input shape = (timeslot, od, od)
    # Results TM
    # The TF array for random choosing the measured flows
    labels = np.ones((rnn_input.shape[0], rnn_input.shape[1], rnn_input.shape[2]))
    tf = 7.2

    tm_labels = np.concatenate([np.expand_dims(rnn_input, axis=3), np.expand_dims(labels, axis=3)], axis=3)

    day_size = 24 * (60 / 5)
    iterated_multi_steps_tm = np.zeros(shape=(test_data - Config.LSTM_STEP - Config.IMS_STEP, 12, 12))

    # Predict the TM from time slot look_back
    for ts in range(0, test_data.shape[0] - Config.LSTM_STEP, 1):
        date = int(ts / day_size)
        # print ('--- Predict at timeslot %i ---' % tslot)

        if ts < test_data.shape[0] - Config.LSTM_STEP - Config.IMS_STEP:
            iterated_multi_steps_tm = ims_tm_prediction(
                tm_labels=tm_labels,
                forward_model=forward_model,
                backward_model=backward_model,
                iterated_multi_steps_tm=iterated_multi_steps_tm)

        rnn_input = np.copy(tm_labels[ts:(ts + Config.LSTM_STEP), :, :, :])  # shape(timesteps, od, od , 2)

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
        corrected_data = np.copy(test_data[ts + Config.LSTM_STEP, :, :])
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
                      cnn_layers=Config.CNN_LAYERS,
                      a_filters=Config.FILTERS,
                      a_strides=Config.STRIDES,
                      dropouts=Config.DROPOUTS,
                      kernel_sizes=Config.KERNEL_SIZE,
                      rnn_dropouts=Config.RNN_DROPOUTS,
                      alg_name=alg_name,
                      tag=tag,
                      check_point=True,
                      saving_path=Config.MODEL_SAVE + '{}-{}-{}/fw/'.format(data_name, alg_name, tag))

    # CNN_BRNN backward model
    bw_net = ConvLSTM(input_shape=input_shape,
                      cnn_layers=Config.CNN_LAYERS,
                      a_filters=Config.FILTERS,
                      a_strides=Config.STRIDES,
                      dropouts=Config.DROPOUTS,
                      kernel_sizes=Config.KERNEL_SIZE,
                      rnn_dropouts=Config.RNN_DROPOUTS,
                      alg_name=alg_name,
                      tag=tag,
                      check_point=True,
                      saving_path=Config.MODEL_SAVE + '{}-{}-{}/bw/'.format(data_name, alg_name, tag))

    return fw_net, bw_net


def train_fwbw_conv_lstm(data, args):
    gpu = args.gpu

    if gpu is None:
        gpu = 0

    with tf.device('/device:GPU:{}'.format(gpu)):

        print('|--- Splitting train-test set.')
        train_data, valid_data, test_data = prepare_train_valid_test_3d(data=data)
        print('|--- Normalizing the train set.')
        mean_train = np.mean(train_data)
        std_train = np.std(train_data)
        train_data = (train_data - mean_train) / std_train
        valid_data = (valid_data - mean_train) / std_train
        test_data = (test_data - mean_train) / std_train

        print("|--- Create FWBW_CONVLSTM model.")
        input_shape = (Config.LSTM_STEP,
                       Config.CNN_WIDE, Config.CNN_HIGH, Config.CNN_CHANNEL)

        fw_net, bw_net = build_model(args, input_shape)

        if Config.MON_RAIO is not None:
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
                                         Config.MON_RAIO,
                                         0.5,
                                         Config.BATCH_SIZE),
                    epochs=Config.N_EPOCH,
                    steps_per_epoch=Config.NUM_ITER,
                    initial_epoch=from_epoch,
                    validation_data=generator_convlstm_train_data(valid_data, input_shape, Config.MON_RAIO,
                                                                  0.5,
                                                                  Config.BATCH_SIZE),
                    validation_steps=int(Config.NUM_ITER * 0.2),
                    callbacks=fw_net.callbacks_list,
                    use_multiprocessing=True,
                    workers=2,
                    max_queue_size=1024)
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
                    validation_data=generator_convlstm_train_data(valid_data, input_shape, Config.MON_RAIO,
                                                                  0.5,
                                                                  Config.BATCH_SIZE),
                    validation_steps=int(Config.NUM_ITER * 0.2),
                    callbacks=fw_net.callbacks_list,
                    use_multiprocessing=True,
                    workers=2,
                    max_queue_size=1028)

            # Plot the training history
            if training_fw_history is not None:
                fw_net.plot_training_history(training_fw_history)

        train_data_bw = np.flip(train_data, axis=0)
        vallid_data_bw = np.flip(valid_data, axis=0)

        # Training cnn_brnn backward model
        if os.path.isfile(path=bw_net.saving_path + 'weights-%i-0.00.hdf5' % Config.N_EPOCH):
            print('|--- Backward model exist!')
            bw_net.load_model_from_check_point(_from_epoch=Config.BEST_CHECKPOINT, weights_file_type='hdf5')
        else:
            print('|---Compile model. Saving path: %s' % bw_net.saving_path)
            # Load model from check point
            from_epoch_backward = bw_net.load_model_from_check_point()
            if from_epoch_backward > 0:

                bw_net.model.compile(loss='mean_squared_error', optimizer='adam',
                                     metrics=['mse', 'mae', 'accuracy'])

                training_bw_history = bw_net.model.fit_generator(
                    generator_train_data(train_data_bw,
                                         input_shape,
                                         Config.MON_RAIO,
                                         0.5,
                                         Config.BATCH_SIZE),
                    epochs=Config.N_EPOCH,
                    steps_per_epoch=Config.NUM_ITER,
                    initial_epoch=from_epoch_backward,
                    validation_data=generator_convlstm_train_data(vallid_data_bw,
                                                                  input_shape,
                                                                  Config.MON_RAIO,
                                                                  0.5,
                                                                  Config.BATCH_SIZE),
                    validation_steps=int(Config.NUM_ITER * 0.2),
                    callbacks=bw_net.callbacks_list,
                    use_multiprocessing=True,
                    workers=2,
                    max_queue_size=1028)

            else:
                print('|--- Training new backward model.')

                bw_net.model.compile(loss='mean_squared_error', optimizer='adam',
                                     metrics=['mse', 'mae', 'accuracy'])

                training_bw_history = bw_net.model.fit_generator(
                    generator_train_data(train_data_bw,
                                         input_shape,
                                         Config.MON_RAIO,
                                         0.5,
                                         Config.BATCH_SIZE),
                    epochs=Config.N_EPOCH,
                    steps_per_epoch=Config.NUM_ITER,
                    validation_data=generator_convlstm_train_data(vallid_data_bw,
                                                                  input_shape,
                                                                  Config.MON_RAIO,
                                                                  0.5,
                                                                  Config.BATCH_SIZE),
                    validation_steps=int(Config.NUM_ITER * 0.2),
                    callbacks=bw_net.callbacks_list,
                    use_multiprocessing=True,
                    workers=2,
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
        shape=(test_set.shape[0] - Config.LSTM_STEP - Config.IMS_STEP,
               Config.IMS_STEP, test_set.shape[1],
               test_set.shape[2]))

    for ts in range(test_set.shape[0] - Config.LSTM_STEP - Config.IMS_STEP):
        multi_step_test_set = np.copy(
            test_set[(ts + Config.LSTM_STEP): (ts + Config.LSTM_STEP + Config.IMS_STEP), :, :])
        iterated_multi_step_test_set[ts] = multi_step_test_set

    return iterated_multi_step_test_set


def test_fwbw_conv_lstm(data, args):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_valid_test_3d(data=data)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data = (train_data - mean_train) / std_train
    valid_data = (valid_data - mean_train) / std_train
    test_data_normalized = (test_data - mean_train) / std_train

    print("|--- Create FWBW_CONVLSTM model.")
    input_shape = (Config.LSTM_STEP,
                   Config.CNN_WIDE, Config.CNN_HIGH, Config.CNN_CHANNEL)

    fw_net, bw_net = build_model(args, input_shape)

    results_summary = pd.read_csv(Config.RESULTS_PATH + 'sample_results.csv')

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
        rmse.append(calculate_rmse(y_true=test_data_normalized, y_pred=np.copy(pred_tm)))

        iterated_multi_steps_tm = iterated_multi_steps_tm * std_train + mean_train

        iterated_multi_step_test_set = calculate_lstm_iterated_multi_step_tm_prediction_errors(test_set=test_data)

        measured_matrix = np.zeros(shape=iterated_multi_step_test_set.shape)
        err_ims.append(error_ratio(y_pred=iterated_multi_steps_tm,
                                   y_true=iterated_multi_step_test_set,
                                   measured_matrix=measured_matrix))

        r2_score_ims.append(calculate_r2_score(y_true=iterated_multi_step_test_set, y_pred=iterated_multi_steps_tm))
        rmse_ims.append(calculate_rmse(y_true=iterated_multi_step_test_set, y_pred=iterated_multi_steps_tm))

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
