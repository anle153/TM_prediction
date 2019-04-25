import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from Models.ConvLSTM_model import ConvLSTM
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_3d, generator_convlstm_train_data, \
    generator_convlstm_train_data_fix_ratio, create_offline_convlstm_data_fix_ratio
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


def ims_tm_prediction(init_data_labels, forward_model, backward_model):
    multi_steps_tm = np.zeros(shape=(init_data_labels.shape[0] + Config.IMS_STEP,
                                     init_data_labels.shape[1], init_data_labels.shape[2], init_data_labels.shape[3]))

    multi_steps_tm[0:init_data_labels.shape[0], :, :, :] = init_data_labels

    for ts_ahead in range(Config.IMS_STEP):
        rnn_input = multi_steps_tm[-Config.LSTM_STEP:, :, :, :]  # shape(timesteps, od, od , 2)

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

        sampling = np.zeros(shape=(Config.CNN_WIDE, Config.CNN_HIGH, 1))

        # Calculating the true value for the TM
        new_input = predict_tm

        # Concaternating the new tm to the final results
        # Shape = (12, 12, 2)
        new_input = np.concatenate([np.expand_dims(new_input, axis=2), sampling], axis=2)
        multi_steps_tm[ts_ahead + Config.LSTM_STEP] = new_input  # Shape = (timestep, 12, 12, 2)

    return multi_steps_tm[-1, :, :, 0]


def predict_fwbw_conv_lstm(initial_data, test_data, forward_model, backward_model):
    init_labels = np.ones((initial_data.shape[0], initial_data.shape[1], initial_data.shape[2]))

    tm_labels = np.zeros(
        shape=(initial_data.shape[0] + test_data.shape[0], initial_data.shape[1], initial_data.shape[2], 2))
    tm_labels[0:initial_data.shape[0], :, :, 0] = initial_data
    tm_labels[0:init_labels.shape[0], :, :, 1] = init_labels

    ims_tm = np.zeros(shape=(test_data.shape[0] - Config.IMS_STEP + 1, test_data.shape[1], test_data.shape[2]))

    for ts in tqdm(range(test_data.shape[0])):

        if ts <= test_data.shape[0] - Config.IMS_STEP:
            ims_tm[ts] = ims_tm_prediction(init_data_labels=tm_labels[ts:ts + Config.LSTM_STEP, :, :, :],
                                           forward_model=forward_model,
                                           backward_model=backward_model)

        rnn_input = tm_labels[ts:(ts + Config.LSTM_STEP), :, :, :]  # shape(timesteps, od, od , 2)

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
        corrected_data = test_data[ts, :, :]
        ground_truth = corrected_data * sampling

        # Calculating the true value for the TM
        new_tm = pred_tm + ground_truth

        # Concaternating the new tm to the final results
        new_tm = np.concatenate([np.expand_dims(new_tm, axis=2), np.expand_dims(sampling, axis=2)], axis=2)
        tm_labels[ts + Config.LSTM_STEP] = new_tm  # Shape = (timestep, 12, 12, 2)

    return tm_labels[Config.LSTM_STEP:, :, :, :], ims_tm


def build_model(args, input_shape):
    print('|--- Build models.')
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


def load_trained_models(args, input_shape, fw_ckp, bw_ckp):
    print('|--- Load trained model')
    fw_net, bw_net = build_model(args, input_shape)
    fw_net.model.load_weights(fw_net.checkpoints_path + "weights-{:02d}.hdf5".format(fw_ckp))
    bw_net.model.load_weights(bw_net.checkpoints_path + "weights-{:02d}.hdf5".format(bw_ckp))

    return fw_net, bw_net


def train_fwbw_conv_lstm(data, args):
    print('|-- Run model training.')
    gpu = args.gpu

    data_name = args.data_name
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    with tf.device('/device:GPU:{}'.format(gpu)):

        print('|--- Splitting train-test set.')
        train_data, valid_data, test_data = prepare_train_valid_test_3d(data=data, day_size=day_size)
        print('|--- Normalizing the train set.')
        mean_train = np.mean(train_data)
        std_train = np.std(train_data)
        train_data_normalized = (train_data - mean_train) / std_train
        valid_data_normalized = (valid_data - mean_train) / std_train
        # test_data = (test_data - mean_train) / std_train

        input_shape = (Config.LSTM_STEP,
                       Config.CNN_WIDE, Config.CNN_HIGH, Config.CNN_CHANNEL)

        fw_net, bw_net = build_model(args, input_shape)

        # -------------------------------- Create offline training and validating dataset ------------------------------
        if not os.path.isfile(fw_net.saving_path + 'trainX_fw.npy'):
            trainX_fw, trainY_fw = create_offline_convlstm_data_fix_ratio(train_data_normalized,
                                                                               input_shape, Config.MON_RAIO, 0.5)
            np.save(fw_net.saving_path + 'trainX_fw.npy', trainX_fw)
            np.save(fw_net.saving_path + 'trainY_fw.npy', trainY_fw)
        else:
            trainX_fw = np.load(fw_net.saving_path + 'trainX_fw.npy')
            trainY_fw = np.load(fw_net.saving_path + 'trainY_fw.npy')

        if not os.path.isfile(fw_net.saving_path + 'validX_fw.npy'):
            validX_fw, validY_fw = create_offline_convlstm_data_fix_ratio(valid_data_normalized,
                                                                          input_shape, Config.MON_RAIO, 0.5)
            np.save(fw_net.saving_path + 'validX_fw.npy', validX_fw)
            np.save(fw_net.saving_path + 'validY_fw.npy', validY_fw)
        else:
            validX_fw = np.load(fw_net.saving_path + 'validX_fw.npy')
            validY_fw = np.load(fw_net.saving_path + 'validY_fw.npy')
        # --------------------------------------------------------------------------------------------------------------

        if Config.MON_RAIO is not None:
            generator_train_data = generator_convlstm_train_data_fix_ratio
        else:
            generator_train_data = generator_convlstm_train_data

        if os.path.isfile(path=fw_net.checkpoints_path + 'weights-{:02d}-0.00.hdf5'.format(Config.N_EPOCH)):
            print('|--- Forward model exist!')
            fw_net.load_model_from_check_point(_from_epoch=Config.FW_BEST_CHECKPOINT)
        else:
            print('|--- Compile model. Saving path %s --- ' % fw_net.saving_path)

            # Load model check point
            from_epoch = fw_net.load_model_from_check_point()
            if from_epoch > 0:
                print('|--- Continue training forward model from epoch %i --- ' % from_epoch)
                training_fw_history = fw_net.model.fit(x=trainX_fw,
                                                       y=trainY_fw,
                                                       batch_size=Config.BATCH_SIZE,
                                                       epochs=Config.N_EPOCH,
                                                       callbacks=fw_net.callbacks_list,
                                                       validation_data=(validX_fw, validY_fw),
                                                       shuffle=True,
                                                       initial_epoch=from_epoch)
            else:
                print('|--- Training new forward model.')

                training_fw_history = fw_net.model.fit(x=trainX_fw,
                                                       y=trainY_fw,
                                                       batch_size=Config.BATCH_SIZE,
                                                       epochs=Config.N_EPOCH,
                                                       callbacks=fw_net.callbacks_list,
                                                       validation_data=(validX_fw, validY_fw),
                                                       shuffle=True)

            # Plot the training history
            if training_fw_history is not None:
                fw_net.plot_training_history(training_fw_history)

        # -------------------------------- Create offline training and validating dataset ------------------------------

        train_data_bw_normalized = np.flip(train_data_normalized, axis=0)
        valid_data_bw_normalized = np.flip(valid_data_normalized, axis=0)

        if not os.path.isfile(bw_net.saving_path + 'trainX_bw.npy'):
            trainX_bw, trainY_bw = create_offline_convlstm_data_fix_ratio(train_data_bw_normalized,
                                                                          input_shape, Config.MON_RAIO, 0.5)
            np.save(bw_net.saving_path + 'trainX_bw.npy', trainX_bw)
            np.save(bw_net.saving_path + 'trainY_bw.npy', trainY_bw)
        else:
            trainX_bw = np.load(bw_net.saving_path + 'trainX_bw.npy')
            trainY_bw = np.load(bw_net.saving_path + 'trainY_bw.npy')

        if not os.path.isfile(bw_net.saving_path + 'validX_bw.npy'):
            validX_bw, validY_bw = create_offline_convlstm_data_fix_ratio(valid_data_bw_normalized,
                                                                          input_shape, Config.MON_RAIO, 0.5)
            np.save(bw_net.saving_path + 'validX_bw.npy', validX_bw)
            np.save(bw_net.saving_path + 'validY_bw.npy', validY_bw)
        else:
            validX_bw = np.load(bw_net.saving_path + 'validX_bw.npy')
            validY_bw = np.load(bw_net.saving_path + 'validY_bw.npy')

        # --------------------------------------------Training bw model-------------------------------------------------
        if os.path.isfile(path=bw_net.saving_path + 'weights-%i-0.00.hdf5' % Config.N_EPOCH):
            print('|--- Backward model exist!')
            bw_net.load_model_from_check_point(_from_epoch=Config.BW_BEST_CHECKPOINT)
        else:
            print('|---Compile model. Saving path: %s' % bw_net.saving_path)
            from_epoch_bw = bw_net.load_model_from_check_point()
            if from_epoch_bw > 0:

                training_bw_history = bw_net.model.fit(x=trainX_bw,
                                                       y=trainY_bw,
                                                       batch_size=Config.BATCH_SIZE,
                                                       epochs=Config.N_EPOCH,
                                                       callbacks=bw_net.callbacks_list,
                                                       validation_data=(validX_bw, validY_bw),
                                                       shuffle=True,
                                                       initial_epoch=from_epoch_bw)

            else:
                print('|--- Training new backward model.')

                training_bw_history = bw_net.model.fit(x=trainX_bw,
                                                       y=trainY_bw,
                                                       batch_size=Config.BATCH_SIZE,
                                                       epochs=Config.N_EPOCH,
                                                       callbacks=bw_net.callbacks_list,
                                                       validation_data=(validX_bw, validY_bw),
                                                       shuffle=True)
            if training_bw_history is not None:
                bw_net.plot_training_history(training_bw_history)

        print('---------------------------------FW_NET SUMMARY---------------------------------')
        print(fw_net.model.summary())
        print('---------------------------------BW_NET SUMMARY---------------------------------')
        print(bw_net.model.summary())

    return


def ims_tm_ytrue(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.IMS_STEP + 1, test_data.shape[1], test_data.shape[2]))

    for i in range(Config.IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.IMS_STEP + 1] = test_data[i]

    return ims_test_set


def test_fwbw_conv_lstm(data, args):
    print('|-- Run model testing.')
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_valid_test_3d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    # train_data_normalized = (train_data - mean_train) / std_train
    valid_data_normalized = (valid_data - mean_train) / std_train
    test_data_normalized = (test_data - mean_train) / std_train

    input_shape = (Config.LSTM_STEP,
                   Config.CNN_WIDE, Config.CNN_HIGH, Config.CNN_CHANNEL)

    fw_net, bw_net = load_trained_models(args, input_shape, Config.FW_BEST_CHECKPOINT, Config.BW_BEST_CHECKPOINT)

    results_summary = pd.read_csv(Config.RESULTS_PATH + 'sample_results.csv')

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    measured_matrix_ims = np.zeros((test_data.shape[0] - Config.IMS_STEP + 1, Config.CNN_WIDE, Config.CNN_HIGH))

    for i in range(Config.TESTING_TIME):
        print('|--- Run time {}'.format(i))

        tm_labels, ims_tm = predict_fwbw_conv_lstm(
            initial_data=valid_data_normalized[-Config.LSTM_STEP:, :, :],
            test_data=test_data_normalized,
            forward_model=fw_net.model,
            backward_model=bw_net.model)

        pred_tm = tm_labels[:, :, :, 0]
        measured_matrix = tm_labels[:, :, :, 1]

        pred_tm = pred_tm * std_train + mean_train

        err.append(error_ratio(y_true=test_data, y_pred=pred_tm, measured_matrix=measured_matrix))
        r2_score.append(calculate_r2_score(y_true=test_data, y_pred=pred_tm))
        rmse.append(calculate_rmse(y_true=test_data, y_pred=pred_tm))

        ims_tm = ims_tm * std_train + mean_train
        ims_ytrue = ims_tm_ytrue(test_data=test_data)

        err_ims.append(error_ratio(y_pred=ims_tm,
                                   y_true=ims_ytrue,
                                   measured_matrix=measured_matrix_ims))

        r2_score_ims.append(calculate_r2_score(y_true=ims_ytrue, y_pred=ims_tm))
        rmse_ims.append(calculate_rmse(y_true=ims_ytrue, y_pred=ims_tm))

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
