import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from Models.FWBW_CONV_LSTM import FWBW_CONV_LSTM
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_2d, create_offline_fwbw_conv_lstm_data_fix_ratio, \
    data_scalling
from common.error_utils import calculate_consecutive_loss_3d, recovery_loss_3d, error_ratio, calculate_r2_score, \
    calculate_rmse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def plot_test_data(prefix, raw_data, pred_fw, pred_bw, current_data):
    saving_path = Config.RESULTS_PATH + 'plot_fwbw_conv_lstm/'

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    from matplotlib import pyplot as plt
    for flow_x in range(raw_data.shape[1]):
        for flow_y in range(raw_data.shape[2]):
            plt.plot(raw_data[:, flow_x, flow_y], label='Actual')
            plt.plot(pred_fw[:, flow_x, flow_y], label='Pred_fw')
            plt.plot(pred_bw[:, flow_x, flow_y], label='Pred_bw')
            plt.plot(current_data[:, flow_x, flow_y], label='Current_pred')

            plt.legend()
            plt.savefig(saving_path + '{}_flow_{:02d}-{:02d}.png'.format(prefix, flow_x, flow_y))
            plt.close()


def calculate_flows_weights_3d(rnn_input, rl_forward, rl_backward, measured_matrix):
    eps = 10e-5

    cl = calculate_consecutive_loss_3d(measured_matrix).astype(float)

    flows_stds = np.std(rnn_input, axis=0)

    cl_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(cl)
    flows_stds_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(flows_stds)
    rl_forward_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(rl_forward)
    rl_backward_scaled = MinMaxScaler(feature_range=(eps, 1.0)).fit_transform(rl_backward)

    w = 1 / (rl_forward_scaled * Config.FWBW_CONV_LSTM_HYPERPARAMS[0] +
             rl_backward_scaled * Config.FWBW_CONV_LSTM_HYPERPARAMS[1] +
             cl_scaled * Config.FWBW_CONV_LSTM_HYPERPARAMS[2] +
             flows_stds_scaled * Config.FWBW_CONV_LSTM_HYPERPARAMS[3])

    return w


def set_measured_flow_3d(rnn_input, labels, forward_pred, backward_pred):
    rl_forward, rl_backward = calculate_forward_backward_loss_3d(measured_block=labels,
                                                                 pred_forward=forward_pred,
                                                                 pred_backward=backward_pred,
                                                                 rnn_input=rnn_input)

    w = calculate_flows_weights_3d(rnn_input=rnn_input,
                                   rl_forward=rl_forward,
                                   rl_backward=rl_backward,
                                   measured_matrix=labels)

    sampling = np.zeros(shape=(rnn_input.shape[1] * rnn_input.shape[2]))
    m = int(Config.FWBW_CONV_LSTM_MON_RAIO * rnn_input.shape[1] * rnn_input.shape[2])

    w = w.flatten()
    sorted_idx_w = np.argpartition(w, m)
    sampling[sorted_idx_w[:m]] = 1

    sampling = np.expand_dims(sampling, axis=0)

    sampling = np.reshape(sampling, newshape=(rnn_input.shape[1], rnn_input.shape[2]))

    return sampling


def calculate_updated_weights_3d(measured_block, forward_loss, backward_loss):
    measured_count = np.sum(measured_block, axis=0).astype(float)
    _eta = measured_count / Config.FWBW_CONV_LSTM_STEP

    alpha = 1 - _eta  # shape = (od, od)
    alpha = np.tile(np.expand_dims(alpha, axis=0), (Config.FWBW_CONV_LSTM_STEP, 1, 1))

    # Calculate rho
    rho = np.zeros((Config.FWBW_CONV_LSTM_STEP, measured_block.shape[1], measured_block.shape[1]))
    mu = np.zeros((Config.FWBW_CONV_LSTM_STEP, measured_block.shape[1], measured_block.shape[1]))
    for j in range(0, Config.FWBW_CONV_LSTM_STEP):
        _mu = np.sum(measured_block[:(j + 1)], axis=0) / float(j + 1)
        mu[j] = _mu

        _rho = np.sum(measured_block[j:, :, :], axis=0) / float(Config.FWBW_CONV_LSTM_STEP - j)
        rho[j] = _rho

    forward_loss = np.tile(np.expand_dims(forward_loss, axis=0), (Config.FWBW_CONV_LSTM_STEP, 1, 1))
    backward_loss = np.tile(np.expand_dims(backward_loss, axis=0), (Config.FWBW_CONV_LSTM_STEP, 1, 1))

    beta = (backward_loss + mu) * (1.0 - alpha) / (forward_loss + backward_loss + mu + rho)

    gamma = (forward_loss + rho) * (1.0 - alpha) / (forward_loss + backward_loss + mu + rho)

    return alpha[1:-1], beta[1:-1], gamma[1:-1]


def calculate_forward_backward_loss_3d(measured_block, pred_forward, pred_backward, rnn_input):
    eps = 10e-8

    rl_forward = recovery_loss_3d(rnn_input=rnn_input[1:], rnn_updated=pred_forward[:-1],
                                  measured_matrix=measured_block[1:])
    rl_forward[rl_forward == 0] = eps

    rl_backward = recovery_loss_3d(rnn_input=rnn_input[:-1], rnn_updated=pred_backward[1:],
                                   measured_matrix=measured_block[:-1])
    rl_backward[rl_backward == 0] = eps

    return rl_forward, rl_backward


def updating_historical_data_3d(rnn_input, pred_forward, pred_backward, labels):
    rnn_input = np.copy(rnn_input)
    measured_block = np.copy(labels)

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

    # updated_rnn_input = considered_rnn_input * alpha + considered_forward * beta + considered_backward * gamma
    updated_rnn_input = (considered_rnn_input + considered_forward + considered_backward) / 3.0

    sampling_measured_matrix = measured_block[1:-1]
    inv_sampling_measured_matrix = 1 - sampling_measured_matrix

    # if ts == 20:
    #     print('Alpha: {}'.format(alpha[:, 0, 3]))
    #     print('Beta: {}'.format(beta[:, 0, 3]))
    #     print('Gamma: {}'.format(gamma[:, 0, 3]))

    rnn_pred_value = updated_rnn_input * inv_sampling_measured_matrix

    return rnn_pred_value


def ims_tm_prediction(init_data, init_labels, model):
    multi_steps_tm = np.zeros(shape=(init_data.shape[0] + Config.FWBW_CONV_LSTM_IMS_STEP,
                                     init_data.shape[1], init_data.shape[2]))

    multi_steps_tm[0:init_data.shape[0]] = init_data

    labels = np.zeros(shape=(init_labels.shape[0] + Config.FWBW_CONV_LSTM_IMS_STEP,
                             init_labels.shape[1], init_labels.shape[2]))
    labels[0:init_labels.shape[0]] = init_labels

    for ts_ahead in range(Config.FWBW_CONV_LSTM_IMS_STEP):
        rnn_input = np.zeros(
            shape=(Config.FWBW_CONV_LSTM_STEP, Config.FWBW_CONV_LSTM_WIDE, Config.FWBW_CONV_LSTM_HIGH, 2))

        rnn_input[:, :, :, 0] = multi_steps_tm[ts_ahead:(ts_ahead + Config.FWBW_CONV_LSTM_STEP)]
        rnn_input[:, :, :, 1] = labels[ts_ahead:(ts_ahead + Config.FWBW_CONV_LSTM_STEP)]

        rnn_input = np.expand_dims(rnn_input, axis=0)  # shape(1, timesteps, od, od , 2)

        # Prediction results from forward network
        predictX, predictX_backward = model.predict(rnn_input)  # shape(1, timesteps, od, od , 1)

        predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, od, od , 1)
        predictX = np.reshape(predictX, newshape=(predictX.shape[0],
                                                  Config.FWBW_CONV_LSTM_WIDE,
                                                  Config.FWBW_CONV_LSTM_HIGH))

        predict_tm = np.copy(predictX[-1])

        predictX_backward = np.squeeze(predictX_backward, axis=0)  # shape(timesteps, #nflows)

        # Flipping the backward prediction
        predictX_backward = np.flip(predictX_backward, axis=0)
        predictX_backward = np.reshape(predictX_backward, newshape=(predictX_backward.shape[0],
                                                                    Config.FWBW_CONV_LSTM_WIDE,
                                                                    Config.FWBW_CONV_LSTM_HIGH))

        # Correcting the imprecise input data
        rnn_pred_value = updating_historical_data_3d(
            rnn_input=multi_steps_tm[ts_ahead:ts_ahead + Config.FWBW_CONV_LSTM_STEP],
            pred_forward=predictX,
            pred_backward=predictX_backward,
            labels=labels[ts_ahead:ts_ahead + Config.FWBW_CONV_LSTM_STEP])

        multi_steps_tm[(ts_ahead + 1):(ts_ahead + Config.FWBW_CONV_LSTM_STEP - 1)] = \
            multi_steps_tm[(ts_ahead + 1):(ts_ahead + Config.FWBW_CONV_LSTM_STEP - 1)] * \
            labels[(ts_ahead + 1):(ts_ahead + Config.FWBW_CONV_LSTM_STEP - 1)] + \
            rnn_pred_value

        multi_steps_tm[ts_ahead + Config.FWBW_CONV_LSTM_STEP] = predict_tm

    return multi_steps_tm[-1]


def predict_fwbw_conv_lstm(initial_data, test_data, model):
    tf_a = np.array([1.0, 0.0])

    tm_labels = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    tm_labels[0:initial_data.shape[0], :, :] = initial_data

    _tm_labels = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    _tm_labels[0:initial_data.shape[0], :, :] = initial_data

    labels = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    labels[0:initial_data.shape[0], :, :] = np.ones(shape=initial_data.shape)

    ims_tm = np.zeros(
        shape=(test_data.shape[0] - Config.FWBW_CONV_LSTM_IMS_STEP + 1, test_data.shape[1], test_data.shape[2]))

    raw_data = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1], test_data.shape[2]))

    raw_data[0:initial_data.shape[0]] = initial_data
    raw_data[initial_data.shape[0]:] = test_data

    for ts in tqdm(range(test_data.shape[0])):

        if Config.FWBW_CONV_LSTM_IMS and (ts <= test_data.shape[0] - Config.FWBW_CONV_LSTM_IMS_STEP):
            ims_tm[ts] = ims_tm_prediction(init_data=tm_labels[ts:ts + Config.FWBW_CONV_LSTM_STEP],
                                           init_labels=labels[ts:ts + Config.FWBW_CONV_LSTM_STEP],
                                           model=model)
        rnn_input = np.zeros(
            shape=(Config.FWBW_CONV_LSTM_STEP, Config.FWBW_CONV_LSTM_WIDE, Config.FWBW_CONV_LSTM_HIGH, 2))

        rnn_input[:, :, :, 0] = tm_labels[ts:(ts + Config.FWBW_CONV_LSTM_STEP)]
        rnn_input[:, :, :, 1] = labels[ts:(ts + Config.FWBW_CONV_LSTM_STEP)]

        rnn_input = np.expand_dims(rnn_input, axis=0)

        # Prediction results from forward network
        predictX, predictX_backward = model.predict(rnn_input)  # shape(1, timesteps, od, od , 1)
        predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, #nflows)
        predictX = np.reshape(predictX, newshape=(predictX.shape[0], test_data.shape[1], test_data.shape[2]))

        predict_tm = np.copy(predictX[-1])

        predictX_backward = np.squeeze(predictX_backward, axis=0)  # shape(timesteps, #nflows)

        # Flipping the backward prediction
        predictX_backward = np.flip(predictX_backward, axis=0)
        predictX_backward = np.reshape(predictX_backward,
                                       newshape=(predictX_backward.shape[0], test_data.shape[1], test_data.shape[2]))

        # if ts == 20:
        #     plot_test_data('Before_update', raw_data[ts + 1:ts + Config.FWBW_CONV_LSTM_STEP - 1],
        #                    predictX[:-2],
        #                    predictX_backward[2:],
        #                    tm_labels[ts + 1:ts + Config.FWBW_CONV_LSTM_STEP - 1])

        # before_ = np.copy(tm_labels[ts + 1:ts + Config.FWBW_CONV_LSTM_STEP - 1])

        # _err_1 = error_ratio(y_pred=tm_labels[ts:ts + Config.FWBW_CONV_LSTM_STEP],
        #                      y_true=raw_data[ts:ts + Config.FWBW_CONV_LSTM_STEP],
        #                      measured_matrix=labels[ts: ts + Config.FWBW_CONV_LSTM_STEP])

        # Correcting the imprecise input data
        rnn_pred_value = updating_historical_data_3d(rnn_input=tm_labels[ts:ts + Config.FWBW_CONV_LSTM_STEP],
                                                     pred_forward=predictX,
                                                     pred_backward=predictX_backward,
                                                     labels=labels[ts:ts + Config.FWBW_CONV_LSTM_STEP])

        tm_labels[(ts + 1):(ts + Config.FWBW_CONV_LSTM_STEP - 1)] = \
            tm_labels[(ts + 1):(ts + Config.FWBW_CONV_LSTM_STEP - 1)] * \
            labels[(ts + 1):(ts + Config.FWBW_CONV_LSTM_STEP - 1)] + \
            rnn_pred_value

        if Config.FWBW_CONV_LSTM_RANDOM_ACTION:
            sampling = np.random.choice(tf_a, size=(test_data.shape[1], test_data.shape[2]),
                                        p=(Config.FWBW_CONV_LSTM_MON_RAIO, 1 - Config.FWBW_CONV_LSTM_MON_RAIO))
        else:
            sampling = set_measured_flow_3d(rnn_input=tm_labels[ts:ts + Config.FWBW_CONV_LSTM_STEP],
                                            labels=labels[ts:ts + Config.FWBW_CONV_LSTM_STEP],
                                            forward_pred=predictX,
                                            backward_pred=predictX_backward)

        # Selecting next monitored flows randomly
        inv_sampling = 1 - sampling

        pred_tm = predict_tm * inv_sampling
        corrected_data = test_data[ts]
        ground_truth = corrected_data * sampling

        # Calculating the true value for the TM
        new_tm = pred_tm + ground_truth

        tm_labels[ts + Config.FWBW_CONV_LSTM_STEP] = new_tm
        _tm_labels[ts + Config.FWBW_CONV_LSTM_STEP] = new_tm
        labels[ts + Config.FWBW_CONV_LSTM_STEP] = sampling

    _err_1 = error_ratio(y_pred=tm_labels[Config.FWBW_CONV_LSTM_STEP:],
                         y_true=raw_data[Config.FWBW_CONV_LSTM_STEP:],
                         measured_matrix=labels[Config.FWBW_CONV_LSTM_STEP:])
    _err_2 = error_ratio(y_pred=_tm_labels[Config.FWBW_CONV_LSTM_STEP:],
                         y_true=raw_data[Config.FWBW_CONV_LSTM_STEP:],
                         measured_matrix=labels[Config.FWBW_CONV_LSTM_STEP:])

    print('Err_w: {} -- Err_wo: {}'.format(_err_1, _err_2))

    return tm_labels[Config.FWBW_CONV_LSTM_STEP:], labels[Config.FWBW_CONV_LSTM_STEP:], ims_tm, _tm_labels[
                                                                                                Config.FWBW_CONV_LSTM_STEP:]


def build_model(input_shape):
    print('|--- Build models.')
    alg_name = Config.ALG
    tag = Config.TAG
    data_name = Config.DATA_NAME

    fwbw_conv_lstm_net = FWBW_CONV_LSTM(input_shape=input_shape,
                                        cnn_layers=Config.FWBW_CONV_LSTM_LAYERS,
                                        a_filters=Config.FWBW_CONV_LSTM_FILTERS,
                                        a_strides=Config.FWBW_CONV_LSTM_STRIDES,
                                        dropouts=Config.FWBW_CONV_LSTM_DROPOUTS,
                                        kernel_sizes=Config.FWBW_CONV_LSTM_KERNEL_SIZE,
                                        rnn_dropouts=Config.FWBW_CONV_LSTM_RNN_DROPOUTS,
                                        alg_name=alg_name,
                                        tag=tag,
                                        check_point=True,
                                        saving_path=Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name, alg_name, tag,
                                                                                              Config.SCALER))

    print(fwbw_conv_lstm_net.model.summary())
    fwbw_conv_lstm_net.plot_models()
    return fwbw_conv_lstm_net


def load_trained_models(input_shape, best_ckp):
    fwbw_conv_lstm_net = build_model(input_shape)
    print('|--- Load trained model from: {}'.format(fwbw_conv_lstm_net.checkpoints_path))
    fwbw_conv_lstm_net.model.load_weights(fwbw_conv_lstm_net.checkpoints_path + "weights-{:02d}.hdf5".format(best_ckp))

    return fwbw_conv_lstm_net


def train_fwbw_conv_lstm(data):
    print('|-- Run model training.')

    gpu = Config.GPU

    data_name = Config.DATA_NAME
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
        assert Config.FWBW_CONV_LSTM_HIGH == 12
        assert Config.FWBW_CONV_LSTM_WIDE == 12
    else:
        day_size = Config.GEANT_DAY_SIZE
        assert Config.FWBW_CONV_LSTM_HIGH == 23
        assert Config.FWBW_CONV_LSTM_WIDE == 23

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')
    train_data_normalized2d, valid_data_normalized2d, _, scalers = data_scalling(train_data2d,
                                                                                 valid_data2d,
                                                                                 test_data2d)

    train_data_normalized = np.reshape(np.copy(train_data_normalized2d), newshape=(train_data_normalized2d.shape[0],
                                                                                   Config.FWBW_CONV_LSTM_WIDE,
                                                                                   Config.FWBW_CONV_LSTM_HIGH))
    valid_data_normalized = np.reshape(np.copy(valid_data_normalized2d), newshape=(valid_data_normalized2d.shape[0],
                                                                                   Config.FWBW_CONV_LSTM_WIDE,
                                                                                   Config.FWBW_CONV_LSTM_HIGH))

    input_shape = (Config.FWBW_CONV_LSTM_STEP,
                   Config.FWBW_CONV_LSTM_WIDE, Config.FWBW_CONV_LSTM_HIGH, Config.FWBW_CONV_LSTM_CHANNEL)

    with tf.device('/device:GPU:{}'.format(gpu)):
        fwbw_conv_lstm_net = build_model(input_shape)

    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------Training fw model-------------------------------------------------

    if not Config.FWBW_CONV_LSTM_VALID_TEST or \
            not os.path.isfile(
                fwbw_conv_lstm_net.checkpoints_path + 'weights-{:02d}.hdf5'.format(
                    Config.FWBW_CONV_LSTM_BEST_CHECKPOINT)):
        print('|--- Compile model. Saving path %s --- ' % fwbw_conv_lstm_net.saving_path)
        # -------------------------------- Create offline training and validating dataset ------------------------------

        print('|--- Create offline train set for forward net!')

        trainX, trainY_fw, trainY_bw = create_offline_fwbw_conv_lstm_data_fix_ratio(train_data_normalized,
                                                                                    input_shape,
                                                                                    Config.FWBW_CONV_LSTM_MON_RAIO,
                                                                                    train_data_normalized.std(), 3)
        print('|--- Create offline valid set for forward net!')

        validX, validY_fw, validY_bw = create_offline_fwbw_conv_lstm_data_fix_ratio(valid_data_normalized,
                                                                                    input_shape,
                                                                                    Config.FWBW_CONV_LSTM_MON_RAIO,
                                                                                    train_data_normalized.std(),
                                                                                    1)

        # Load model check point
        from_epoch = fwbw_conv_lstm_net.load_model_from_check_point()
        if from_epoch > 0:
            print('|--- Continue training forward model from epoch %i --- ' % from_epoch)
            training_history = fwbw_conv_lstm_net.model.fit(x=trainX,
                                                            y={'fw_outputs': trainY_fw, 'bw_outputs': trainY_bw},
                                                            batch_size=Config.FWBW_CONV_LSTM_BATCH_SIZE,
                                                            epochs=Config.FWBW_CONV_LSTM_N_EPOCH,
                                                            callbacks=fwbw_conv_lstm_net.callbacks_list,
                                                            validation_data=(validX, {'fw_outputs': validY_fw,
                                                                                      'bw_outputs': validY_bw}),
                                                            shuffle=True,
                                                            initial_epoch=from_epoch,
                                                            verbose=2)
        else:
            print('|--- Training new forward model.')

            training_history = fwbw_conv_lstm_net.model.fit(x=trainX,
                                                            y={'fw_outputs': trainY_fw, 'bw_outputs': trainY_bw},
                                                            batch_size=Config.FWBW_CONV_LSTM_BATCH_SIZE,
                                                            epochs=Config.FWBW_CONV_LSTM_N_EPOCH,
                                                            callbacks=fwbw_conv_lstm_net.callbacks_list,
                                                            validation_data=(validX, {'fw_outputs': validY_fw,
                                                                                      'bw_outputs': validY_bw}),
                                                            shuffle=True,
                                                            verbose=2)

        # Plot the training history
        if training_history is not None:
            fwbw_conv_lstm_net.plot_training_history(training_history)
    else:
        fwbw_conv_lstm_net.load_model_from_check_point(_from_epoch=Config.FWBW_CONV_LSTM_BEST_CHECKPOINT)

    # --------------------------------------------------------------------------------------------------------------
    run_test(valid_data2d, valid_data_normalized2d, train_data_normalized2d[-Config.FWBW_CONV_LSTM_STEP:],
             fwbw_conv_lstm_net, scalers)

    return


def ims_tm_test_data(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.LSTM_IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.LSTM_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.LSTM_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def test_fwbw_conv_lstm(data):
    print('|-- Run model testing.')
    gpu = Config.GPU

    data_name = Config.DATA_NAME
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
        assert Config.FWBW_CONV_LSTM_HIGH == 12
        assert Config.FWBW_CONV_LSTM_WIDE == 12

    else:
        day_size = Config.GEANT_DAY_SIZE
        assert Config.FWBW_CONV_LSTM_HIGH == 23
        assert Config.FWBW_CONV_LSTM_WIDE == 23

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')

    if 'Abilene' in data_name:
        print('|--- Remove last 3 days in test data.')
        test_data2d = test_data2d[0:-day_size * 3]

    _, valid_data_normalized2d, test_data_normalized2d, scalers = data_scalling(train_data2d,
                                                                                valid_data2d,
                                                                                test_data2d)
    input_shape = (Config.FWBW_CONV_LSTM_STEP,
                   Config.FWBW_CONV_LSTM_WIDE, Config.FWBW_CONV_LSTM_HIGH, Config.FWBW_CONV_LSTM_CHANNEL)

    with tf.device('/device:GPU:{}'.format(gpu)):
        fwbw_conv_lstm_net = load_trained_models(input_shape, Config.FWBW_CONV_LSTM_BEST_CHECKPOINT)

    run_test(test_data2d, test_data_normalized2d, valid_data_normalized2d[-Config.FWBW_CONV_LSTM_STEP:],
             fwbw_conv_lstm_net, scalers)

    return


def run_test(test_data2d, test_data_normalized2d, init_data2d, fwbw_conv_lstm_net, scalers):
    alg_name = Config.ALG
    tag = Config.TAG
    data_name = Config.DATA_NAME

    results_summary = pd.DataFrame(index=range(Config.FWBW_CONV_LSTM_TESTING_TIME),
                                   columns=['No.', 'err', 'r2', 'rmse', 'err_ims', 'r2_ims', 'rmse_ims'])

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    measured_matrix_ims2d = np.zeros((test_data2d.shape[0] - Config.FWBW_CONV_LSTM_IMS_STEP + 1,
                                      Config.FWBW_CONV_LSTM_WIDE * Config.FWBW_CONV_LSTM_HIGH))
    # if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name)):
    #     np.save(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name),
    #             test_data2d)
    #
    # if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER)):
    #     np.save(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER),
    #             test_data_normalized2d)

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name,
                                                                      alg_name, tag, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER))

    for i in range(Config.FWBW_CONV_LSTM_TESTING_TIME):
        print('|--- Run time {}'.format(i))

        init_data = np.reshape(init_data2d, newshape=(init_data2d.shape[0],
                                                      Config.FWBW_CONV_LSTM_WIDE,
                                                      Config.FWBW_CONV_LSTM_HIGH))
        test_data_normalized = np.reshape(test_data_normalized2d, newshape=(test_data_normalized2d.shape[0],
                                                                            Config.FWBW_CONV_LSTM_WIDE,
                                                                            Config.FWBW_CONV_LSTM_HIGH))

        pred_tm, measured_matrix, ims_tm, pred_tm_wo_corr = predict_fwbw_conv_lstm(initial_data=init_data,
                                                                                   test_data=test_data_normalized,
                                                                                   model=fwbw_conv_lstm_net.model)

        pred_tm2d = np.reshape(np.copy(pred_tm), newshape=(pred_tm.shape[0], pred_tm.shape[1] * pred_tm.shape[2]))
        pred_tm2d_wo = np.reshape(np.copy(pred_tm_wo_corr), newshape=(
        pred_tm_wo_corr.shape[0], pred_tm_wo_corr.shape[1] * pred_tm_wo_corr.shape[2]))
        measured_matrix2d = np.reshape(np.copy(measured_matrix),
                                       newshape=(measured_matrix.shape[0],
                                                 measured_matrix.shape[1] * measured_matrix.shape[2]))
        # np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred_scaled-{}.npy'.format(data_name, alg_name, tag,
        #                                                                       Config.SCALER, i),
        #         pred_tm2d)

        pred_tm_invert2d = scalers.inverse_transform(pred_tm2d)
        pred_tm_wo_invert2d = scalers.inverse_transform(pred_tm2d_wo)

        if np.any(np.isinf(pred_tm_invert2d)):
            raise ValueError('Value is infinity!')
        elif np.any(np.isnan(pred_tm_invert2d)):
            raise ValueError('Value is NaN!')

        if np.any(np.isinf(pred_tm_wo_invert2d)):
            raise ValueError('Value is infinity!')
        elif np.any(np.isnan(pred_tm_wo_invert2d)):
            raise ValueError('Value is NaN!')

        err.append(error_ratio(y_true=test_data2d, y_pred=pred_tm_invert2d, measured_matrix=measured_matrix2d))
        r2_score.append(calculate_r2_score(y_true=test_data2d, y_pred=pred_tm_invert2d))
        rmse.append(calculate_rmse(y_true=test_data2d / 1000000, y_pred=pred_tm_invert2d / 1000000))

        err_wo = error_ratio(y_true=test_data2d, y_pred=pred_tm_wo_invert2d, measured_matrix=measured_matrix2d)
        r2_score_wo = calculate_r2_score(y_true=test_data2d, y_pred=pred_tm_wo_invert2d)
        rmse_wo = calculate_rmse(y_true=test_data2d / 1000000, y_pred=pred_tm_wo_invert2d / 1000000)

        if Config.FWBW_CONV_LSTM_IMS:
            # Calculate error for multistep-ahead-prediction

            ims_tm2d = np.reshape(np.copy(ims_tm), newshape=(ims_tm.shape[0], ims_tm.shape[1] * ims_tm.shape[2]))

            ims_tm_invert2d = scalers.inverse_transform(ims_tm2d)

            ims_ytrue2d = ims_tm_test_data(test_data=test_data2d)

            err_ims.append(error_ratio(y_pred=ims_tm_invert2d,
                                       y_true=ims_ytrue2d,
                                       measured_matrix=measured_matrix_ims2d))

            r2_score_ims.append(calculate_r2_score(y_true=ims_ytrue2d, y_pred=ims_tm_invert2d))
            rmse_ims.append(calculate_rmse(y_true=ims_ytrue2d / 1000000, y_pred=ims_tm_invert2d / 1000000))
        else:
            err_ims.append(0)
            r2_score_ims.append(0)
            rmse_ims.append(0)

        print('Result: err\trmse\tr2 \t\t err_ims\trmse_ims\tr2_ims')
        print('        {}\t{}\t{} \t\t {}\t{}\t{}'.format(err[i], rmse[i], r2_score[i],
                                                          err_ims[i], rmse_ims[i],
                                                          r2_score_ims[i]))
        print('Result without data correction: err\trmse\tr2')
        print('        {}\t{}\t{}'.format(err_wo, rmse_wo, r2_score_wo))
        # np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred-{}.npy'.format(data_name, alg_name, tag,
        #                                                                Config.SCALER, i),
        #         pred_tm_invert2d)
        # np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/measure-{}.npy'.format(data_name, alg_name, tag,
        #                                                                   Config.SCALER, i),
        #         measured_matrix2d)

    results_summary['No.'] = range(Config.FWBW_CONV_LSTM_TESTING_TIME)
    results_summary['err'] = err
    results_summary['r2'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['err_ims'] = err_ims
    results_summary['r2_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    results_summary.to_csv(Config.RESULTS_PATH +
                           '{}-{}-{}-{}/results.csv'.format(data_name, alg_name, tag, Config.SCALER),
                           index=False)

    print('Test: {}-{}-{}-{}'.format(data_name, alg_name, tag, Config.SCALER))

    print('avg_err: {} - avg_rmse: {} - avg_r2: {}'.format(np.mean(np.array(err)),
                                                           np.mean(np.array(rmse)),
                                                           np.mean(np.array(r2_score))))

    return
