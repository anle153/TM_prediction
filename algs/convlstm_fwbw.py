import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from Models.ConvLSTM_model_fwbw import ConvLSTM_FWBW
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_2d, create_offline_fwbw_convlstm_data, data_scalling
from common.error_utils import calculate_consecutive_loss_3d, recovery_loss_3d, error_ratio, calculate_r2_score, \
    calculate_rmse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def plot_test_data(prefix, raw_data, pred_fw, pred_bw, current_data):
    saving_path = Config.RESULTS_PATH + 'plot_fwbw_convlstm/'

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

    w = 1 / (rl_forward_scaled * Config.FWBW_CONVLSTM_HYPERPARAMS[0] +
             rl_backward_scaled * Config.FWBW_CONVLSTM_HYPERPARAMS[1] +
             cl_scaled * Config.FWBW_CONVLSTM_HYPERPARAMS[2] +
             flows_stds_scaled * Config.FWBW_CONVLSTM_HYPERPARAMS[3])

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
    m = int(Config.FWBW_CONVLSTM_MON_RAIO * rnn_input.shape[1] * rnn_input.shape[2])

    w = w.flatten()
    sorted_idx_w = np.argpartition(w, m)
    sampling[sorted_idx_w[:m]] = 1

    sampling = np.expand_dims(sampling, axis=0)

    sampling = np.reshape(sampling, newshape=(rnn_input.shape[1], rnn_input.shape[2]))

    return sampling


def calculate_forward_backward_loss_3d(measured_block, pred_forward, pred_backward, rnn_input):
    eps = 10e-8

    rl_forward = recovery_loss_3d(rnn_input=rnn_input[1:], rnn_updated=pred_forward[:-1],
                                  measured_matrix=measured_block[1:])
    rl_forward[rl_forward == 0] = eps

    rl_backward = recovery_loss_3d(rnn_input=rnn_input[:-1], rnn_updated=pred_backward[1:],
                                   measured_matrix=measured_block[:-1])
    rl_backward[rl_backward == 0] = eps

    return rl_forward, rl_backward


# def ims_tm_prediction(init_data_labels, forward_model, backward_model):
#     multi_steps_tm = np.zeros(shape=(init_data_labels.shape[0] + Config.FWBW_CONVLSTM_IMS_STEP,
#                                      init_data_labels.shape[1], init_data_labels.shape[2], init_data_labels.shape[3]))
#
#     multi_steps_tm[0:init_data_labels.shape[0], :, :, :] = init_data_labels
#
#     for ts_ahead in range(Config.FWBW_CONVLSTM_IMS_STEP):
#         rnn_input = multi_steps_tm[-Config.FWBW_CONVLSTM_STEP:, :, :, :]  # shape(timesteps, od, od , 2)
#
#         rnn_input_forward = np.expand_dims(rnn_input, axis=0)  # shape(1, timesteps, od, od , 2)
#
#         rnn_input_backward = np.flip(rnn_input, axis=0)
#         rnn_input_backward = np.expand_dims(rnn_input_backward, axis=0)  # shape(1, timesteps, od, od , 2)
#
#         # Prediction results from forward network
#         predictX = forward_model.predict(rnn_input_forward)  # shape(1, timesteps, od, od , 1)
#
#         predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, od, od , 1)
#         predictX = np.squeeze(predictX, axis=3)  # shape(timesteps, od, od)
#
#         # Prediction results from backward network
#         predictX_backward = backward_model.predict(rnn_input_backward)  # shape(1, timesteps, od, od , 1)
#
#         predictX_backward = np.squeeze(predictX_backward, axis=0)  # shape(timesteps, od, od , 1)
#         predictX_backward = np.squeeze(predictX_backward, axis=3)  # shape(timesteps, od, od)
#
#         # Flipping the backward prediction
#         predictX_backward = np.flip(predictX_backward, axis=0)
#
#         # Correcting the imprecise input data
#         updating_historical_data_3d(tm_labels=multi_steps_tm, pred_forward=predictX, pred_backward=predictX_backward,
#                                     rnn_input_labels=rnn_input, ts=ts_ahead)
#
#         predict_tm = predictX[-1, :, :]
#
#         sampling = np.zeros(shape=(Config.FWBW_CONVLSTM_WIDE, Config.FWBW_CONVLSTM_HIGH, 1))
#
#         # Calculating the true value for the TM
#         new_input = predict_tm
#
#         # Concaternating the new tm to the final results
#         # Shape = (12, 12, 2)
#         new_input = np.concatenate([np.expand_dims(new_input, axis=2), sampling], axis=2)
#         multi_steps_tm[ts_ahead + Config.FWBW_CONVLSTM_STEP] = new_input  # Shape = (timestep, 12, 12, 2)
#
#     return multi_steps_tm[-1, :, :, 0]


def predict_fwbw_convlstm(initial_data, test_data, model):
    tf_a = np.array([1.0, 0.0])

    tm_labels = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    tm_labels[0:initial_data.shape[0], :, :] = initial_data

    _tm_labels = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    _tm_labels[0:initial_data.shape[0], :, :] = initial_data

    labels = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    labels[0:initial_data.shape[0], :, :] = np.ones(shape=initial_data.shape)

    ims_tm = np.zeros(
        shape=(test_data.shape[0] - Config.FWBW_CONVLSTM_IMS_STEP + 1, test_data.shape[1], test_data.shape[2]))

    raw_data = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1], test_data.shape[2]))

    raw_data[0:initial_data.shape[0]] = initial_data
    raw_data[initial_data.shape[0]:] = test_data

    for ts in range(test_data.shape[0]):

        # if Config.FWBW_IMS and (ts <= test_data.shape[0] - Config.FWBW_CONVLSTM_IMS_STEP):
        #     ims_tm[ts] = ims_tm_prediction(init_data_labels=tm_labels[ts:ts + Config.FWBW_CONVLSTM_STEP, :, :, :],
        #                                    forward_model=forward_model,
        #                                    backward_model=backward_model)
        rnn_input = np.zeros(
            shape=(Config.FWBW_CONVLSTM_STEP,
                   Config.FWBW_CONVLSTM_WIDE, Config.FWBW_CONVLSTM_HIGH, Config.FWBW_CONVLSTM_CHANNEL))

        rnn_input[:, :, :, 0] = tm_labels[ts:(ts + Config.FWBW_CONVLSTM_STEP)]
        rnn_input[:, :, :, 1] = labels[ts:(ts + Config.FWBW_CONVLSTM_STEP)]

        rnn_input_backward = np.flip(np.copy(rnn_input), axis=0)

        rnn_input_forward = np.expand_dims(rnn_input, axis=0)

        # Prediction results from forward network
        predict_tm, corr_data = model.predict(rnn_input_forward)  # shape(1, timesteps, od, od , 1)
        predict_tm = np.reshape(predict_tm, newshape=(predict_tm.shape[0], test_data.shape[1], test_data.shape[2]))

        # if ts == 20:
        #     plot_test_data('Before_update', raw_data[ts + 1:ts + Config.FWBW_CONVLSTM_STEP - 1],
        #                    predictX[:-2],
        #                    predictX_backward[2:],
        #                    tm_labels[ts + 1:ts + Config.FWBW_CONVLSTM_STEP - 1])

        # before_ = np.copy(tm_labels[ts + 1:ts + Config.FWBW_CONVLSTM_STEP - 1])

        # _err_1 = error_ratio(y_pred=tm_labels[ts:ts + Config.FWBW_CONVLSTM_STEP],
        #                      y_true=raw_data[ts:ts + Config.FWBW_CONVLSTM_STEP],
        #                      measured_matrix=labels[ts: ts + Config.FWBW_CONVLSTM_STEP])

        # Correcting the imprecise input data

        corr_data = np.reshape(corr_data, (Config.FWBW_CONVLSTM_STEP - 2,
                                           Config.FWBW_CONVLSTM_WIDE, Config.FWBW_CONVLSTM_HIGH))
        _labels = 1 - labels[(ts + 1):(ts + Config.FWBW_LSTM_STEP - 1)]

        corr_data = corr_data * _labels
        tm_labels[(ts + 1):(ts + Config.FWBW_CONVLSTM_STEP - 1)] = \
            tm_labels[(ts + 1):(ts + Config.FWBW_CONVLSTM_STEP - 1)] * \
            labels[(ts + 1):(ts + Config.FWBW_CONVLSTM_STEP - 1)] + \
            corr_data

        if Config.FWBW_CONVLSTM_RANDOM_ACTION:
            sampling = np.random.choice(tf_a, size=(test_data.shape[1], test_data.shape[2]),
                                        p=(Config.FWBW_CONVLSTM_MON_RAIO, 1 - Config.FWBW_CONVLSTM_MON_RAIO))
        # else:
        #     sampling = set_measured_flow_3d(rnn_input=tm_labels[ts:ts + Config.FWBW_CONVLSTM_STEP],
        #                                     labels=labels[ts:ts + Config.FWBW_CONVLSTM_STEP],
        #                                     forward_pred=predictX,
        #                                     backward_pred=predictX_backward)

        # Selecting next monitored flows randomly
        inv_sampling = 1 - sampling

        pred_tm = predict_tm * inv_sampling
        corrected_data = test_data[ts]
        ground_truth = corrected_data * sampling

        # Calculating the true value for the TM
        new_tm = pred_tm + ground_truth

        tm_labels[ts + Config.FWBW_CONVLSTM_STEP] = new_tm
        _tm_labels[ts + Config.FWBW_CONVLSTM_STEP] = new_tm
        labels[ts + Config.FWBW_CONVLSTM_STEP] = sampling

    _err_1 = error_ratio(y_pred=tm_labels[Config.FWBW_CONVLSTM_STEP:],
                         y_true=raw_data[Config.FWBW_CONVLSTM_STEP:],
                         measured_matrix=labels[Config.FWBW_CONVLSTM_STEP:])
    _err_2 = error_ratio(y_pred=_tm_labels[Config.FWBW_CONVLSTM_STEP:],
                         y_true=raw_data[Config.FWBW_CONVLSTM_STEP:],
                         measured_matrix=labels[Config.FWBW_CONVLSTM_STEP:])

    print('Err_w: {} -- Err_wo: {}'.format(_err_1, _err_2))

    return tm_labels[Config.FWBW_CONVLSTM_STEP:], labels[Config.FWBW_CONVLSTM_STEP:], ims_tm


def build_model(input_shape):
    print('|--- Build models.')
    alg_name = Config.ALG
    tag = Config.TAG
    data_name = Config.DATA_NAME

    net = ConvLSTM_FWBW(input_shape=input_shape,
                        cnn_layers=Config.FWBW_CONVLSTM_LAYERS,
                        a_filters=Config.FWBW_CONVLSTM_FILTERS,
                        a_strides=Config.FWBW_CONVLSTM_STRIDES,
                        dropouts=Config.FWBW_CONVLSTM_DROPOUTS,
                        kernel_sizes=Config.FWBW_CONVLSTM_KERNEL_SIZE,
                        rnn_dropouts=Config.FWBW_CONVLSTM_RNN_DROPOUTS,
                        alg_name=alg_name,
                        tag=tag,
                        check_point=True,
                        saving_path=Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER))

    print(net.model.summary())
    # net.plot_models()

    return net


def load_trained_models(input_shape, best_ckp):
    net = build_model(input_shape)
    print('|--- Load trained model from: {}'.format(net.checkpoints_path))
    net.model.load_weights(net.checkpoints_path + "weights-{:02d}.hdf5".format(best_ckp))

    return net


def train_fwbw_convlstm(data, experiment):
    print('|-- Run model training.')

    params = Config.set_comet_params_fwbw_convlstm()

    gpu = Config.GPU

    data_name = Config.DATA_NAME
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
        assert Config.FWBW_CONVLSTM_HIGH == 12
        assert Config.FWBW_CONVLSTM_WIDE == 12
    else:
        day_size = Config.GEANT_DAY_SIZE
        assert Config.FWBW_CONVLSTM_HIGH == 23
        assert Config.FWBW_CONVLSTM_WIDE == 23

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')
    train_data_normalized2d, valid_data_normalized2d, _, scalers = data_scalling(np.copy(train_data2d),
                                                                                 np.copy(valid_data2d),
                                                                                 np.copy(test_data2d))

    train_data_normalized = np.reshape(np.copy(train_data_normalized2d), newshape=(train_data_normalized2d.shape[0],
                                                                                   Config.FWBW_CONVLSTM_WIDE,
                                                                                   Config.FWBW_CONVLSTM_HIGH))
    valid_data_normalized = np.reshape(np.copy(valid_data_normalized2d), newshape=(valid_data_normalized2d.shape[0],
                                                                                   Config.FWBW_CONVLSTM_WIDE,
                                                                                   Config.FWBW_CONVLSTM_HIGH))

    input_shape = (Config.FWBW_CONVLSTM_STEP,
                   Config.FWBW_CONVLSTM_WIDE, Config.FWBW_CONVLSTM_HIGH, Config.FWBW_CONVLSTM_CHANNEL)

    with tf.device('/device:GPU:{}'.format(gpu)):
        net = build_model(input_shape)

    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------Training fw model-------------------------------------------------

    if os.path.isfile(path=net.checkpoints_path + 'weights-{:02d}.hdf5'.format(Config.FWBW_CONVLSTM_N_EPOCH)):
        print('|--- Forward model exist! Load model from epoch: {}'.format(Config.FW_BEST_CHECKPOINT))
        net.load_model_from_check_point(_from_epoch=Config.FW_BEST_CHECKPOINT)
    else:
        print('|--- Compile model. Saving path %s --- ' % net.saving_path)
        # -------------------------------- Create offline training and validating dataset ------------------------------

        print('|--- Create offline train set!')

        trainX, trainY_1, trainY_2 = create_offline_fwbw_convlstm_data(train_data_normalized,
                                                                       input_shape, Config.FWBW_CONVLSTM_MON_RAIO,
                                                                       train_data_normalized.mean(),
                                                                       4)
        print('|--- Create offline valid set!')

        validX, validY_1, validY_2 = create_offline_fwbw_convlstm_data(valid_data_normalized,
                                                                       input_shape, Config.FWBW_CONVLSTM_MON_RAIO,
                                                                       train_data_normalized.mean(),
                                                                       1)

        # Load model check point
        from_epoch = net.load_model_from_check_point()
        if from_epoch > 0:
            print('|--- Continue training forward model from epoch %i --- ' % from_epoch)
            training_fw_history = net.model.fit(x=trainX,
                                                y={'pred_data': trainY_1, 'corr_data': trainY_2},
                                                batch_size=Config.FWBW_CONVLSTM_BATCH_SIZE,
                                                epochs=Config.FWBW_CONVLSTM_N_EPOCH,
                                                callbacks=net.callbacks_list,
                                                validation_data=(
                                                validX, {'pred_data': validY_1, 'corr_data': validY_2}),
                                                shuffle=True,
                                                initial_epoch=from_epoch,
                                                verbose=2)
        else:
            print('|--- Training new forward model.')

            training_fw_history = net.model.fit(x=trainX,
                                                y={'pred_data': trainY_1, 'corr_data': trainY_2},
                                                batch_size=Config.FWBW_CONVLSTM_BATCH_SIZE,
                                                epochs=Config.FWBW_CONVLSTM_N_EPOCH,
                                                callbacks=net.callbacks_list,
                                                validation_data=(
                                                validX, {'pred_data': validY_1, 'corr_data': validY_2}),
                                                shuffle=True,
                                                verbose=2)

        # Plot the training history
        if training_fw_history is not None:
            net.plot_training_history(training_fw_history)

    # --------------------------------------------------------------------------------------------------------------
    run_test(experiment, valid_data2d, valid_data_normalized2d, train_data_normalized2d[-Config.FWBW_CONVLSTM_STEP:],
             net, params, scalers)

    return


def ims_tm_test_data(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.LSTM_IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.LSTM_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.LSTM_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def test_fwbw_convlstm(data, experiment):
    print('|-- Run model testing.')
    gpu = Config.GPU

    params = Config.set_comet_params_fwbw_convlstm()

    data_name = Config.DATA_NAME
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    if not Config.ALL_DATA:
        data = data[0:Config.NUM_DAYS * day_size]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')

    if 'Abilene' in data_name:
        print('|--- Remove last 3 days in test data.')
        test_data2d = test_data2d[0:-day_size * 3]

    _, valid_data_normalized2d, test_data_normalized2d, scalers = data_scalling(train_data2d,
                                                                                valid_data2d,
                                                                                test_data2d)
    input_shape = (Config.FWBW_CONVLSTM_STEP,
                   Config.FWBW_CONVLSTM_WIDE, Config.FWBW_CONVLSTM_HIGH, Config.FWBW_CONVLSTM_CHANNEL)

    with tf.device('/device:GPU:{}'.format(gpu)):
        net = load_trained_models(input_shape, Config.FW_BEST_CHECKPOINT)

    run_test(experiment, test_data2d, test_data_normalized2d, valid_data_normalized2d[-Config.FWBW_CONVLSTM_STEP:],
             net, params, scalers)

    return


def run_test(experiment, test_data2d, test_data_normalized2d, init_data2d, net, params, scalers):
    alg_name = Config.ALG
    tag = Config.TAG
    data_name = Config.DATA_NAME

    results_summary = pd.DataFrame(index=range(Config.FWBW_CONVLSTM_TESTING_TIME),
                                   columns=['No.', 'err', 'r2', 'rmse', 'err_ims', 'r2_ims', 'rmse_ims'])

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    measured_matrix_ims2d = np.zeros((test_data2d.shape[0] - Config.FWBW_CONVLSTM_IMS_STEP + 1,
                                      Config.FWBW_CONVLSTM_WIDE * Config.FWBW_CONVLSTM_HIGH))
    if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name)):
        np.save(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name),
                test_data2d)

    if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER)):
        np.save(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER),
                test_data_normalized2d)

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name,
                                                                      alg_name, tag, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER))

    with experiment.test():
        for i in range(Config.FWBW_CONVLSTM_TESTING_TIME):
            print('|--- Run time {}'.format(i))

            init_data = np.reshape(init_data2d, newshape=(init_data2d.shape[0],
                                                          Config.FWBW_CONVLSTM_WIDE,
                                                          Config.FWBW_CONVLSTM_HIGH))
            test_data_normalized = np.reshape(test_data_normalized2d, newshape=(test_data_normalized2d.shape[0],
                                                                                Config.FWBW_CONVLSTM_WIDE,
                                                                                Config.FWBW_CONVLSTM_HIGH))

            pred_tm, measured_matrix, ims_tm = predict_fwbw_convlstm(initial_data=init_data,
                                                                     test_data=test_data_normalized,
                                                                     model=net.model)

            pred_tm2d = np.reshape(np.copy(pred_tm), newshape=(pred_tm.shape[0], pred_tm.shape[1] * pred_tm.shape[2]))
            measured_matrix2d = np.reshape(np.copy(measured_matrix),
                                           newshape=(measured_matrix.shape[0],
                                                     measured_matrix.shape[1] * measured_matrix.shape[2]))
            np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred_scaled-{}.npy'.format(data_name, alg_name, tag,
                                                                                  Config.SCALER, i),
                    pred_tm2d)

            pred_tm_invert2d = scalers.inverse_transform(pred_tm2d)

            if np.any(np.isinf(pred_tm_invert2d)):
                raise ValueError('Value is infinity!')
            elif np.any(np.isnan(pred_tm_invert2d)):
                raise ValueError('Value is NaN!')

            err.append(error_ratio(y_true=test_data2d, y_pred=pred_tm_invert2d, measured_matrix=measured_matrix2d))
            r2_score.append(calculate_r2_score(y_true=test_data2d, y_pred=pred_tm_invert2d))
            rmse.append(calculate_rmse(y_true=test_data2d / 1000000, y_pred=pred_tm_invert2d / 1000000))

            if Config.FWBW_IMS:
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
            np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred-{}.npy'.format(data_name, alg_name, tag,
                                                                           Config.SCALER, i),
                    pred_tm_invert2d)
            np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/measure-{}.npy'.format(data_name, alg_name, tag,
                                                                              Config.SCALER, i),
                    measured_matrix2d)

        results_summary['No.'] = range(Config.FWBW_CONVLSTM_TESTING_TIME)
        results_summary['err'] = err
        results_summary['r2'] = r2_score
        results_summary['rmse'] = rmse
        results_summary['err_ims'] = err_ims
        results_summary['r2_ims'] = r2_score_ims
        results_summary['rmse_ims'] = rmse_ims

        results_summary.to_csv(Config.RESULTS_PATH +
                               '{}-{}-{}-{}/results.csv'.format(data_name, alg_name, tag, Config.SCALER),
                               index=False)

        metrics = {
            'err': results_summary['err'],
            'rmse': results_summary['rmse'],
            'r2': results_summary['r2'],
            'err_ims': results_summary['err_ims'],
            'rmse_ims': results_summary['rmse_ims'],
            'r2_ims': results_summary['rmse_ims'],
        }

        # experiment.log_metrics(metrics)
        # experiment.log_parameters(params)
        print('avg_err: {} - avg_rmse: {} - avg_r2: {}'.format(np.mean(np.array(err)),
                                                               np.mean(np.array(rmse)),
                                                               np.mean(np.array(r2_score))))

    return
