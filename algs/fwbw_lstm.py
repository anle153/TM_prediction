import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm

from Models.fwbw_LSTM import fwbw_lstm_model
from common import Config_fwbw_lstm as Config
from common.DataPreprocessing import prepare_train_valid_test_2d, create_offline_fwbw_lstm
from common.error_utils import error_ratio, calculate_r2_score, calculate_rmse, calculate_mape

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def prepare_input_online_prediction(data, labels):
    labels = labels.astype(int)
    data_x = np.zeros(shape=(data.shape[1], Config.FWBW_LSTM_STEP, Config.FWBW_LSTM_FEATURES))
    for flow_id in range(data.shape[1]):
        x = data[:, flow_id]
        label = labels[:, flow_id]

        data_x[flow_id, :, 0] = x
        data_x[flow_id, :, 1] = label

    return data_x


def data_correction_v3(rnn_input, pred_backward, labels):
    # Shape = (#n_flows, #time-steps)
    _rnn_input = np.copy(rnn_input.T)
    _labels = np.copy(labels.T)

    beta = np.zeros(_rnn_input.shape)
    for i in range(_rnn_input.shape[1] - Config.FWBW_LSTM_R):
        mu = np.sum(_labels[:, i + 1:i + Config.FWBW_LSTM_R + 1], axis=1) / Config.FWBW_LSTM_R

        h = np.arange(1, Config.FWBW_LSTM_R + 1)

        rho = (1 / (np.log(Config.FWBW_LSTM_R) + 1)) * np.sum(_labels[:, i + 1:i + Config.FWBW_LSTM_R + 1] / h, axis=1)

        beta[:, i] = mu * rho

    considered_backward = pred_backward[:, 1:]
    considered_rnn_input = _rnn_input[:, 0:-1]

    beta[beta > 0.5] = 0.5

    alpha = 1.0 - beta

    alpha = alpha[:, 0:-1]
    beta = beta[:, 0:-1]
    # gamma = gamma[:, 1:-1]

    # corrected_data = considered_rnn_input * alpha + considered_rnn_input * beta + considered_backward * gamma
    corrected_data = considered_rnn_input * alpha + considered_backward * beta

    return corrected_data.T


def data_correction_v4(rnn_input, pred_backward, labels, fw_mon_ratio):
    # Shape = (#n_flows, #time-steps)
    _rnn_input = np.copy(rnn_input.T)
    _labels = np.copy(labels.T)

    beta = np.zeros(_rnn_input.shape)
    for i in range(_rnn_input.shape[1] - Config.FWBW_LSTM_R):
        mu = np.sum(_labels[:, i + 1:i + Config.FWBW_LSTM_R + 1], axis=1) / Config.FWBW_LSTM_R

        h = np.arange(1, Config.FWBW_LSTM_R + 1)

        rho = (1 / (np.log(Config.FWBW_LSTM_R) + 1)) * np.sum(_labels[:, i + 1:i + Config.FWBW_LSTM_R + 1] / h, axis=1)

        beta[:, i] = mu * rho

        np.argwhere(fw_mon_ratio[:, i] - mu >= 0)

    considered_backward = pred_backward[:, 1:]
    considered_rnn_input = _rnn_input[:, 0:-1]

    beta[beta > 0.8] = 0.5

    alpha = 1.0 - beta

    alpha = alpha[:, 0:-1]
    beta = beta[:, 0:-1]
    # gamma = gamma[:, 1:-1]

    # corrected_data = considered_rnn_input * alpha + considered_rnn_input * beta + considered_backward * gamma
    corrected_data = considered_rnn_input * alpha + considered_backward * beta

    return corrected_data.T


def predict_fwbw_lstm_ims_v2(initial_data, initial_labels, model):
    ims_tm_pred = np.zeros(shape=(initial_data.shape[0] + Config.FWBW_LSTM_IMS_STEP, initial_data.shape[1]))
    ims_tm_pred[0:initial_data.shape[0]] = initial_data

    labels = np.zeros(shape=(initial_data.shape[0] + Config.FWBW_LSTM_IMS_STEP, initial_data.shape[1]))
    labels[0:initial_labels.shape[0]] = initial_labels

    for ts_ahead in range(Config.FWBW_LSTM_IMS_STEP):
        rnn_input = prepare_input_online_prediction(data=ims_tm_pred[ts_ahead:ts_ahead + Config.FWBW_LSTM_STEP],
                                                    labels=labels[ts_ahead:ts_ahead + Config.FWBW_LSTM_STEP])
        fw_outputs, _ = model.predict(rnn_input)

        fw_outputs = np.squeeze(fw_outputs, axis=2)  # Shape(#n_flows, #time-steps)

        pred_next_tm = np.copy(fw_outputs[:, -1])

        # corrected_data = data_correction_v3(rnn_input=np.copy(ims_tm_pred[ts_ahead: ts_ahead + Config.FWBW_LSTM_STEP]),
        #                                     pred_backward=bw_outputs,
        #                                     labels=labels[ts_ahead: ts_ahead + Config.FWBW_LSTM_STEP])
        #
        # measured_data = ims_tm_pred[ts_ahead:ts_ahead + Config.FWBW_LSTM_STEP - 1] * \
        #                 labels[ts_ahead:ts_ahead + Config.FWBW_LSTM_STEP - 1]
        # pred_data = corrected_data * (1.0 - labels[ts_ahead:ts_ahead + Config.FWBW_LSTM_STEP - 1])
        # ims_tm_pred[ts_ahead:ts_ahead + Config.FWBW_LSTM_STEP - 1] = measured_data + pred_data

        ims_tm_pred[ts_ahead + Config.FWBW_LSTM_STEP] = pred_next_tm

    return ims_tm_pred[-1, :]


def calculate_consecutive_loss(measured_matrix):
    """

    :param measured_matrix: shape(#n_flows, #time-steps)
    :return: consecutive_losses: shape(#n_flows)
    """

    consecutive_losses = []
    for flow_id in range(measured_matrix.shape[0]):
        flows_labels = measured_matrix[flow_id, :]
        if flows_labels[-1] == 1:
            consecutive_losses.append(1)
        else:
            measured_idx = np.argwhere(flows_labels == 1)
            if measured_idx.size == 0:
                consecutive_losses.append(measured_matrix.shape[1])
            else:
                consecutive_losses.append(measured_matrix.shape[1] - measured_idx[-1][0])

    consecutive_losses = np.asarray(consecutive_losses)
    return consecutive_losses


def set_measured_flow(rnn_input, pred_forward, labels, ):
    """

    :param rnn_input: shape(#n_flows, #time-steps)
    :param pred_forward: shape(#n_flows, #time-steps)
    :param labels: shape(n_flows, #time-steps)
    :return:
    """

    n_flows = rnn_input.shape[0]

    fw_losses = []
    for flow_id in range(rnn_input.shape[0]):
        idx_fw = labels[flow_id, 1:]

        fw_losses.append(error_ratio(y_true=rnn_input[flow_id, 1:][idx_fw == 1.0],
                                     y_pred=pred_forward[flow_id, :-1][idx_fw == 1.0],
                                     measured_matrix=np.zeros(idx_fw[idx_fw == 1.0].shape)))

    fw_losses = np.array(fw_losses)
    fw_losses[fw_losses == 0.] = np.max(fw_losses)

    w = calculate_flows_weights(fw_losses=fw_losses,
                                measured_matrix=labels)

    sampling = np.zeros(shape=n_flows)
    m = int(Config.FWBW_LSTM_MON_RATIO * n_flows)

    w = w.flatten()
    sorted_idx_w = np.argsort(w)
    sampling[sorted_idx_w[:m]] = 1

    return sampling


def set_measured_flow_fairness(rnn_input, labels):
    """

    :param rnn_input: shape(#n_flows, #time-steps)
    :param labels: shape(n_flows, #time-steps)
    :return:
    """

    n_flows = rnn_input.shape[0]

    cl = calculate_consecutive_loss(labels).astype(float)

    w = 1 / cl

    sampling = np.zeros(shape=n_flows)
    m = int(Config.FWBW_LSTM_MON_RATIO * n_flows)

    w = w.flatten()
    sorted_idx_w = np.argsort(w)
    sampling[sorted_idx_w[:m]] = 1

    return sampling


def calculate_flows_weights(fw_losses, measured_matrix):
    """

    :param fw_losses: shape(#n_flows)
    :param measured_matrix: shape(#n_flows, #time-steps)
    :return: w: flow weight shape(#n_flows)
    """

    cl = calculate_consecutive_loss(measured_matrix).astype(float)

    w = 1 / (fw_losses * Config.FWBW_LSTM_HYPERPARAMS[0] +
             cl * Config.FWBW_LSTM_HYPERPARAMS[1])

    return w


def predict_fwbw_lstm_v2(initial_data, test_data, model):
    tf_a = np.array([1.0, 0.0])

    # Initialize traffic matrix data
    tm_pred = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1]))
    tm_pred[0:initial_data.shape[0]] = initial_data

    # Initialize predicted_traffic matrice
    predicted_tm = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1]))
    predicted_tm[0:initial_data.shape[0]] = initial_data

    # Initialize measurement matrix
    labels = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1]))
    labels[0:initial_data.shape[0]] = np.ones(shape=initial_data.shape)

    # Forward losses
    ims_tm = np.zeros(shape=(test_data.shape[0] - Config.FWBW_LSTM_IMS_STEP + 1, test_data.shape[1]))

    raw_data = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1]))

    raw_data[0:initial_data.shape[0]] = initial_data
    raw_data[initial_data.shape[0]:] = test_data

    prediction_times = []
    import pandas as pd
    import time
    dump_prediction_time = pd.DataFrame(index=range(test_data.shape[0]), columns=['time_step', 'pred_time'])

    # Predict the TM from time slot look_back
    for ts in tqdm(range(test_data.shape[0])):

        _start = time.time()

        if Config.FWBW_LSTM_IMS and (ts < test_data.shape[0] - Config.FWBW_LSTM_IMS_STEP + 1):
            ims_tm[ts] = predict_fwbw_lstm_ims_v2(initial_data=np.copy(tm_pred[ts: ts + Config.FWBW_LSTM_STEP]),
                                                  initial_labels=np.copy(labels[ts: ts + Config.FWBW_LSTM_STEP]),
                                                  model=model)

        # Create 3D input for rnn
        # Shape(#n_flows, #time-steps, #features)
        rnn_input = prepare_input_online_prediction(data=tm_pred[ts: ts + Config.FWBW_LSTM_STEP],
                                                    labels=labels[ts: ts + Config.FWBW_LSTM_STEP])

        fw_outputs, bw_outputs = model.predict(rnn_input)  # Shape(#n_flows, #time-step, 1)

        fw_outputs = np.squeeze(fw_outputs, axis=2)  # Shape(#n_flows, #time-step)

        pred_next_tm = np.copy(fw_outputs[:, -1])  # Shape(#n_flows,)

        # Insert pred_next_tm to predicted traffic matrices
        predicted_tm[ts] = pred_next_tm

        # Data Correction: Shape(#time-steps, flows) for [ts+1 : ts + Config.FWBW_LSTM_STEP - 1]
        corrected_data = data_correction_v3(rnn_input=np.copy(tm_pred[ts: ts + Config.FWBW_LSTM_STEP]),
                                            pred_backward=bw_outputs,
                                            labels=labels[ts: ts + Config.FWBW_LSTM_STEP])
        # corrected_data = data_correction_v2(rnn_input=np.copy(tm_pred[ts: ts + Config.FWBW_LSTM_STEP]),
        #                                     pred_backward=bw_outputs,
        #                                     labels=labels[ts: ts + Config.FWBW_LSTM_STEP])

        measured_data = tm_pred[ts:ts + Config.FWBW_LSTM_STEP - 1] * labels[ts:ts + Config.FWBW_LSTM_STEP - 1]
        pred_data = corrected_data * (1.0 - labels[ts:ts + Config.FWBW_LSTM_STEP - 1])
        tm_pred[ts:ts + Config.FWBW_LSTM_STEP - 1] = measured_data + pred_data

        # Partial monitoring
        if Config.FWBW_LSTM_FLOW_SELECTION == Config.FLOW_SELECTIONS[0]:
            sampling = np.random.choice(tf_a, size=(test_data.shape[1]),
                                        p=[Config.FWBW_LSTM_MON_RATIO, 1 - Config.FWBW_LSTM_MON_RATIO])
        elif Config.FWBW_LSTM_FLOW_SELECTION == Config.FLOW_SELECTIONS[1]:
            sampling = set_measured_flow_fairness(rnn_input=np.copy(tm_pred[ts: ts + Config.FWBW_LSTM_STEP].T),
                                                  labels=labels[ts: ts + Config.FWBW_LSTM_STEP].T)
        else:
            sampling = set_measured_flow(rnn_input=np.copy(tm_pred[ts: ts + Config.FWBW_LSTM_STEP].T),
                                         pred_forward=fw_outputs,
                                         labels=labels[ts: ts + Config.FWBW_LSTM_STEP].T)

        new_input = pred_next_tm * (1.0 - sampling) + test_data[ts] * sampling

        tm_pred[ts + Config.FWBW_LSTM_STEP] = new_input
        labels[ts + Config.FWBW_LSTM_STEP] = sampling

        prediction_times.append(time.time() - _start)

    dump_prediction_time['time_step'] = range(test_data.shape[0])
    dump_prediction_time['pred_time'] = prediction_times
    dump_prediction_time.to_csv(Config.RESULTS_PATH + '{}-{}-{}-{}/Prediction_times.csv'.format(Config.DATA_NAME,
                                                                                                Config.ALG,
                                                                                                Config.TAG,
                                                                                                Config.SCALER),
                                index=False)

    return tm_pred[Config.FWBW_LSTM_STEP:], labels[Config.FWBW_LSTM_STEP:], ims_tm, predicted_tm[Config.FWBW_LSTM_STEP:]


def build_model(input_shape):
    print('|--- Build models fwbw-lstm.')

    # fwbw-lstm model
    fwbw_net = fwbw_lstm_model(input_shape=input_shape,
                               hidden=Config.FWBW_LSTM_HIDDEN_UNIT,
                               drop_out=Config.FWBW_LSTM_DROPOUT,
                               alg_name=Config.ALG, tag=Config.TAG, check_point=True,
                               saving_path=Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(Config.DATA_NAME, Config.ALG,
                                                                                     Config.TAG, Config.SCALER))
    fwbw_net.construct_fwbw_lstm()
    print(fwbw_net.model.summary())
    fwbw_net.plot_models()
    return fwbw_net


def load_trained_models(input_shape, ckp):
    fwbw_net = build_model(input_shape)
    print('|--- Load trained model from: {}'.format(fwbw_net.checkpoints_path))
    fwbw_net.model.load_weights(fwbw_net.checkpoints_path + "weights-{:02d}.hdf5".format(ckp))

    return fwbw_net


def train_fwbw_lstm(data):
    print('|-- Run model training fwbw_lstm.')

    if Config.DATA_NAME == Config.DATA_SETS[0]:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')

    scalers = PowerTransformer()
    scalers.fit(train_data2d)

    train_data_normalized2d = scalers.transform(train_data2d)
    valid_data_normalized2d = scalers.transform(valid_data2d)

    input_shape = (Config.FWBW_LSTM_STEP, Config.FWBW_LSTM_FEATURES)

    with tf.device('/device:GPU:{}'.format(Config.GPU)):
        fwbw_net = build_model(input_shape)

    # --------------------------------------------Training fw model-------------------------------------------------

    if not Config.FWBW_LSTM_VALID_TEST or \
            not os.path.isfile(
                fwbw_net.checkpoints_path + 'weights-{:02d}.hdf5'.format(Config.FWBW_LSTM_BEST_CHECKPOINT)):
        print('|--- Compile model. Saving path %s --- ' % fwbw_net.saving_path)
        # -------------------------------- Create offline training and validating dataset --------------------------

        print('|--- Create offline train set for forward net!')

        train_x, train_y_1, train_y_2 = create_offline_fwbw_lstm(train_data_normalized2d,
                                                                 input_shape, Config.FWBW_LSTM_MON_RATIO,
                                                                 train_data_normalized2d.std())
        print('|--- Create offline valid set for forward net!')

        valid_x, valid_y_1, valid_y_2 = create_offline_fwbw_lstm(valid_data_normalized2d,
                                                                 input_shape, Config.FWBW_LSTM_MON_RATIO,
                                                                 train_data_normalized2d.std())

        # Load model check point
        from_epoch = fwbw_net.load_model_from_check_point()
        if from_epoch > 0:
            print('|--- Continue training forward model from epoch %i --- ' % from_epoch)
            training_fw_history = fwbw_net.model.fit(x=train_x,
                                                     y=[train_y_1, train_y_2],
                                                     batch_size=Config.FWBW_LSTM_BATCH_SIZE,
                                                     epochs=Config.FWBW_LSTM_N_EPOCH,
                                                     callbacks=fwbw_net.callbacks_list,
                                                     validation_data=(valid_x, [valid_y_1, valid_y_2]),
                                                     shuffle=True,
                                                     initial_epoch=from_epoch,
                                                     verbose=2)
        else:
            print('|--- Training new forward model.')

            training_fw_history = fwbw_net.model.fit(x=train_x,
                                                     y=[train_y_1, train_y_2],
                                                     batch_size=Config.FWBW_LSTM_BATCH_SIZE,
                                                     epochs=Config.FWBW_LSTM_N_EPOCH,
                                                     callbacks=fwbw_net.callbacks_list,
                                                     validation_data=(valid_x, [valid_y_1, valid_y_2]),
                                                     shuffle=True,
                                                     verbose=2)

        # Plot the training history
        if training_fw_history is not None:
            fwbw_net.plot_training_history(training_fw_history)
            fwbw_net.save_model_history(training_fw_history)

    else:
        fwbw_net.load_model_from_check_point(_from_epoch=Config.FWBW_LSTM_BEST_CHECKPOINT)
    # --------------------------------------------------------------------------------------------------------------

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                      Config.ALG, Config.TAG, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                Config.ALG, Config.TAG, Config.SCALER))
    results_summary = pd.DataFrame(index=range(Config.FWBW_LSTM_TESTING_TIME),
                                   columns=['No.', 'err', 'r2', 'rmse', 'err_ims', 'r2_ims',
                                            'rmse_ims'])

    results_summary = run_test(valid_data2d, valid_data_normalized2d, fwbw_net, scalers, results_summary)

    result_file_name = 'Valid_results_{}.csv'.format(Config.FWBW_LSTM_FLOW_SELECTION)

    results_summary.to_csv(Config.RESULTS_PATH +
                           '{}-{}-{}-{}/{}'.format(Config.DATA_NAME, Config.ALG, Config.TAG,
                                                   Config.SCALER, result_file_name),
                           index=False)

    return


def ims_tm_test_data(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.FWBW_LSTM_IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.FWBW_LSTM_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.FWBW_LSTM_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def test_fwbw_lstm(data):
    print('|-- Run model testing.')

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

    if Config.DATA_NAME == Config.DATA_SETS[0]:
        print('|--- Remove last 3 days in test data.')
        test_data2d = test_data2d[0:-day_size * 3]

    scalers = PowerTransformer()
    scalers.fit(train_data2d)

    test_data_normalized2d = scalers.transform(test_data2d)

    input_shape = (Config.FWBW_LSTM_STEP, Config.FWBW_LSTM_FEATURES)

    with tf.device('/device:GPU:{}'.format(Config.GPU)):
        fwbw_net = load_trained_models(input_shape, Config.FWBW_LSTM_BEST_CHECKPOINT)

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                      Config.ALG, Config.TAG, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                Config.ALG, Config.TAG, Config.SCALER))
    results_summary = pd.DataFrame(index=range(Config.FWBW_LSTM_TESTING_TIME),
                                   columns=['No.', 'err', 'r2', 'rmse', 'err_ims', 'r2_ims',
                                            'rmse_ims'])

    results_summary = run_test(test_data2d, test_data_normalized2d, fwbw_net, scalers, results_summary)

    if Config.FWBW_LSTM_IMS:
        result_file_name = 'Test_results_ims_{}_{}.csv'.format(Config.FWBW_LSTM_IMS_STEP,
                                                               Config.FWBW_LSTM_FLOW_SELECTION)
    else:
        result_file_name = 'Test_results_{}.csv'.format(Config.FWBW_LSTM_FLOW_SELECTION)

    results_summary.to_csv(Config.RESULTS_PATH +
                           '{}-{}-{}-{}/{}'.format(Config.DATA_NAME, Config.ALG, Config.TAG,
                                                   Config.SCALER, result_file_name),
                           index=False)

    return


def prepare_test_set(test_data2d, test_data_normalized2d):
    if Config.DATA_NAME == Config.DATA_SETS[0]:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    idx = np.random.random_integers(Config.FWBW_LSTM_STEP,
                                    test_data2d.shape[0] - day_size * Config.FWBW_LSTM_TEST_DAYS - 10)

    test_data_normalize = np.copy(test_data_normalized2d[idx:idx + day_size * Config.FWBW_LSTM_TEST_DAYS])
    init_data_normalize = np.copy(test_data_normalized2d[idx - Config.FWBW_LSTM_STEP: idx])
    test_data = test_data2d[idx:idx + day_size * Config.FWBW_LSTM_TEST_DAYS]

    return test_data_normalize, init_data_normalize, test_data


def prepare_test_set_last_5days(test_data2d, test_data_normalized2d):
    if Config.DATA_NAME == Config.DATA_SETS[0]:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    idx = test_data2d.shape[0] - day_size * 5 - 10

    test_data_normalize = np.copy(test_data_normalized2d[idx:idx + day_size * 5])
    init_data_normalize = np.copy(test_data_normalized2d[idx - Config.FWBW_LSTM_STEP: idx])
    test_data = test_data2d[idx:idx + day_size * 5]

    return test_data_normalize, init_data_normalize, test_data


def prepare_test_set_last_day(test_data2d, test_data_normalized2d):
    if Config.DATA_NAME == Config.DATA_SETS[0]:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    idx = test_data2d.shape[0] - day_size * 1 - 10

    test_data_normalize = np.copy(test_data_normalized2d[idx:idx + day_size * 1])
    init_data_normalize = np.copy(test_data_normalized2d[idx - Config.FWBW_LSTM_STEP: idx])
    test_data = test_data2d[idx:idx + day_size * 1]

    return test_data_normalize, init_data_normalize, test_data


def run_test(test_data2d, test_data_normalized2d, fwbw_net, scalers, results_summary):
    mape, r2_score, rmse = [], [], []
    mape_ims, r2_score_ims, rmse_ims = [], [], []

    # per_gain = []

    for i in range(Config.FWBW_LSTM_TESTING_TIME):
        print('|--- Run time {}'.format(i))
        # test_data_normalize, init_data_normalize, test_data = prepare_test_set(test_data2d, test_data_normalized2d)
        test_data_normalize, init_data_normalize, test_data = prepare_test_set_last_5days(test_data2d,
                                                                                          test_data_normalized2d)
        ims_test_data = ims_tm_test_data(test_data=test_data)
        measured_matrix_ims = np.zeros(shape=ims_test_data.shape)

        pred_tm2d, measured_matrix2d, ims_tm2d, predicted_tm2d = predict_fwbw_lstm_v2(initial_data=init_data_normalize,
                                                                                      test_data=test_data_normalize,
                                                                                      model=fwbw_net.model)

        pred_tm_invert2d = scalers.inverse_transform(pred_tm2d)
        predicted_tm_invert2d = scalers.inverse_transform(predicted_tm2d)

        # pred_tm_wo_invert2d = scalers.inverse_transform(pred_tm2d_wo)
        if np.any(np.isinf(pred_tm_invert2d)):
            raise ValueError('Value is infinity!')
        elif np.any(np.isnan(pred_tm_invert2d)):
            raise ValueError('Value is NaN!')

        if np.any(np.isinf(predicted_tm_invert2d)):
            raise ValueError('Value is infinity!')
        elif np.any(np.isnan(predicted_tm_invert2d)):
            raise ValueError('Value is NaN!')

        r2_score.append(calculate_r2_score(y_true=test_data, y_pred=predicted_tm_invert2d))
        rmse.append(calculate_rmse(y_true=test_data, y_pred=predicted_tm_invert2d))
        mape.append(calculate_mape(y_true=test_data, y_pred=predicted_tm_invert2d))

        if Config.FWBW_LSTM_IMS:
            # Calculate error for multistep-ahead-prediction
            ims_tm_invert2d = scalers.inverse_transform(ims_tm2d)

            # err_ims.append(error_ratio(y_pred=ims_tm_invert2d,
            #                            y_true=ims_test_data,
            #                            measured_matrix=measured_matrix_ims))

            mape_ims.append(calculate_mape(y_true=ims_test_data, y_pred=ims_tm_invert2d))
            r2_score_ims.append(calculate_r2_score(y_true=ims_test_data, y_pred=ims_tm_invert2d))
            rmse_ims.append(calculate_rmse(y_true=ims_test_data, y_pred=ims_tm_invert2d))
        else:
            mape_ims.append(0)
            r2_score_ims.append(0)
            rmse_ims.append(0)

        print('Result: mape\trmse\tr2 \t\t err_ims\trmse_ims\tr2_ims')
        print('        {}\t{}\t{} \t\t {}\t{}\t{}'.format(mape[i], rmse[i], r2_score[i],
                                                          mape_ims[i], rmse_ims[i], r2_score_ims[i]))

    results_summary['No.'] = range(Config.FWBW_LSTM_TESTING_TIME)
    results_summary['mape'] = mape
    results_summary['r2'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['mape_ims'] = mape_ims
    results_summary['r2_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    print('Test: {}-{}-{}-{}-{}'.format(Config.DATA_NAME, Config.ALG, Config.TAG, Config.SCALER,
                                        Config.FWBW_LSTM_FLOW_SELECTION))

    print('avg_err: {} - avg_rmse: {} - avg_r2: {}'.format(np.mean(np.array(mape)),
                                                           np.mean(np.array(rmse)),
                                                           np.mean(np.array(r2_score))))

    return results_summary
