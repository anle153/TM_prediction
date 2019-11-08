import numpy as np
import tensorflow as tf
from tqdm import tqdm

from Models.lstm.fwbw_lstm_supervisor import FwbwLstmRegression
from common.error_utils import error_ratio, calculate_r2_score, calculate_rmse, calculate_mape

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
session = tf.Session(config=config_gpu)


def prepare_input_online_prediction(data, labels):
    labels = labels.astype(int)
    data_x = np.zeros(shape=(data.shape[1], config['model']['seq_len'], config['model']['input_dim']))
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
    for i in range(_rnn_input.shape[1] - int(config['model']['seq_len'] / 3.0)):
        mu = np.sum(_labels[:, i + 1:i + int(config['model']['seq_len'] / 3.0) + 1], axis=1) / int(
            config['model']['seq_len'] / 3.0)

        h = np.arange(1, int(config['model']['seq_len'] / 3.0) + 1)

        rho = (1 / (np.log(int(config['model']['seq_len'] / 3.0)) + 1)) * np.sum(
            _labels[:, i + 1:i + int(config['model']['seq_len'] / 3.0) + 1] / h, axis=1)

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


def data_correction_v4(rnn_input, pred_backward, labels, fw_mon_ratio, seq_len):
    # Shape = (#n_flows, #time-steps)

    r = int(seq_len / 3)

    _rnn_input = np.copy(rnn_input.T)
    _labels = np.copy(labels.T)

    beta = np.zeros(_rnn_input.shape)
    for i in range(_rnn_input.shape[1] - r):
        mu = np.sum(_labels[:, i + 1:i + r + 1], axis=1) / r

        h = np.arange(1, r + 1)

        rho = (1 / (np.log(r) + 1)) * np.sum(_labels[:, i + 1:i + r + 1] / h, axis=1)

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


def predict_fwbw_lstm_ims_v2(initial_data, initial_labels, model, seq_len, horizon):
    ims_tm_pred = np.zeros(shape=(initial_data.shape[0] + horizon, initial_data.shape[1]))
    ims_tm_pred[0:initial_data.shape[0]] = initial_data

    labels = np.zeros(shape=(initial_data.shape[0] + horizon, initial_data.shape[1]))
    labels[0:initial_labels.shape[0]] = initial_labels

    for ts_ahead in range(horizon):
        rnn_input = prepare_input_online_prediction(data=ims_tm_pred[ts_ahead:ts_ahead + seq_len],
                                                    labels=labels[ts_ahead:ts_ahead + seq_len])
        fw_outputs, _ = model.predict(rnn_input)

        fw_outputs = np.squeeze(fw_outputs, axis=2)  # Shape(#n_flows, #time-steps)

        pred_next_tm = np.copy(fw_outputs[:, -1])

        # corrected_data = data_correction_v3(rnn_input=np.copy(ims_tm_pred[ts_ahead: ts_ahead + config['model']['seq_len']]),
        #                                     pred_backward=bw_outputs,
        #                                     labels=labels[ts_ahead: ts_ahead + config['model']['seq_len']])
        #
        # measured_data = ims_tm_pred[ts_ahead:ts_ahead + config['model']['seq_len'] - 1] * \
        #                 labels[ts_ahead:ts_ahead + config['model']['seq_len'] - 1]
        # pred_data = corrected_data * (1.0 - labels[ts_ahead:ts_ahead + config['model']['seq_len'] - 1])
        # ims_tm_pred[ts_ahead:ts_ahead + config['model']['seq_len'] - 1] = measured_data + pred_data

        ims_tm_pred[ts_ahead + seq_len] = pred_next_tm

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


def set_measured_flow(rnn_input, pred_forward, labels, mon_ratio):
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
    m = int(mon_ratio * n_flows)

    w = w.flatten()
    sorted_idx_w = np.argsort(w)
    sampling[sorted_idx_w[:m]] = 1

    return sampling


def set_measured_flow_fairness(rnn_input, labels, mon_ratio):
    """

    :param rnn_input: shape(#n_flows, #time-steps)
    :param labels: shape(n_flows, #time-steps)
    :return:
    """

    n_flows = rnn_input.shape[0]

    cl = calculate_consecutive_loss(labels).astype(float)

    w = 1 / cl

    sampling = np.zeros(shape=n_flows)
    m = int(mon_ratio * n_flows)

    w = w.flatten()
    sorted_idx_w = np.argsort(w)
    sampling[sorted_idx_w[:m]] = 1

    return sampling


def calculate_flows_weights(fw_losses, measured_matrix, flow_slection_params):
    """

    :param fw_losses: shape(#n_flows)
    :param measured_matrix: shape(#n_flows, #time-steps)
    :return: w: flow weight shape(#n_flows)
    """

    cl = calculate_consecutive_loss(measured_matrix).astype(float)

    w = 1 / (fw_losses * flow_slection_params[0] +
             cl * flow_slection_params[1])

    return w


def predict_fwbw_lstm_v2(initial_data, test_data, model, seq_len, horizon, mon_ratio, flow_selection):
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
    ims_tm = np.zeros(shape=(test_data.shape[0] - horizon + 1, test_data.shape[1]))

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

        if ts < test_data.shape[0] - horizon + 1:
            ims_tm[ts] = predict_fwbw_lstm_ims_v2(initial_data=np.copy(tm_pred[ts: ts + seq_len]),
                                                  initial_labels=np.copy(labels[ts: ts + seq_len]),
                                                  model=model, seq_len=seq_len, horizon=horizon)

        # Create 3D input for rnn
        # Shape(#n_flows, #time-steps, #features)
        rnn_input = prepare_input_online_prediction(data=tm_pred[ts: ts + seq_len],
                                                    labels=labels[ts: ts + seq_len])

        fw_outputs, bw_outputs = model.predict(rnn_input)  # Shape(#n_flows, #time-step, 1)

        fw_outputs = np.squeeze(fw_outputs, axis=2)  # Shape(#n_flows, #time-step)

        pred_next_tm = np.copy(fw_outputs[:, -1])  # Shape(#n_flows,)

        # Insert pred_next_tm to predicted traffic matrices
        predicted_tm[ts] = pred_next_tm

        # Data Correction: Shape(#time-steps, flows) for [ts+1 : ts + config['model']['seq_len'] - 1]
        corrected_data = data_correction_v3(rnn_input=np.copy(tm_pred[ts: ts + seq_len]),
                                            pred_backward=bw_outputs,
                                            labels=labels[ts: ts + seq_len])
        measured_data = tm_pred[ts:ts + seq_len - 1] * labels[ts:ts + seq_len - 1]
        pred_data = corrected_data * (1.0 - labels[ts:ts + seq_len - 1])
        tm_pred[ts:ts + seq_len - 1] = measured_data + pred_data

        # Partial monitoring
        if flow_selection == 'Random':
            sampling = np.random.choice(tf_a, size=(test_data.shape[1]),
                                        p=[mon_ratio, 1 - mon_ratio])
        elif flow_selection == 'Fairness':
            sampling = set_measured_flow_fairness(rnn_input=np.copy(tm_pred[ts: ts + seq_len].T),
                                                  labels=labels[ts: ts + seq_len].T)
        else:
            sampling = set_measured_flow(rnn_input=np.copy(tm_pred[ts: ts + seq_len].T),
                                         pred_forward=fw_outputs,
                                         labels=labels[ts: ts + seq_len].T)

        new_input = pred_next_tm * (1.0 - sampling) + test_data[ts] * sampling

        tm_pred[ts + seq_len] = new_input
        labels[ts + seq_len] = sampling

        prediction_times.append(time.time() - _start)

    dump_prediction_time['time_step'] = range(test_data.shape[0])
    dump_prediction_time['pred_time'] = prediction_times
    dump_prediction_time.to_csv(
        config['test']['results_path'] + '{}-{}-{}-{}/Prediction_times.csv'.format(config['data']['data_name'],
                                                                                   config['alg'],
                                                                                   Config.TAG,
                                                                                   Config.SCALER),
        index=False)

    return tm_pred[seq_len:], labels[seq_len:], ims_tm, predicted_tm[seq_len:]


def build_model(config):
    print('|--- Build models fwbw-lstm.')

    # fwbw-lstm model
    fwbw_net = FwbwLstmRegression(**config)
    fwbw_net.construct_fwbw_lstm()
    # print(fwbw_net.model.summary())
    fwbw_net.plot_models()
    return fwbw_net


def train_fwbw_lstm(config):
    print('|-- Run model training fwbw_lstm.')

    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        fwbw_net = build_model(config)

    fwbw_net.train()

    return


def evaluate_fwbw_lstm(config):
    print('|--- EVALUATE FWBW-LSTM')
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        fwbw_net = build_model(config)

    fwbw_net.load()
    fwbw_net.evaluate()


def test_fwbw_lstm(config):
    print('|--- TEST FWBW-LSTM')
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        fwbw_net = build_model(config)
    fwbw_net.load()
    fwbw_net.test()


def ims_tm_test_data(test_data, horizon):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - horizon + 1, test_data.shape[1]))

    for i in range(horizon - 1, test_data.shape[0], 1):
        ims_test_set[i - horizon + 1] = test_data[i]

    return ims_test_set


def prepare_test_set_last_5days(test_data2d, test_data_normalized2d, day_size, seq_len):

    idx = test_data2d.shape[0] - day_size * 5 - 10

    test_data_normalize = np.copy(test_data_normalized2d[idx:idx + day_size * 5])
    init_data_normalize = np.copy(test_data_normalized2d[idx - seq_len: idx])
    test_data = test_data2d[idx:idx + day_size * 5]

    return test_data_normalize, init_data_normalize, test_data


def run_test(test_data2d, test_data_normalized2d, fwbw_net, scalers, results_summary):
    mape, r2_score, rmse = [], [], []
    mape_ims, r2_score_ims, rmse_ims = [], [], []

    # per_gain = []

    for i in range(config['test']['run_times']):
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

        if config['model']['horizon']:
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

    results_summary['No.'] = range(config['test']['run_times'])
    results_summary['mape'] = mape
    results_summary['r2'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['mape_ims'] = mape_ims
    results_summary['r2_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    print('Test: {}-{}-{}-{}-{}'.format(config['data']['data_name'], config['alg'], Config.TAG, Config.SCALER,
                                        config['test']['flow_selection']))

    print('avg_err: {} - avg_rmse: {} - avg_r2: {}'.format(np.mean(np.array(mape)),
                                                           np.mean(np.array(rmse)),
                                                           np.mean(np.array(r2_score))))

    return results_summary
