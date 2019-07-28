from math import sqrt

import numpy as np
from scipy.stats import sem, t
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CONFIDENCE = 0.95



def flow_measurement_percentage(measured_matrix):
    """
    Calculate the measurement percentage of the flow over the lookback
    :param measured_matrix:
    :return:
    """
    labels = measured_matrix.astype(int)

    count_measurement = np.count_nonzero(labels, axis=1).astype(float)

    return count_measurement / labels.shape[1]


def recovery_loss(rnn_input, rnn_updated, measured_matrix):
    """
    Calculate the recovery loss for each flow using this equation: r_l = sqrt(sum((y_true - y_pred)^2))
    :param rnn_input: array-like, shape = (od x look_back)
    :param rnn_updated: array-like, shape = (od x look_back)
    :param measured_matrix: array-like, shape = (od x look_back)
    :return: shape = (od, )
    """
    labels = measured_matrix.astype(int)
    r_l = []
    for flow_id in range(rnn_input.shape[0]):
        flow_label = labels[flow_id, :]
        n_measured_data = np.count_nonzero(flow_label)
        if n_measured_data == 0:  # If no measured data point ==> negative recovery loss
            r_l.append(labels.shape[1])
        else:
            # Only consider the data point which is measured
            observated_idx = np.where(flow_label == 1)
            flow_true = rnn_input[flow_id, :]
            flow_pred = rnn_updated[flow_id, :]
            r_l.append(sqrt(np.sum(np.square(flow_true[observated_idx] - flow_pred[observated_idx]))))

    r_l = np.asarray(r_l)
    r_l[r_l < 0] = r_l.max()

    return r_l


def calculate_r2_score(y_true, y_pred):
    y_true_flatten = y_true.flatten()
    y_pred_flatten = y_pred.flatten()

    r2 = r2_score(y_true=y_true_flatten, y_pred=y_pred_flatten)
    return r2


def calculate_mape(y_true, y_pred):

    y_true_flatten = y_true.flatten()
    y_pred_flatten = y_pred.flatten()

    y_true_flatten[y_true_flatten == 0] = 10e-5

    mape = (np.sum(np.abs((y_true_flatten - y_pred_flatten) / y_true_flatten))) / np.size(y_true_flatten)

    return mape


def recovery_loss_3d(rnn_input, rnn_updated, measured_matrix):
    labels = measured_matrix.astype(int)
    r_l = []
    for flow_id_i in range(rnn_input.shape[1]):
        for flow_id_j in range(rnn_input.shape[2]):
            flow_label = labels[:, flow_id_i, flow_id_j]
            n_measured_data = np.count_nonzero(flow_label)
            if n_measured_data == 0:
                r_l.append(-1)
            else:
                # Only consider the data point which is measured
                observated_idx = np.where(flow_label == 1)
                flow_true = rnn_input[:, flow_id_i, flow_id_j]
                flow_pred = rnn_updated[:, flow_id_i, flow_id_j]
                r_l.append(sqrt(np.sum(np.square(flow_true[observated_idx] - flow_pred[observated_idx]))))

    r_l = np.asarray(r_l)
    r_l[r_l < 0] = r_l.max() + 1
    r_l = np.reshape(r_l, newshape=(rnn_input.shape[1], rnn_input.shape[2]))

    return r_l


def calculate_rmse(y_true, y_pred):
    ytrue_flatten = y_true.flatten()
    ypred_flatten = y_pred.flatten()
    rmse = sqrt(mean_squared_error(y_true=ytrue_flatten, y_pred=ypred_flatten))
    return rmse


def mean_abs_error(y_true, y_pred):
    ytrue = y_true.flatten()
    ypred = y_pred.flatten()
    return mean_absolute_error(y_true=ytrue, y_pred=ypred)


def calculate_error_ratio_by_day(y_true, y_pred, sampling_itvl, measured_matrix):
    day_size = 24 * (60 / sampling_itvl)
    ndays = y_true.shape[0] / (day_size)

    errors_by_day = []

    for day in range(ndays):
        upperbound = (day + 1) * day_size if (day + 1) * day_size < y_true.shape[0] else y_true.shape[0]
        ytrue_by_day = y_true[day * day_size:upperbound, :]
        y_pred_by_day = y_pred[day * day_size:upperbound, :]
        measured_matrix_by_day = measured_matrix[day * day_size:upperbound, :]
        errors_by_day.append(
            error_ratio(y_true=ytrue_by_day, y_pred=y_pred_by_day, measured_matrix=measured_matrix_by_day))

    return errors_by_day


def error_ratio(y_true, y_pred, measured_matrix):
    y_true_flatten = y_true.flatten()
    y_pred_flatten = y_pred.flatten()
    measured_matrix = measured_matrix.flatten()
    observated_indice = np.where(measured_matrix == 0.0)

    e1 = sqrt(np.sum(np.square(y_true_flatten[observated_indice] - y_pred_flatten[observated_indice])))
    e2 = sqrt(np.sum(np.square(y_true_flatten[observated_indice])))
    if e2 == 0:
        return 0
    else:
        return e1 / e2


def calculate_confident_interval(data):
    n = len(data)
    std_err = sem(data)

    h = std_err * t.ppf((1 + CONFIDENCE) / 2, n - 1)

    return h


def normalized_mean_absolute_error(y_true, y_hat):
    """
    Calculate the normalized mean absolute error
    :param y_true:
    :param y_hat:
    :return:
    """
    mae_y_yhat = mean_absolute_error(y_true=y_true, y_pred=y_hat)
    mae_y_zero = mean_absolute_error(y_true=y_true, y_pred=np.zeros(shape=y_true.shape))
    if mae_y_zero == 0:
        return 0
    else:
        return mae_y_yhat / mae_y_zero


def normalized_mean_squared_error(y_true, y_hat):
    mse_y_yhat = mean_squared_error(y_true=y_true, y_pred=y_hat)
    mse_y_zero = mean_squared_error(y_true=y_true, y_pred=np.zeros(shape=y_true.shape))
    if mse_y_zero == 0:
        return 0
    else:
        return mse_y_yhat / mse_y_zero
