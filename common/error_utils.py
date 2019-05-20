import datetime
import os
from math import sqrt

import matplotlib as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def plot_errors(x_axis, xlabel, errors, filename, title='', saving_path='/home/anle/TM_estimation_figures/'):
    now = datetime.datetime.now()

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    plt.title('Errors\n' + title)
    plt.plot(x_axis, errors[:, 0], label='NMAE')
    plt.plot(x_axis, errors[:, 1], label='NMSE')
    plt.xlabel(xlabel)
    if errors.shape[1] == 4:
        plt.plot(x_axis, errors[:, 3], label='Error_ratio')
    plt.legend()

    plt.savefig(saving_path + str(now) + '_Errors_' + filename)
    plt.close()

    plt.title('R2-Score')
    plt.plot(x_axis, errors[:, 2])
    plt.xlabel(xlabel)
    plt.savefig(saving_path + str(now) + '_R2_Score_' + filename)
    plt.close()
    print('--- Saving figures at %s ---' % saving_path)


def calculate_measured_weights(rnn_input, forward_pred, backward_pred, measured_matrix, hyperparams):
    """
    Calculated measured weight for determine which flows should be measured in next time slot.
    We measured first K flows which have small weight
    The weight is calculated based on the formular: w = (1/rlf) + (1/rlb) + mp + f + 1/cl
        w: the measurement weight
        rlf: the recovery loss forward rnn
        rlb: the recovery loss backward rnn
        mp: data point measurement percentage
        f: measure how fluctuate the flow is
        cl: measure the consecutive loss
    :param rnn_input:
    :param forward_pred:
    :param backward_pred: the backward pred has been flipped
    :param measured_matrix: shape = (od x timeslot)
    :return:
    """

    eps = 10e-5

    rnn_first_input_updated = np.expand_dims(backward_pred[:, 1], axis=1)
    rnn_last_input_updated = np.expand_dims(forward_pred[:, -2], axis=1)
    rnn_updated_input_forward = np.concatenate([rnn_first_input_updated, forward_pred[:, 0:-2], rnn_last_input_updated],
                                               axis=1)
    rnn_updated_input_backward = np.concatenate([rnn_first_input_updated, backward_pred[:, 2:], rnn_last_input_updated],
                                                axis=1)

    rl_forward = recovery_loss(rnn_input=rnn_input, rnn_updated=rnn_updated_input_forward,
                               measured_matrix=measured_matrix)
    rl_forward[rl_forward == 0] = eps

    rl_backward = recovery_loss(rnn_input=rnn_input, rnn_updated=rnn_updated_input_backward,
                                measured_matrix=measured_matrix)
    rl_backward[rl_backward == 0] = eps

    cl = calculate_consecutive_loss(measured_matrix).astype(float)

    flows_stds = np.std(rnn_input, axis=1)

    w = 1 / (rl_forward * hyperparams[0] +
             rl_backward * hyperparams[1] +
             cl * hyperparams[2] +
             flows_stds * hyperparams[3])

    return w


def flow_measurement_percentage(measured_matrix):
    """
    Calculate the measurement percentage of the flow over the lookback
    :param measured_matrix:
    :return:
    """
    labels = measured_matrix.astype(int)

    count_measurement = np.count_nonzero(labels, axis=1).astype(float)

    return count_measurement / labels.shape[1]


def calculate_consecutive_loss(measured_matrix):
    """
    Calculate the last consecutive loss count from the last time slot
    :param measured_matrix:
    :return:
    """
    labels = measured_matrix.astype(int)

    consecutive_losses = []
    for flow_id in range(labels.shape[0]):
        flows_labels = labels[flow_id, :]
        if flows_labels[-1] == 1:
            consecutive_losses.append(1)
        else:
            measured_idx = np.argwhere(flows_labels == 1)
            if measured_idx.size == 0:
                consecutive_losses.append(labels.shape[1])
            else:
                consecutive_losses.append(labels.shape[1] - measured_idx[-1][0])

    consecutive_losses = np.asarray(consecutive_losses)
    return consecutive_losses


def calculate_consecutive_loss_3d(measured_matrix):
    """
    Calculate the last consecutive loss count from the last time slot
    :param measured_matrix: shape=(time, od, od)
    :return:
    """
    labels = measured_matrix.astype(int)

    consecutive_losses = []
    for flow_id_i in range(labels.shape[1]):
        for flow_id_j in range(labels.shape[2]):
            flows_labels = labels[:, flow_id_i, flow_id_j]
            if flows_labels[-1] == 1:
                consecutive_losses.append(1)
            else:
                measured_idx = np.argwhere(flows_labels == 1)
                if measured_idx.size == 0:
                    consecutive_losses.append(labels.shape[0])
                else:
                    consecutive_losses.append(labels.shape[0] - measured_idx[-1][0])

    consecutive_losses = np.asarray(consecutive_losses)
    consecutive_losses = np.reshape(consecutive_losses, newshape=(labels.shape[1], labels.shape[2]))
    return consecutive_losses


def mean_absolute_errors_by_day(y_true, y_pred, sampling_itvl):
    """
    Calculate the mean absolute error of the traffic matrix within a day
    :param y_true:
    :param y_pred:
    :param sampling_itvl:
    :return:
    """
    day_size = 24 * (60 / sampling_itvl)
    ndays = y_true.shape[0] / (day_size)

    mean_abs_errors_by_day = []
    for day in range(ndays):
        upperbound = (day + 1) * day_size if (day + 1) * day_size < y_true.shape[0] else y_true.shape[0]
        ytrue_by_day = y_true[day * day_size:upperbound, :]
        y_pred_by_day = y_pred[day * day_size:upperbound, :]
        mean_abs_errors_by_day.append(mean_abs_error(y_true=ytrue_by_day, y_pred=y_pred_by_day))
    return mean_abs_errors_by_day


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


def recovery_loss_3d(rnn_input, rnn_updated, measured_matrix):
    """
    Calculate the recovery loss for each flow using this equation: r_l = sqrt(sum((y_true - y_pred)^2))
    :param rnn_input: array-like, shape = (time x od x od)
    :param rnn_updated: array-like, shape = (time x od x od)
    :param measured_matrix: array-like, shape = (time x od x od)
    :return: shape = (od, od)
    """
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


def root_means_squared_error_by_day(y_true, y_pred, sampling_itvl):
    day_size = 24 * (60 / sampling_itvl)
    ndays = y_true.shape[0] / (day_size)

    rmse_by_day = []
    for day in range(ndays):
        upperbound = (day + 1) * day_size if (day + 1) * day_size < y_true.shape[0] else y_true.shape[0]
        ytrue_by_day = y_true[day * day_size:upperbound, :]
        y_pred_by_day = y_pred[day * day_size:upperbound, :]
        rmse_by_day.append(calculate_rmse(ytrue_by_day, y_pred_by_day))
    return rmse_by_day


def calculate_rmse(y_true, y_pred):
    ytrue_flatten = y_true.flatten()
    ypred_flatten = y_pred.flatten()
    err = sqrt(np.sum(np.square(ytrue_flatten - ypred_flatten)) / ytrue_flatten.size)
    return err


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
