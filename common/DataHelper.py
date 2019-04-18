import datetime
import xml.etree.ElementTree as et
from math import sqrt, log

import scipy.io as sio
from scipy.signal import argrelextrema
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from tensorflow.python.client import device_lib

from FlowClassification.SpatialClustering import *
from common import Config

HOME = os.path.expanduser('~')


########################################################################################################################
#         Calculating Error: error_ratio, normalized mean absolute error, normalized mean squred error                 #
########################################################################################################################


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


def calculate_flow_fluctuation(rnn_input, show=False):
    """
    Calculate the value of flow fluctuation
    :param rnn_input:
    :param show:
    :return:
    """

    fluctuations = []
    for flow_id in range(rnn_input.shape[0]):
        flow = rnn_input[flow_id, :]
        # plt.plot(flow)
        # plt.title('Flow')
        # plt.show()
        # plt.close()

        _s, f, _c = dfa(flow, scale_lim=[log(rnn_input.shape[1], 2) - 1, log(rnn_input.shape[1], 2)], show=show)
        print('F = %f' % f[-1])
        fluctuations.append(f[-1])

    fluctuations = np.asarray(fluctuations)

    return fluctuations


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
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    r2 = r2_score(y_true=y_true, y_pred=y_pred)
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
        rmse_by_day.append(rmse_tm_prediction(ytrue_by_day, y_pred_by_day))
    return rmse_by_day


def rmse_tm_prediction(y_true, y_pred):
    ytrue = y_true.flatten()
    ypred = y_pred.flatten()
    err = sqrt(np.sum(np.square(ytrue - ypred)) / ytrue.size)
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
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    measured_matrix = measured_matrix.flatten()
    observated_indice = np.where(measured_matrix == False)

    e1 = sqrt(np.sum(np.square(y_true[observated_indice] - y_pred[observated_indice])))
    e2 = sqrt(np.sum(np.square(y_true[observated_indice])))
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


def path_exist(path):
    """
    Checking the dataset path
    :param path:
    :return:
    """
    return os.path.exists(path)


########################################################################################################################
#                             Loading ABILENE Traffic trace into Traffic Matrix                                        #
#                                             Number of node: 12                                                       #
########################################################################################################################


ABILENE24_DIM = 144


def convert_abilene_24(path_dir='/home/anle/Documents/sokendai/research/TM_estimation_RNN/Dataset'
                                '/Abilene_24/Abilene/2004/Measured'):
    if os.path.exists(path_dir):
        list_files = os.listdir(path_dir)
        list_files = sorted(list_files, key=lambda x: x[:-4])

        TM = np.empty((12, 12, 0))
        for raw_file in list_files:
            if raw_file.endswith('.dat'):
                print(raw_file)
                _tm = np.genfromtxt(path_dir + '/' + raw_file, delimiter=',')
                _tm = np.expand_dims(_tm, axis=2)
                TM = np.concatenate([TM, _tm], axis=2)

    print('--- Finish load original Abilene3d to csv. Saving at ./Dataset/Abilene3d')
    np.save('./Dataset/Abilene3d', TM)


def load_abilene_3d(path_dir='/home/anle/Documents/sokendai/research/TM_estimation_RNN/Dataset'
                             '/Abilene_24/Abilene/2004/Measured'):
    if os.path.exists(path_dir):
        list_files = os.listdir(path_dir)
        list_files = sorted(list_files, key=lambda x: x[:-4])

        TM = np.empty((12, 12, 0))
        for raw_file in list_files:
            if raw_file.endswith('.dat'):
                print(raw_file)
                _tm = np.genfromtxt(path_dir + '/' + raw_file, delimiter=',')
                _tm = np.expand_dims(_tm, axis=0)
                TM = np.concatenate([TM, _tm], axis=0) if TM.size else _tm

    print('--- Finish converting Abilene24 to csv. Saing at ./Dataset/Abilene24_3.csv')
    np.save(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d', TM)


def load_Abilene_dataset_from_matlab(path='./Dataset/abilene.mat'):
    """
    Load Abilene from original matlab file
    :param path: dataset path
    :return:
    """
    if path_exist(path):
        # ...
        data = sio.loadmat(path)
        X = data['X']
        A = data['A']
        odnames = data['odnames']
        edgenames = data['edgenames']
        return X
    else:
        return None, None, None, None


def load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene.csv'):
    """
    Load Abilene dataset from csv file. If file is not found, create the one from original matlab file and remove noise
    :param csv_file_path:
    :return: A traffic matrix (m x k)
    """
    if not os.path.exists(csv_file_path):
        print('--- %s not found. Create csv file from original matlab file ---' % csv_file_path)
        abilene_data = np.asarray(load_Abilene_dataset_from_matlab('./Dataset/SAND_TM_Estimation_Data.mat')) / 1000000
        # noise_removed(data=abilene_data, sampling_interval=5, threshold=30)
        np.savetxt(csv_file_path, abilene_data, delimiter=',')
        return abilene_data
    else:
        print('--- Load dataset from %s' % csv_file_path)
        return np.genfromtxt(csv_file_path, delimiter=',')


def create_abilene_data_3d(path):
    tm_3d = np.zeros(shape=(2016 * 24, 12, 12))
    for i in range(24):
        print('Read file X{:02d}'.format(i + 1))
        raw_data = np.genfromtxt(path + 'X{:02d}'.format(i + 1), delimiter=' ')
        tm = raw_data[:, range(0, 720, 5)].reshape((2016, 12, 12))
        tm_3d[i * 2016: (i + 1) * 2016, :, :] = tm

    np.save(Config.DATA_PATH + 'Abilene.npy', tm_3d)


def create_abilene_data_2d(path):
    tm_2d = np.zeros(shape=(2016 * 24, 144))
    for i in range(24):
        print('Read file X{:02d}'.format(i + 1))
        raw_data = np.genfromtxt(path + 'X{:02d}'.format(i + 1), delimiter=' ')
        tm = raw_data[:, range(0, 720, 5)]
        tm_2d[i * 2016: (i + 1) * 2016, :] = tm

    np.save(Config.DATA_PATH + 'Abilene2d.npy', tm_2d)

########################################################################################################################
#                             Loading GEANT Traffic trace into Traffic Matrix from XML files                           #
#                                                 Number of node: 23                                                   #
########################################################################################################################

MATRIX_DIM = 23
GEANT_XML_PATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices'


def get_row(xmlRow):
    """
    Parse Traffic matrix row from XLM element "src"
    :param xmlRow: XML element "src"
    :return: Traffic row corresponds to the measured traffic of a source node.
    """
    TM_row = [0] * MATRIX_DIM
    for dst in xmlRow.iter('dst'):
        dstId = int(dst.get('id'))
        TM_row[dstId - 1] = float(dst.text)

    return TM_row


def load_Geant_from_xml(datapath=GEANT_XML_PATH):
    TM = np.empty((0, MATRIX_DIM * MATRIX_DIM))

    if path_exist(datapath):
        list_files = os.listdir(datapath)
        list_files = sorted(list_files, key=lambda x: x[:-4])

        for file in list_files:
            if file.endswith(".xml"):
                print('----- Load file: %s -----' % file)
                data = et.parse(datapath + '/' + file)
                root = data.getroot()

                TM_t = []
                for src in root.iter('src'):
                    TM_row = get_row(xmlRow=src)
                    TM_t.append(TM_row)

                aRow = np.asarray(TM_t).reshape(1, MATRIX_DIM * MATRIX_DIM)
                TM = np.concatenate([TM, aRow]) if TM.size else aRow

    return TM


def load_Geant_from_csv(csv_file_path='./Dataset/Geant_noise_removed.csv'):
    """

    :param csv_file_path:
    :return:
    """
    if os.path.exists(csv_file_path):
        return np.genfromtxt(csv_file_path, delimiter=',')
    else:
        print('--- Find not found. Create Dataset from XML file ---')
        data = load_Geant_from_xml(datapath=GEANT_XML_PATH) / 1000
        noise_removed(data=data, sampling_interval=15, threshold=30)
        np.savetxt(csv_file_path, data, delimiter=",")
        return data


########################################################################################################################
#                                                 Data visualization                                                   #
########################################################################################################################


def visualize_retsult_by_flows(y_true,
                               y_pred,
                               sampling_itvl,
                               measured_matrix=[],
                               saving_path='/home/anle/TM_estimation_figures/',
                               description='',
                               visualized_day=-1,
                               show=False):
    """
    Visualize the original flows and the predicted flows over days
    :param y_true: (numpy.ndarray) the measured TM
    :param y_pred: (numpy.ndarray) the predicted TM
    :param sampling_itvl: (int) sampling interval between each sampling
    :param measured_matrix: (numpy.ndarray) identify which elements in the predicted TM are predicted using RNN
    :param saving_path: (str) path to saved figures directory
    :param description: (str) (optional) the description of this visualization
    :return:
    """

    # Get date-time when visualizing, create dir corresponding to the date-time and the description.
    import datetime
    now = datetime.datetime.now()
    description = description + '_' + str(now)
    if not os.path.exists(saving_path + description + '/'):
        os.makedirs(saving_path + description + '/')

    # Calculate no. time slots within a day and the no. days over the period
    n_ts_day = 24 * (60 / sampling_itvl)
    n_days = int(y_true.shape[0] / n_ts_day)

    # Calculate the nmse and plot both original and predicted data of each day by flow.
    path = saving_path + description + '/'

    for day in range(n_days):
        if (visualized_day != -1 and visualized_day == day) or visualized_day == -1:
            if not os.path.exists(path + 'Day%i/' % day):
                os.makedirs(path + 'Day%i/' % day)
            for flowID in range(y_true.shape[1]):
                print('--- Visualize flow %i in day %i' % (flowID, day))
                upperbound = (day + 1) * n_ts_day if (day + 1) * n_ts_day < y_true.shape[0] else y_true.shape[0]
                y1 = y_true[day * n_ts_day:upperbound, flowID]
                y2 = y_pred[day * n_ts_day:upperbound, flowID]
                sampling = measured_matrix[day * n_ts_day:upperbound, flowID]
                arg_sampling = np.argwhere(sampling == True).squeeze(axis=1)

                rmse_by_day = rmse_tm_prediction(y_true=np.expand_dims(y1, axis=1), y_pred=np.expand_dims(y2, axis=1))

                plt.title('Flow %i prediction result - Day %i \n RMSE: %f' % (flowID, day, rmse_by_day))
                plt.plot(y1, label='Original Data')
                plt.plot(y2, label='Prediction Data')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Mbps')
                # Mark the measured data in the predicted data as red start
                plt.plot(arg_sampling, y2[arg_sampling], 'r*')
                plt.savefig(path + 'Day%i/' % day + '%i.png' % flowID)
                if show:
                    plt.show()
                plt.close()


def plot_flow_acf(data):
    path = '/home/anle/TM_estimation_figures/ACF/'
    if not os.path.exists(path):
        os.makedirs(path)

    for flowID in range(data.shape[1]):
        print('--- Plotting acf of flow %i' % flowID)
        acf_plt = plot_acf(x=data[:, flowID], lags=288 * 3)
        plt.show()
        # acf_plt.savefig(path+'acf_flow_%i.png'%flowID)


def remove_zero_flow(data, eps=0.001):
    means = np.mean(data, axis=0)
    non_zero_data = data[:, means > eps]

    return non_zero_data


def get_max_acf(data, interval=5):
    day_size = 24 * (60 / interval)

    flows_acf = []
    for flow_id in range(data.shape[1]):
        flow_acf = acf(data[:, flow_id], nlags=day_size * 3)
        arg_local_max = argrelextrema(flow_acf, np.greater)
        flow_acf_local_max_index = np.argmax(flow_acf[arg_local_max[0]])
        flows_acf.append(arg_local_max[0][flow_acf_local_max_index])
        plt.plot(flow_acf)
        plt.plot(arg_local_max[0][flow_acf_local_max_index], flow_acf[arg_local_max[0][flow_acf_local_max_index]], 'r*')
        plt.show()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
