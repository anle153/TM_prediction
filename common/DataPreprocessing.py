from multiprocessing import Process, Pipe, cpu_count

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer

from FlowClassification.SpatialClustering import *
from common.error_utils import calculate_mape


def prepare_train_test_3d(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days * 0.8)

    train_set = data[0:train_size * day_size, :, :]
    test_set = data[train_size * day_size:, :, :]

    return train_set, test_set


def prepare_train_valid_test_3d(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days * 0.6 * day_size)

    valid_size = int(n_days * 0.2 * day_size)

    train_set = data[0:train_size]
    valid_set = data[train_size:(train_size + valid_size)]
    test_set = data[(train_size + valid_size):]

    return train_set, valid_set, test_set


def prepare_train_test_2d(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days * 0.8)

    train_set = data[0:train_size * day_size, :]
    test_set = data[train_size * day_size:, :]

    return train_set, test_set


def prepare_test_one_week(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days - 1)

    train_set = data[0:train_size * day_size]
    test_set = data[-1 * day_size:]

    return train_set, test_set


def prepare_train_valid_test_2d(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days * 0.6)

    valid_size = int(n_days * 0.2)

    train_set = data[0:train_size * day_size, :]
    valid_set = data[train_size * day_size:(train_size * day_size + valid_size * day_size), :]
    test_set = data[(train_size * day_size + valid_size * day_size):, :]

    return train_set, valid_set, test_set


########################################################################################################################
#                                        Generator training data                                                       #

def create_offline_fwbw_conv_lstm_data_fix_ratio(data, input_shape, mon_ratio, eps, data_time=1):
    _tf = np.array([1.0, 0.0])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]

    data_x = np.zeros(((data.shape[0] - ntimesteps - 1) * data_time, ntimesteps, wide, high, channel))
    data_y_1 = np.zeros(((data.shape[0] - ntimesteps - 1) * data_time, ntimesteps, wide * high))
    data_y_2 = np.zeros(((data.shape[0] - ntimesteps - 1) * data_time, ntimesteps, wide * high))

    for time in range(data_time):
        _labels = np.random.choice(_tf,
                                   size=data.shape,
                                   p=(mon_ratio, 1 - mon_ratio))
        _data = np.copy(data)

        _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

        _traffic_labels = np.zeros((_data.shape[0], wide, high, channel))
        _traffic_labels[:, :, :, 0] = _data
        _traffic_labels[:, :, :, 1] = _labels

        for idx in range(1, _traffic_labels.shape[0] - ntimesteps):
            _x = _traffic_labels[idx: (idx + ntimesteps)]

            data_x[idx + time * (data.shape[0] - ntimesteps - 1) - 1] = _x

            _y = data[(idx + 1):(idx + ntimesteps + 1)]
            _y = np.reshape(_y, newshape=(ntimesteps, wide * high))

            _y_2 = data[(idx - 1):(idx + ntimesteps - 1)]
            _y_2 = np.reshape(np.flip(_y_2, axis=0), newshape=(ntimesteps, wide * high))

            data_y_1[idx + time * (data.shape[0] - ntimesteps - 1) - 1] = _y

            data_y_2[idx + time * (data.shape[0] - ntimesteps - 1) - 1] = _y_2

    return data_x, data_y_1, data_y_2


def create_offline_conv_lstm_data_fix_ratio(data, input_shape, mon_ratio, eps, data_time=1):

    _tf = np.array([1.0, 0.0])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]
    data_x = np.zeros(
        ((data.shape[0] - ntimesteps) * data_time, ntimesteps, wide, high, channel))
    data_y = np.zeros(((data.shape[0] - ntimesteps) * data_time, wide * high))

    for time in range(data_time):
        _labels = np.random.choice(_tf,
                                   size=data.shape,
                                   p=(mon_ratio, 1 - mon_ratio))
        _data = np.copy(data)

        _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

        _traffic_labels = np.zeros((_data.shape[0], wide, high, channel))
        _traffic_labels[:, :, :, 0] = _data
        _traffic_labels[:, :, :, 1] = _labels

        for idx in range(_traffic_labels.shape[0] - ntimesteps):
            _x = _traffic_labels[idx: (idx + ntimesteps)]

            data_x[idx + time * (data.shape[0] - ntimesteps)] = _x

            _y = data[idx + ntimesteps]
            _y = np.reshape(_y, newshape=(wide * high))

            data_y[idx + time * (data.shape[0] - ntimesteps)] = _y

    return data_x, data_y


def create_offline_cnnlstm_data_fix_ratio(data, input_shape, mon_ratio, eps, data_time=1):

    _tf = np.array([1.0, 0.0])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]
    data_x = np.zeros(
        ((data.shape[0] - ntimesteps) * data_time, ntimesteps, wide, high, channel))
    data_y = np.zeros(((data.shape[0] - ntimesteps) * data_time, ntimesteps, wide * high))

    for time in range(data_time):
        _labels = np.random.choice(_tf,
                                   size=data.shape,
                                   p=(mon_ratio, 1 - mon_ratio))
        _data = np.copy(data)

        _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

        _traffic_labels = np.zeros((_data.shape[0], wide, high, channel))
        _traffic_labels[:, :, :, 0] = _data
        _traffic_labels[:, :, :, 1] = _labels

        for idx in range(_traffic_labels.shape[0] - ntimesteps):
            _x = _traffic_labels[idx: (idx + ntimesteps)]

            data_x[idx + time * (data.shape[0] - ntimesteps)] = _x

            _y = data[(idx + 1):(idx + ntimesteps + 1)]
            _y = np.reshape(_y, newshape=(ntimesteps, wide * high))

            data_y[idx + time * (data.shape[0] - ntimesteps)] = _y

    return data_x, data_y


def create_offline_fwbw_convlstm_data(data, input_shape, mon_ratio, eps, data_time=1):

    _tf = np.array([1.0, 0.0])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]
    data_x = np.zeros(
        ((data.shape[0] - ntimesteps) * data_time, ntimesteps, wide, high, channel))
    data_y_1 = np.zeros(((data.shape[0] - ntimesteps) * data_time, wide * high))
    data_y_2 = np.zeros(((data.shape[0] - ntimesteps) * data_time, ntimesteps - 2, wide * high))

    for time in range(data_time):
        _labels = np.random.choice(_tf,
                                   size=data.shape,
                                   p=(mon_ratio, 1 - mon_ratio))
        _data = np.copy(data)

        _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

        _traffic_labels = np.zeros((_data.shape[0], wide, high, channel))
        _traffic_labels[:, :, :, 0] = _data
        _traffic_labels[:, :, :, 1] = _labels

        for idx in range(_traffic_labels.shape[0] - ntimesteps):
            _x = _traffic_labels[idx: (idx + ntimesteps)]

            data_x[idx + time * (data.shape[0] - ntimesteps)] = _x

            _y = data[(idx + 1):(idx + ntimesteps - 1)]
            _y = np.reshape(_y, newshape=(ntimesteps - 2, wide * high))

            data_y_1[idx + time * (data.shape[0] - ntimesteps)] = data[idx + ntimesteps].flatten()
            data_y_2[idx + time * (data.shape[0] - ntimesteps)] = _y

    return data_x, data_y_1, data_y_2


def create_offline_lstm_nn_data(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf, size=data.shape, p=(mon_ratio, 1 - mon_ratio))
    data_x = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, features))
    data_y = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, 1))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            data_x[i, :, 0] = _x
            data_x[i, :, 1] = _label

            _y = data[(idx + 1):(idx + ntimesteps + 1), flow]

            data_y[i] = np.array(_y).reshape((ntimesteps, 1))

            i += 1

    return data_x, data_y


def create_offline_reslstm_nn_data(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    data_x = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, features))
    data_y = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], 1))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            data_x[i, :, 0] = _x
            data_x[i, :, 1] = _label

            data_y[i] = data[idx + ntimesteps, flow]

            i += 1

    return data_x, data_y


def create_offline_res_lstm_2_data(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    data_x_1 = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, features))
    data_x_2 = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, 1))
    data_y = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], 1))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            data_x_1[i, :, 0] = _x
            data_x_2[i] = np.reshape(_x, newshape=(ntimesteps, 1))
            data_x_1[i, :, 1] = _label

            data_y[i] = data[idx + ntimesteps, flow]

            i += 1

    return data_x_1, data_x_2, data_y


def create_offline_fwbw_lstm_2(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    data_x = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, features))
    data_y_1 = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps))
    data_y_2 = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps - 2))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            data_x[i, :, 0] = _x
            data_x[i, :, 1] = _label

            data_y_1[i] = data[(idx + 1):(idx + ntimesteps + 1), flow]
            data_y_2[i] = data[(idx + 1):(idx + ntimesteps - 1), flow]
            i += 1

    return data_x, data_y_1, data_y_2


def create_offline_fwbw_lstm(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    data_x = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps, features))
    data_y_1 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps, 1))
    data_y_2 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(1, _data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            data_x[i, :, 0] = _x
            data_x[i, :, 1] = _label

            _y_1 = data[(idx + 1):(idx + ntimesteps + 1), flow]
            _y_2 = data[(idx - 1):(idx + ntimesteps - 1), flow]

            data_y_1[i] = np.reshape(_y_1, newshape=(ntimesteps, 1))
            data_y_2[i] = _y_2
            i += 1

    return data_x, data_y_1, data_y_2


def create_offline_fwbw_lstm_no_sc(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    data_x = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps, features))
    data_y_1 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps, 1))
    data_y_2 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps, 1))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(1, _data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            data_x[i, :, 0] = _x
            data_x[i, :, 1] = _label

            _y_1 = data[(idx + 1):(idx + ntimesteps + 1), flow]
            _y_2 = data[(idx - 1):(idx + ntimesteps - 1), flow]

            data_y_1[i] = np.reshape(_y_1, newshape=(ntimesteps, 1))
            data_y_2[i] = np.reshape(_y_2, newshape=(ntimesteps, 1))
            i += 1

    return data_x, data_y_1, data_y_2


def create_offline_res_fwbw_lstm(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    data_x_1 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps, features))
    data_x_2 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps, 1))
    data_y_1 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps))
    data_y_2 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(1, _data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            data_x_1[i, :, 0] = _x
            data_x_1[i, :, 1] = _label

            data_x_2[i] = np.reshape(_x, newshape=(ntimesteps, 1))

            _y_1 = data[(idx + 1):(idx + ntimesteps + 1), flow]
            _y_2 = data[(idx - 1):(idx + ntimesteps - 1), flow]

            data_y_1[i] = _y_1
            data_y_2[i] = _y_2
            i += 1

    return data_x_1, data_x_2, data_y_1, data_y_2


def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


def calc_change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)


def create_xgb_features(x):
    x_step = []
    x_step.append(x.mean())
    x_step.append(x.std())
    x_step.append(x.max())
    x_step.append(x.min())

    x_step.append(np.mean(np.diff(x)))
    x_step.append(calc_change_rate(x))
    x_step.append(np.abs(x).max())
    x_step.append(np.abs(x).min())

    # x_step.append(x.max() / np.abs(x.min()))
    # x_step.append(x.max() - np.abs(x.min()))
    # x_step.append(len(x[np.abs(x) > 1000000]))
    x_step.append(x.sum())

    x_step.append(np.quantile(x, 0.95))
    x_step.append(np.quantile(x, 0.99))
    x_step.append(np.quantile(x, 0.05))
    x_step.append(np.quantile(x, 0.01))

    # x_step.append(np.quantile(np.abs(x), 0.95))
    # x_step.append(np.quantile(np.abs(x), 0.99))
    # x_step.append(np.quantile(np.abs(x), 0.05))
    # x_step.append(np.quantile(np.abs(x), 0.01))

    x_step.append(add_trend_feature(x))
    x_step.append(add_trend_feature(x, abs_values=True))
    # x_step.append(np.abs(x).mean())
    # x_step.append(np.abs(x).std())

    x_step.append(x.mad())
    x_step.append(x.kurtosis())
    x_step.append(x.skew())
    x_step.append(x.median())

    return x_step


def create_offline_xgb_data(data, ntimesteps, features, mon_ratio, eps, connection, proc_id):
    _tf = np.array([True, False])
    measured_matrix = np.random.choice(_tf,
                                       size=data.shape,
                                       p=(mon_ratio, 1 - mon_ratio))
    _labels = measured_matrix.astype(int)
    data_x = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], features))
    data_y = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1]))

    _data = np.copy(data)

    _data[_labels == 0] = np.random.uniform(_data[_labels == 0] - eps, _data[_labels == 0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - ntimesteps - 1):
            _x = _data[idx: (idx + ntimesteps), flow]
            _x = pd.Series(_x)
            data_x[i, :] = np.array(create_xgb_features(_x))

            _y = _data[(idx + ntimesteps + 1), flow]

            data_y[i] = _y

            i += 1

    print("[PROC_ID: %d] Sending result" % proc_id)
    connection.send([data_x, data_y])
    connection.close()
    print("[PROC_ID] RESULT SENT")


def parallel_create_offline_xgb_data(data, ntimesteps, features, mon_ratio, eps):
    nproc = cpu_count()
    quota = int(data.shape[1] / nproc)

    p = [0] * nproc

    connections = []

    for proc_id in range(nproc):
        connections.append(Pipe())

    data_x = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], features))

    data_y = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1]))
    ret_xy = []

    for proc_id in range(nproc):
        data_quota = data[:, proc_id * quota:(proc_id + 1) * quota] if proc_id < (nproc - 1) else data[:,
                                                                                                  proc_id * quota:]
        p[proc_id] = Process(target=create_offline_xgb_data,
                             args=(data_quota,
                                   ntimesteps,
                                   features,
                                   mon_ratio,
                                   eps,
                                   connections[proc_id][1],
                                   proc_id))
        p[proc_id].start()

    for proc_id in range(nproc):
        ret_xy.append(connections[proc_id][0].recv())
        p[proc_id].join()

    for proc_id in range(nproc):
        _start = proc_id * quota * (data.shape[0] - ntimesteps)
        _end = (proc_id + 1) * quota * (data.shape[0] - ntimesteps) if proc_id < (nproc - 1) \
            else (data.shape[0] - ntimesteps) * data.shape[1]

        data_x[_start: _end] = ret_xy[proc_id][0]
        data_y[_start: _end] = ret_xy[proc_id][1]

    return data_x, data_y


########################################################################################################################
#                                                Data scalling                                                         #


class sd_scale():
    def __init__(self):
        self.fit_data = None
        self.__mean = None
        self.__std = None

    def fit(self, data):
        self.fit_data = data
        self.__mean = np.mean(data)
        self.__std = np.std(data)

    def transform(self, data):
        assert self.__mean is not None
        return (data - self.__mean) / self.__std

    def inverse_transform(self, data):
        assert self.__mean is not None
        return (data * self.__std) + self.__mean


def data_scalling(train_data2d):
    scaler = PowerTransformer()
    scaler.fit(train_data2d)
    # train_data_normalized2d = scaler.transform(train_data2d)
    # valid_data_normalized2d = scaler.transform(valid_data2d)
    # test_data_normalized2d = scaler.transform(test_data2d)

    return scaler


def results_processing(tm_true, run_times, path):
    predicted_error = pd.DataFrame(index=range(run_times),
                                   columns=['No.', 'mape', 'mse', 'r2'])
    rets = []

    for i in range(run_times):
        tm_pred = np.load(path + 'Predicted_tm_{}.npy'.format(i))
        mape = calculate_mape(y_true=tm_true, y_pred=tm_pred)
        mse = mean_squared_error(y_true=tm_true, y_pred=tm_pred)
        r2 = r2_score(y_true=tm_true, y_pred=tm_pred)

        rets.append([mape, mse, r2])

    rets = np.asarray(rets)
    predicted_error['No.'] = range(run_times)
    predicted_error['mape'] = rets[:, 0]
    predicted_error['mse'] = rets[:, 1]
    predicted_error['r2'] = rets[:, 2]

    predicted_error.to_csv(path + 'Predicted_error.csv')

    return
