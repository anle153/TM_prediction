from multiprocessing import Process, Pipe, cpu_count

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

from FlowClassification.SpatialClustering import *
from common import Config


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

def create_offline_fwbw_conv_lstm_data_fix_ratio(data, input_shape, mon_ratio, eps, data_time=None):
    if data_time is None:
        data_time = Config.CONV_LSTM_DATA_GENERATE_TIME

    _tf = np.array([1.0, 0.0])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]
    dataX = np.zeros(((data.shape[0] - ntimesteps - 1) * data_time, ntimesteps, wide, high, channel))
    dataY_1 = np.zeros(((data.shape[0] - ntimesteps - 1) * data_time, ntimesteps, wide * high))
    dataY_2 = np.zeros(((data.shape[0] - ntimesteps - 1) * data_time, ntimesteps, wide * high))

    print(dataX.shape)

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

            dataX[idx + time * (data.shape[0] - ntimesteps - 1) - 1] = _x

            _y = data[(idx + 1):(idx + ntimesteps + 1)]
            _y = np.reshape(_y, newshape=(ntimesteps, wide * high))

            _y_2 = data[(idx - 1):(idx + ntimesteps - 1)]
            _y_2 = np.reshape(np.flip(_y_2, axis=0), newshape=(ntimesteps, wide * high))

            dataY_1[idx + time * (data.shape[0] - ntimesteps - 1) - 1] = _y

            dataY_2[idx + time * (data.shape[0] - ntimesteps - 1) - 1] = _y_2

    return dataX, dataY_1, dataY_2


def create_offline_convlstm_data_fix_ratio(data, input_shape, mon_ratio, eps, data_time=None):
    if data_time is None:
        data_time = Config.CONV_LSTM_DATA_GENERATE_TIME

    _tf = np.array([1.0, 0.0])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]
    dataX = np.zeros(
        ((data.shape[0] - ntimesteps) * data_time, ntimesteps, wide, high, channel))
    dataY = np.zeros(((data.shape[0] - ntimesteps) * data_time, wide * high))

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

            dataX[idx + time * (data.shape[0] - ntimesteps)] = _x

            _y = data[idx + ntimesteps]
            _y = np.reshape(_y, newshape=(wide * high))

            dataY[idx + time * (data.shape[0] - ntimesteps)] = _y

    return dataX, dataY


def create_offline_cnnlstm_data_fix_ratio(data, input_shape, mon_ratio, eps, data_time=None):
    if data_time is None:
        data_time = Config.CONV_LSTM_DATA_GENERATE_TIME

    _tf = np.array([1.0, 0.0])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]
    dataX = np.zeros(
        ((data.shape[0] - ntimesteps) * data_time, ntimesteps, wide, high, channel))
    dataY = np.zeros(((data.shape[0] - ntimesteps) * data_time, ntimesteps, wide * high))

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

            dataX[idx + time * (data.shape[0] - ntimesteps)] = _x

            _y = data[(idx + 1):(idx + ntimesteps + 1)]
            _y = np.reshape(_y, newshape=(ntimesteps, wide * high))

            dataY[idx + time * (data.shape[0] - ntimesteps)] = _y

    return dataX, dataY


def create_offline_fwbw_convlstm_data(data, input_shape, mon_ratio, eps, data_time=None):
    if data_time is None:
        data_time = Config.CONV_LSTM_DATA_GENERATE_TIME

    _tf = np.array([1.0, 0.0])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]
    dataX = np.zeros(
        ((data.shape[0] - ntimesteps) * data_time, ntimesteps, wide, high, channel))
    dataY_1 = np.zeros(((data.shape[0] - ntimesteps) * data_time, wide * high))
    dataY_2 = np.zeros(((data.shape[0] - ntimesteps) * data_time, ntimesteps - 2, wide * high))

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

            dataX[idx + time * (data.shape[0] - ntimesteps)] = _x

            _y = data[(idx + 1):(idx + ntimesteps - 1)]
            _y = np.reshape(_y, newshape=(ntimesteps - 2, wide * high))

            dataY_1[idx + time * (data.shape[0] - ntimesteps)] = data[idx + ntimesteps].flatten()
            dataY_2[idx + time * (data.shape[0] - ntimesteps)] = _y

    return dataX, dataY_1, dataY_2


def create_offline_lstm_nn_data(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                                       size=data.shape,
                                       p=(mon_ratio, 1 - mon_ratio))
    dataX = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, features))
    dataY = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, 1))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            dataX[i, :, 0] = _x
            dataX[i, :, 1] = _label

            _y = data[(idx + 1):(idx + ntimesteps + 1), flow]

            dataY[i] = np.array(_y).reshape((ntimesteps, 1))

            i += 1

    return dataX, dataY


def create_offline_fwbw_lstm_2(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    dataX = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, features))
    dataY_1 = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, 1))
    dataY_2 = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps - 2))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            dataX[i, :, 0] = _x
            dataX[i, :, 1] = _label

            _y = data[(idx + 1):(idx + ntimesteps + 1), flow]

            dataY_1[i] = np.reshape(_y, newshape=(ntimesteps, 1))
            dataY_2[i] = data[(idx + 1):(idx + ntimesteps - 1), flow]
            i += 1

    return dataX, dataY_1, dataY_2


def create_offline_fwbw_lstm(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    dataX = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps, features))
    dataY_1 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps))
    dataY_2 = np.zeros(((data.shape[0] - ntimesteps - 1) * data.shape[1], ntimesteps))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(1, _data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            dataX[i, :, 0] = _x
            dataX[i, :, 1] = _label

            _y_1 = data[(idx + 1):(idx + ntimesteps + 1), flow]
            _y_2 = data[(idx - 1):(idx + ntimesteps - 1), flow]

            dataY_1[i] = _y_1
            dataY_2[i] = _y_2
            i += 1

    return dataX, dataY_1, dataY_2


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
    dataX = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], features))
    dataY = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1]))

    _data = np.copy(data)

    _data[_labels == 0] = np.random.uniform(_data[_labels == 0] - eps, _data[_labels == 0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - ntimesteps - 1):
            _x = _data[idx: (idx + ntimesteps), flow]
            _x = pd.Series(_x)
            dataX[i, :] = np.array(create_xgb_features(_x))

            _y = _data[(idx + ntimesteps + 1), flow]

            dataY[i] = _y

            i += 1

    print("[PROC_ID: %d] Sending result" % proc_id)
    connection.send([dataX, dataY])
    connection.close()
    print("[PROC_ID] RESULT SENT")


def parallel_create_offline_xgb_data(data, ntimesteps, features, mon_ratio, eps):
    nproc = cpu_count()
    quota = int(data.shape[1] / nproc)

    p = [0] * nproc

    connections = []

    for proc_id in range(nproc):
        connections.append(Pipe())

    dataX = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], features))

    dataY = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1]))
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

        dataX[_start: _end] = ret_xy[proc_id][0]
        dataY[_start: _end] = ret_xy[proc_id][1]

    return dataX, dataY


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


def data_scalling(train_data2d, valid_data2d, test_data2d):
    if Config.SCALER == Config.SCALERS[0]:  # Power transform
        pt = PowerTransformer(copy=True, standardize=True, method='yeo-johnson')
        pt.fit(train_data2d)
        train_data_normalized2d = pt.transform(train_data2d)
        valid_data_normalized2d = pt.transform(valid_data2d)
        test_data_normalized2d = pt.transform(test_data2d)
        scalers = pt
    elif Config.SCALER == Config.SCALERS[1]:  # Standard Scaler
        ss = StandardScaler(copy=True)
        ss.fit(train_data2d)
        train_data_normalized2d = ss.transform(train_data2d)
        valid_data_normalized2d = ss.transform(valid_data2d)
        test_data_normalized2d = ss.transform(test_data2d)
        scalers = ss
    elif Config.SCALER == Config.SCALERS[2]:
        mm = MinMaxScaler(copy=True)
        mm.fit(train_data2d)
        train_data_normalized2d = mm.transform(train_data2d)
        valid_data_normalized2d = mm.transform(valid_data2d)
        test_data_normalized2d = mm.transform(test_data2d)
        scalers = mm
    elif Config.SCALER == Config.SCALERS[3]:
        bc = PowerTransformer(copy=True, standardize=True, method='box-cox')
        bc.fit(train_data2d)
        train_data_normalized2d = bc.transform(train_data2d)
        valid_data_normalized2d = bc.transform(valid_data2d)
        test_data_normalized2d = bc.transform(test_data2d)
        scalers = bc
    elif Config.SCALER == Config.SCALERS[4]:
        rb = RobustScaler()
        rb.fit(train_data2d)
        train_data_normalized2d = rb.transform(train_data2d)
        valid_data_normalized2d = rb.transform(valid_data2d)
        test_data_normalized2d = rb.transform(test_data2d)
        scalers = rb
    elif Config.SCALER == Config.SCALERS[5]:
        sd = sd_scale()
        sd.fit(train_data2d)
        train_data_normalized2d = sd.transform(train_data2d)
        valid_data_normalized2d = sd.transform(valid_data2d)
        test_data_normalized2d = sd.transform(test_data2d)
        scalers = sd
    else:
        raise Exception('Unknown scaler!')

    return train_data_normalized2d, valid_data_normalized2d, test_data_normalized2d, scalers
