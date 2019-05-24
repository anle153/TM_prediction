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


def generator_convlstm_train_data(data, input_shape, mon_ratio, eps, batch_size):
    _tf = np.array([True, False])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]

    measured_matrix = np.zeros(shape=data.shape)

    sampling_ratioes = np.random.uniform(0.1, 0.4, size=data.shape[0])
    for i in range(data.shape[0]):
        measured_row = np.random.choice(_tf,
                                        size=(data.shape[1], data.shape[2]),
                                        p=(sampling_ratioes[i], 1 - sampling_ratioes[i]))

        measured_matrix[i, :, :] = measured_row

    _labels = measured_matrix.astype(int)
    _data = np.copy(data)

    _data[_labels == 0] = np.random.uniform(_data[_labels == 0] - eps, _data[_labels == 0] + eps)

    _data = np.expand_dims(_data, axis=3)
    _labels = np.expand_dims(_labels, axis=3)

    _data = np.concatenate([_data, _labels], axis=3)

    dataX = np.zeros((batch_size, ntimesteps, wide, high, channel))
    dataY = np.zeros((batch_size, ntimesteps, wide, high, 1))

    while True:

        indices = np.random.randint(0, _data.shape[0] - ntimesteps - 1, size=batch_size)
        for i in range(batch_size):
            idx = indices[i]

            _x = _data[idx: (idx + ntimesteps), :, :, :]

            dataX[i, :, :, :, :] = _x

            _y = _data[(idx + 1):(idx + ntimesteps + 1), :, :, 0]
            _y = np.expand_dims(_y, axis=3)

            dataY[i, :, :, :, :] = _y

        yield dataX, dataY


def generator_convlstm_train_data_fix_ratio(data, input_shape, mon_ratio, eps, batch_size):
    _tf = np.array([True, False])

    ntimesteps = input_shape[0]
    wide = input_shape[1]
    high = input_shape[2]
    channel = input_shape[3]

    measured_matrix = np.random.choice(_tf,
                                       size=data.shape,
                                       p=(mon_ratio, 1 - mon_ratio))
    _labels = measured_matrix.astype(int)
    _data = np.copy(data)

    _data[_labels == 0] = np.random.uniform(_data[_labels == 0] - eps, _data[_labels == 0] + eps)

    _data = np.expand_dims(_data, axis=3)
    _labels = np.expand_dims(_labels, axis=3)

    _data = np.concatenate([_data, _labels], axis=3)

    dataX = np.zeros((batch_size, ntimesteps, wide, high, channel))
    dataY = np.zeros((batch_size, ntimesteps, wide, high, 1))

    while True:

        indices = np.random.randint(0, _data.shape[0] - ntimesteps - 1, size=batch_size)
        for i in range(batch_size):
            idx = indices[i]

            _x = _data[idx: (idx + ntimesteps), :, :, :]

            dataX[i] = _x

            _y = _data[(idx + 1):(idx + ntimesteps + 1), :, :, 0]
            _y = np.expand_dims(_y, axis=3)

            dataY[i] = _y

        yield dataX, dataY


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


def generator_lstm_nn_train_data(data, input_shape, mon_ratio, eps, batch_size):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))

    dataX = np.zeros((batch_size, ntimesteps, features))
    dataY = np.zeros((batch_size, ntimesteps, 1))

    random_data = np.copy(data)

    random_data[_labels == 0] = np.random.uniform(random_data[_labels == 0] - eps, random_data[_labels == 0] + eps)

    while True:
        flows = np.random.randint(0, random_data.shape[1], size=batch_size)
        indices = np.random.randint(0, random_data.shape[0] - ntimesteps - 1, size=batch_size)

        for i in range(batch_size):
            flow = flows[i]
            idx = indices[i]

            _x = random_data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            dataX[i, :, 0] = _x
            dataX[i, :, 1] = _label

            _y = data[(idx + 1):(idx + ntimesteps + 1), flow]

            dataY[i, :, :] = np.array(_y).reshape((ntimesteps, 1))

        yield dataX, dataY


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


def create_offline_fwbw_lstm_data(data, input_shape, mon_ratio, eps):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    dataX = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps, features))
    dataY_2 = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], ntimesteps - 2))
    dataY_1 = np.zeros(((data.shape[0] - ntimesteps) * data.shape[1], 1))

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - ntimesteps):
            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            dataX[i, :, 0] = _x
            dataX[i, :, 1] = _label

            _y = data[(idx + 1):(idx + ntimesteps - 1), flow]

            dataY_2[i] = _y
            dataY_1[i] = data[idx + ntimesteps, flow]
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
