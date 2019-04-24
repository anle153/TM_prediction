from multiprocessing import Process, Pipe

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from FlowClassification.SpatialClustering import *


def prepare_train_test_3d(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days * 0.8 * day_size)

    train_set = data[0:train_size, :, :]
    test_set = data[train_size:, :, :]

    return train_set, test_set


def prepare_train_valid_test_3d(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days * 0.6 * day_size)

    valid_size = int(n_days * 0.2 * day_size)

    train_set = data[0:train_size, :, :]
    valid_set = data[train_size:train_size + valid_size, :, :]
    test_set = data[train_size + valid_size:, :, :]

    return train_set, valid_set, test_set


def prepare_train_test_2d(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days * 0.8 * day_size)

    train_set = data[0:train_size, :]
    test_set = data[train_size:, :]

    return train_set, test_set


def prepare_train_valid_test_2d(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days * 0.6 * day_size)

    valid_size = int(n_days * 0.2 * day_size)

    train_set = data[0:train_size, :]
    valid_set = data[train_size:(train_size + valid_size), :]
    test_set = data[(train_size + valid_size):, :]

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


def generator_lstm_nn_train_data(data, input_shape, mon_ratio, eps, batch_size):
    ntimesteps = input_shape[0]
    features = input_shape[1]

    _tf = np.array([True, False])
    measured_matrix = np.random.choice(_tf,
                                       size=data.shape,
                                       p=(mon_ratio, 1 - mon_ratio))
    _labels = measured_matrix.astype(int)
    dataX = np.zeros((batch_size, ntimesteps, features))
    dataY = np.zeros((batch_size, ntimesteps, 1))

    _data = np.copy(data)

    _data[_labels == 0] = np.random.uniform(_data[_labels == 0] - eps, _data[_labels == 0] + eps)

    while True:
        flows = np.random.randint(0, _data.shape[1], size=batch_size)
        indices = np.random.randint(0, _data.shape[0] - ntimesteps - 1, size=batch_size)

        for i in range(batch_size):
            flow = flows[i]
            idx = indices[i]

            _x = _data[idx: (idx + ntimesteps), flow]
            _label = _labels[idx: (idx + ntimesteps), flow]

            dataX[i, :, 0] = _x
            dataX[i, :, 1] = _label

            _y = _data[(idx + 1):(idx + ntimesteps + 1), flow]

            dataY[i, :, :] = np.array(_y).reshape((ntimesteps, 1))

        yield dataX, dataY
