import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from saxpy.paa import paa
from saxpy.znorm import znorm
from scipy.sparse import linalg
from tqdm import tqdm


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """

        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class DataLoader_dcrnn_fwbw(object):
    def __init__(self, inputs, dec_labels_fw, enc_labels_bw, batch_size,
                 pad_with_last_sample=True, shuffle=False):
        """

        :param inputs:
        :param enc_labels_fw:
        :param dec_labels_fw:
        :param enc_labels_bw:
        :param dec_labels_bw:
        :param batch_size:
        :param pad_with_last_sample:
        :param shuffle:
        """

        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(inputs) % batch_size)) % batch_size
            inputs_padding = np.repeat(inputs[-1:], num_padding, axis=0)
            dec_labels_fw_padding = np.repeat(dec_labels_fw[-1:], num_padding, axis=0)
            enc_labels_bw_padding = np.repeat(enc_labels_bw[-1:], num_padding, axis=0)

            inputs = np.concatenate([inputs, inputs_padding], axis=0)
            dec_labels_fw = np.concatenate([dec_labels_fw, dec_labels_fw_padding], axis=0)
            enc_labels_bw = np.concatenate([enc_labels_bw, enc_labels_bw_padding], axis=0)

        self.size = len(inputs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            inputs, dec_labels_fw, enc_labels_bw = inputs[permutation], dec_labels_fw[permutation], \
                                                   enc_labels_bw[permutation]
        self.inputs = inputs
        self.dec_labels_fw = dec_labels_fw
        self.enc_labels_bw = enc_labels_bw

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                inputs_i = self.inputs[start_ind: end_ind, ...]
                dec_labels_fw_i = self.dec_labels_fw[start_ind: end_ind, ...]
                enc_labels_bw_i = self.enc_labels_bw[start_ind: end_ind, ...]
                yield (inputs_i, dec_labels_fw_i, enc_labels_bw_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.product([x.value for x in variable.get_shape()])
    return total_parameters


def prepare_train_valid_test_2d(data, day_size):
    n_timeslots = data.shape[0]
    n_days = n_timeslots / day_size

    train_size = int(n_days * 0.6)

    valid_size = int(n_days * 0.2)

    train_set = data[0:train_size * day_size, :]
    valid_set = data[train_size * day_size:(train_size * day_size + valid_size * day_size), :]
    test_set = data[(train_size * day_size + valid_size * day_size):, :]

    return train_set, valid_set, test_set


def create_data_dcrnn(data, seq_len, horizon, input_dim, mon_ratio, eps):
    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf, size=data.shape, p=(mon_ratio, 1.0 - mon_ratio))
    _labels = _labels.astype('float32')
    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    x = np.zeros(shape=(data.shape[0] - seq_len - horizon, seq_len, data.shape[1], input_dim), dtype='float32')
    y = np.zeros(shape=(data.shape[0] - seq_len - horizon, horizon, data.shape[1], 1), dtype='float32')

    for idx in range(_data.shape[0] - seq_len - horizon):
        _x = _data[idx: idx + seq_len]
        _label = _labels[idx: idx + seq_len]

        x[idx, :, :, 0] = _x
        x[idx, :, :, 1] = _label

        _y = data[idx + seq_len:idx + seq_len + horizon]

        y[idx] = np.expand_dims(_y, axis=2)

    return x, y


def create_data_dcrnn_weighted(data, seq_len, horizon, input_dim, mon_ratio, eps):
    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf, size=data.shape, p=(mon_ratio, 1.0 - mon_ratio))
    _labels = _labels.astype('float32')
    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    x = np.zeros(shape=(_data.shape[0] - seq_len - seq_len - horizon, seq_len, _data.shape[1], input_dim),
                 dtype='float32')
    y = np.zeros(shape=(_data.shape[0] - seq_len - seq_len - horizon, horizon, _data.shape[1], 1), dtype='float32')

    for idx in range(_data.shape[0] - seq_len - seq_len - horizon):
        _x = _data[idx + seq_len: idx + seq_len + seq_len]
        _label = _labels[idx + seq_len: idx + seq_len + seq_len]

        _w = np.zeros(shape=(seq_len, _data.shape[1], 1))
        for i in range(seq_len):
            _mr = _labels[(idx + i):(idx + seq_len + i)].sum(axis=0) / seq_len
            _w[i] = np.expand_dims(_mr, axis=1)

        # x[idx] = np.stack([_x, _label, _w], axis=2)

        _x = np.expand_dims(_x, axis=2)
        _label = np.expand_dims(_label, axis=2)
        x[idx] = np.concatenate([_x, _label, _w], axis=2)

        _y = data[idx + seq_len + seq_len:idx + seq_len + seq_len + horizon]

        y[idx] = np.expand_dims(_y, axis=2)

    return x, y


def create_data_dcrnn_fwbw(data, seq_len, horizon, input_dim, mon_ratio, eps):
    """

    :param data:
    :param seq_len:
    :param horizon:
    :param input_dim:
    :param mon_ratio:
    :param eps:
    :return:
    """
    _tf = np.array([1.0, 0.0])
    _m_indicators = np.random.choice(_tf, size=data.shape, p=(mon_ratio, 1.0 - mon_ratio))
    _m_indicators = _m_indicators.astype('float32')
    _data = np.copy(data)

    _data[_m_indicators == 0.0] = np.random.uniform(_data[_m_indicators == 0.0] - eps,
                                                    _data[_m_indicators == 0.0] + eps)

    inputs = np.zeros(shape=(data.shape[0] - seq_len - horizon, seq_len, data.shape[1], input_dim), dtype='float32')
    dec_labels_fw = np.zeros(shape=(data.shape[0] - seq_len - horizon, horizon, data.shape[1], 1), dtype='float32')
    enc_labels_bw = np.zeros(shape=(data.shape[0] - seq_len - horizon, seq_len, data.shape[1], 1), dtype='float32')

    for idx in range(horizon, _data.shape[0] - seq_len - horizon, 1):
        _input = _data[idx: idx + seq_len]
        _m = _m_indicators[idx: idx + seq_len]

        inputs[idx, :, :, 0] = _input
        inputs[idx, :, :, 1] = _m

        _dec_labels_fw = data[idx + seq_len:idx + seq_len + horizon]
        dec_labels_fw[idx] = np.expand_dims(_dec_labels_fw, axis=2)

        _enc_labels_bw = data[idx - 1:idx + seq_len - 1]
        enc_labels_bw[idx] = np.expand_dims(_enc_labels_bw, axis=2)

    # return inputs, dec_labels_fw, enc_labels_bw
    return inputs, dec_labels_fw, enc_labels_bw


def correlation_matrix(data, seq_len):
    corr_matrices = np.zeros(shape=(data.shape[0] - seq_len, data.shape[1], data.shape[1]), dtype='float32')

    for i in tqdm(range(data.shape[0] - seq_len)):
        data_corr = data[i:i + seq_len]
        df = pd.DataFrame(data_corr, index=range(data_corr.shape[0]),
                          columns=['{}'.format(x + 1) for x in range(data_corr.shape[1])])

        corr_mx = df.corr().values
        corr_matrices[i] = corr_mx

    nan_idx = []
    for i in range(corr_matrices.shape[0]):
        if not np.any(np.isnan(corr_matrices[i])) and not np.any(np.isinf(corr_matrices[i])):
            nan_idx.append(i)

    corr_matrices = corr_matrices[nan_idx]
    corr_matrix = np.mean(corr_matrices, axis=0)

    return corr_matrix


def od_flow_matrix(flow_index_file='./Dataset/demands.csv'):
    flow_index = pd.read_csv(flow_index_file)
    nflow = flow_index['index'].size
    adj_matrix = np.zeros(shape=(nflow, nflow))

    for i in range(nflow):
        for j in range(nflow):
            if flow_index.iloc[i].d == flow_index.iloc[j].d:
                adj_matrix[i, j] = 1.0

    return adj_matrix


def sd_flow_matrix(flow_index_file='./Dataset/demands.csv'):
    flow_index = pd.read_csv(flow_index_file)
    nflow = flow_index['index'].size
    adj_matrix = np.zeros(shape=(nflow, nflow))

    for i in range(nflow):
        for j in range(nflow):
            if (flow_index.iloc[i].d == flow_index.iloc[j].d) or (flow_index.iloc[i].o == flow_index.iloc[j].o):
                adj_matrix[i, j] = 1.0

    return adj_matrix


def ppa_representation(data, seq_len):
    data_reduced = np.zeros(shape=(int(data.shape[0] / seq_len), data.shape[1]))

    paa_segment = int(data.shape[0] / seq_len)

    for i in tqdm(range(data.shape[1])):
        dat_znorm = znorm(data[:, i])

        data_reduced[:, i] = paa(dat_znorm, paa_segment)

    return data_reduced


def sax_similarity(data, seq_len):
    from tslearn.piecewise import SymbolicAggregateApproximation

    print('|--- Calculating the pairwise distance!')

    ppa_segmet = int(data.shape[0] / seq_len)
    sax_ins = SymbolicAggregateApproximation(n_segments=ppa_segmet, alphabet_size_avg=10)

    sax_repre = sax_ins.fit_transform(np.transpose(data))

    sax_mx_dist = np.zeros(shape=(data.shape[1], data.shape[1]))

    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            sax_mx_dist[i, j] = sax_ins.distance_sax(sax_repre[i], sax_repre[j])

    return sax_mx_dist


def dynamic_time_wrap_PPA(data, seq_len):
    from tslearn.metrics import cdist_dtw
    print('|--- Construct adj_mx by DTW_PPA')

    ppa_re = ppa_representation(data, seq_len)  # (time, -1)
    dtw_mx_dist = cdist_dtw(ppa_re.transpose())
    return dtw_mx_dist


def dynamic_time_wrap(data):
    from tslearn.metrics import cdist_dtw
    print('|--- Construct adj_mx by DTW')
    dtw_mx_dist = cdist_dtw(data.transpose())
    return dtw_mx_dist


def euclidean_PPA(data, seq_len):
    pass


def knn_ts(data, metric='dtw'):
    pass


#                0        1       2      3        4        5        6      7      8      9
ADJ_METHOD = ['CORR1', 'CORR2', 'OD', 'EU_PPA', 'DTW', 'DTW_PPA', 'SAX', 'KNN', 'SD', 'CORR3']


def adj_mx_contruction(adj_method, data, seq_len, adj_dir, pos_thres=0.7, neg_thres=-0.8):
    adj_file_name = '{}-{}'.format(adj_method, pos_thres)
    if adj_method == ADJ_METHOD[1]:
        adj_file_name = adj_file_name + '-{}'.format(neg_thres)

    if os.path.isfile(os.path.join(adj_dir, adj_file_name + '.npy')):
        adj_mx = np.load(os.path.join(adj_dir, adj_file_name + '.npy'))
        return adj_mx

    if adj_method == ADJ_METHOD[0]:
        # Construct graph by using avg correlation (positive)
        adj_mx = correlation_matrix(data, seq_len)
        # adj_mx = (adj_mx - adj_mx.min()) / (adj_mx.max() - adj_mx.min())
        adj_mx[adj_mx < pos_thres] = 0.0
    elif adj_method == ADJ_METHOD[1]:
        # Construct graph by using avg correlation (positive and negative)
        adj_mx = correlation_matrix(data, seq_len)
        # adj_mx = (adj_mx - adj_mx.min()) / (adj_mx.max() - adj_mx.min())
        adj_mx[(pos_thres > adj_mx) * (adj_mx > neg_thres)] = 0.0
    elif adj_method == ADJ_METHOD[2]:
        # Construct graph by destination information
        adj_mx = od_flow_matrix()
    elif adj_method == ADJ_METHOD[3]:
        raise NotImplementedError('Need to be implemented!')
    elif adj_method == ADJ_METHOD[4]:
        # Caculating the pairwise distance of DTW on raw
        dtw_mx_dist = dynamic_time_wrap(data)
        adj_mx = dtw_mx_dist.max() - dtw_mx_dist
        adj_mx = (adj_mx - adj_mx.min()) / (adj_mx.max() - adj_mx.min())
        adj_mx[adj_mx < pos_thres] = 0.0
    elif adj_method == ADJ_METHOD[5]:
        # Caculating the pairwise distance of DTW on PPA
        dtw_ppa_mx_dist = dynamic_time_wrap_PPA(data, seq_len)
        adj_mx = dtw_ppa_mx_dist.max() - dtw_ppa_mx_dist
        adj_mx = (adj_mx - adj_mx.min()) / (adj_mx.max() - adj_mx.min())
        adj_mx[adj_mx < pos_thres] = 0.0
    elif adj_method == ADJ_METHOD[6]:
        # Caculating the pairwise distance of sax representation
        sax_mx_dist = sax_similarity(data, seq_len)
        adj_mx = sax_mx_dist.max() - sax_mx_dist
        adj_mx = (adj_mx - adj_mx.min()) / (adj_mx.max() - adj_mx.min())
        adj_mx[adj_mx < pos_thres] = 0.0
    elif adj_method == ADJ_METHOD[7]:
        raise NotImplementedError('Need to be implemented!')
    elif adj_method == ADJ_METHOD[8]:
        adj_mx = sd_flow_matrix()
    elif adj_method == ADJ_METHOD[9]:
        # Construct graph by using avg correlation (positive/ no weights)
        adj_mx = correlation_matrix(data, seq_len)
        # adj_mx = (adj_mx - adj_mx.min()) / (adj_mx.max() - adj_mx.min())
        adj_mx[adj_mx < pos_thres] = 0.0
        adj_mx[adj_mx >= pos_thres] = 1.0
    else:
        raise ValueError('Adj constructor is not implemented!')

    np.save(os.path.join(adj_dir, adj_file_name), adj_mx)

    return adj_mx


def load_dataset_dcrnn(seq_len, horizon, input_dim, mon_ratio,
                       dataset_dir, data_size, day_size, batch_size, eval_batch_size,
                       pos_thres, neg_thres, val_batch_size, adj_method='CORR1', **kwargs):
    raw_data = np.load(dataset_dir + 'Abilene2d.npy')
    raw_data[raw_data <= 0] = 0.1

    # Convert traffic volume from byte to mega-byte
    # raw_data = raw_data / 1000000

    raw_data = raw_data.astype("float32")

    raw_data = raw_data[:int(raw_data.shape[0] * data_size)]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, day_size=day_size)
    test_data2d = test_data2d[0:-day_size * 3]
    data = {}

    print('|--- Normalizing the train set.')
    scaler = StandardScaler(mean=train_data2d.mean(), std=train_data2d.std())
    train_data_norm = scaler.transform(train_data2d)
    valid_data_norm = scaler.transform(valid_data2d)
    test_data_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data_norm

    x_train, y_train = create_data_dcrnn(data=train_data_norm, seq_len=seq_len, horizon=horizon,
                                         input_dim=input_dim,
                                         mon_ratio=mon_ratio, eps=train_data_norm.std())
    x_val, y_val = create_data_dcrnn(data=valid_data_norm, seq_len=seq_len, horizon=horizon, input_dim=input_dim,
                                     mon_ratio=mon_ratio, eps=train_data_norm.std())
    x_eval, y_eval = create_data_dcrnn(data=test_data_norm, seq_len=seq_len, horizon=horizon, input_dim=input_dim,
                                       mon_ratio=mon_ratio, eps=train_data_norm.std())

    for category in ['train', 'val', 'eval']:
        _x, _y = locals()["x_" + category], locals()["y_" + category]
        print(category, "x: ", _x.shape, "y:", _y.shape)
        data['x_' + category] = _x
        data['y_' + category] = _y
    # Data format

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], val_batch_size, shuffle=False)
    data['eval_loader'] = DataLoader(data['x_eval'], data['y_eval'], eval_batch_size, shuffle=False)
    data['scaler'] = scaler

    print('|--- Get Correlation Matrix')

    adj_mx = adj_mx_contruction(adj_method=adj_method, data=train_data2d, seq_len=seq_len, adj_dir=dataset_dir,
                                pos_thres=pos_thres, neg_thres=neg_thres)

    print('Number of edges: {}'.format(np.sum(adj_mx > 0.0)))

    adj_mx = adj_mx.astype('float32')

    data['adj_mx'] = adj_mx

    return data


def load_dataset_dcrnn_weighted(seq_len, horizon, input_dim, mon_ratio,
                                dataset_dir, data_size, day_size, batch_size, eval_batch_size,
                                pos_thres, neg_thres, val_batch_size, adj_method='CORR1', **kwargs):
    raw_data = np.load(dataset_dir + 'Abilene2d.npy')
    raw_data[raw_data <= 0] = 0.1

    # Convert traffic volume from byte to mega-byte
    # raw_data = raw_data / 1000000

    raw_data = raw_data.astype("float32")

    raw_data = raw_data[:int(raw_data.shape[0] * data_size)]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, day_size=day_size)
    test_data2d = test_data2d[0:-day_size * 3]
    data = {}

    print('|--- Normalizing the train set.')
    scaler = StandardScaler(mean=train_data2d.mean(), std=train_data2d.std())
    train_data_norm = scaler.transform(train_data2d)
    valid_data_norm = scaler.transform(valid_data2d)
    test_data_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data_norm

    x_train, y_train = create_data_dcrnn_weighted(data=train_data_norm, seq_len=seq_len, horizon=horizon,
                                                  input_dim=input_dim,
                                                  mon_ratio=mon_ratio, eps=train_data_norm.std())
    x_val, y_val = create_data_dcrnn_weighted(data=valid_data_norm, seq_len=seq_len, horizon=horizon,
                                              input_dim=input_dim,
                                              mon_ratio=mon_ratio, eps=train_data_norm.std())
    x_eval, y_eval = create_data_dcrnn_weighted(data=test_data_norm, seq_len=seq_len, horizon=horizon,
                                                input_dim=input_dim,
                                                mon_ratio=mon_ratio, eps=train_data_norm.std())

    for category in ['train', 'val', 'eval']:
        _x, _y = locals()["x_" + category], locals()["y_" + category]
        print(category, "x: ", _x.shape, "y:", _y.shape)
        data['x_' + category] = _x
        data['y_' + category] = _y
    # Data format

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], val_batch_size, shuffle=False)
    data['eval_loader'] = DataLoader(data['x_eval'], data['y_eval'], eval_batch_size, shuffle=False)
    data['scaler'] = scaler

    print('|--- Get Correlation Matrix')

    adj_mx = adj_mx_contruction(adj_method=adj_method, data=train_data2d, seq_len=seq_len, adj_dir=dataset_dir,
                                pos_thres=pos_thres, neg_thres=neg_thres)

    print('Number of edges: {}'.format(np.sum(adj_mx)))

    adj_mx = adj_mx.astype('float32')

    data['adj_mx'] = adj_mx

    return data


def load_dataset_dcrnn_fwbw(seq_len, horizon, input_dim, mon_ratio,
                            dataset_dir, data_size, day_size, batch_size, eval_batch_size,
                            pos_thres, neg_thres, val_batch_size, adj_method='CORR1', **kwargs):

    raw_data = np.load(dataset_dir + 'Abilene2d.npy')
    raw_data[raw_data <= 0] = 0.1

    raw_data = raw_data.astype("float32")

    raw_data = raw_data[:int(raw_data.shape[0] * data_size)]

    # Remove last 3 days
    if data_size == 1.0:
        raw_data = raw_data[:-day_size * 3]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, day_size=day_size)
    data = {}

    print('|--- Normalizing the train set.')
    scaler = StandardScaler(mean=train_data2d.mean(), std=train_data2d.std())
    train_data_norm = scaler.transform(train_data2d)
    valid_data_norm = scaler.transform(valid_data2d)
    test_data_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data_norm

    # x(num_sample, seq_len, num_node, input_dim): encoder input
    # y(num_sample, horizon, num_node, output_dim): decoder output
    # l(num_sample, seq_len, num_node, output_dim): encoder output
    inputs_train, dec_labels_fw_train, enc_labels_bw_train = create_data_dcrnn_fwbw(
        data=train_data_norm, seq_len=seq_len, horizon=horizon,
        input_dim=input_dim,
        mon_ratio=mon_ratio, eps=train_data_norm.std())
    inputs_val, dec_labels_fw_val, enc_labels_bw_val = create_data_dcrnn_fwbw(
        data=valid_data_norm,
                                                                              seq_len=seq_len, horizon=horizon,
                                                                              input_dim=input_dim,
                                                                              mon_ratio=mon_ratio, eps=train_data_norm.std())
    inputs_eval, dec_labels_fw_eval, enc_labels_bw_eval = create_data_dcrnn_fwbw(
        data=test_data_norm, seq_len=seq_len, horizon=horizon,
        input_dim=input_dim,
        mon_ratio=mon_ratio, eps=train_data_norm.std())

    for category in ['train', 'val', 'eval']:
        _inputs, _dec_labels_fw, _enc_labels_bw = locals()["inputs_" + category], locals()["dec_labels_fw_" + category], \
                                                  locals()["enc_labels_bw_" + category]

        print(category, "inputs_: ", _inputs.shape, "dec_labels_fw_", _dec_labels_fw.shape, "enc_labels_bw_:",
              _enc_labels_bw.shape)
        data['inputs_' + category] = _inputs
        data['dec_labels_fw_' + category] = _dec_labels_fw
        data['enc_labels_bw_' + category] = _enc_labels_bw

    # Data format
    data['train_loader'] = DataLoader_dcrnn_fwbw(data['inputs_train'],
                                                 data['dec_labels_fw_train'],
                                                 data['enc_labels_bw_train'],
                                                 batch_size, shuffle=True)
    data['val_loader'] = DataLoader_dcrnn_fwbw(data['inputs_val'],
                                               data['dec_labels_fw_val'],
                                               data['enc_labels_bw_val'],
                                               val_batch_size, shuffle=True)
    data['eval_loader'] = DataLoader_dcrnn_fwbw(data['inputs_eval'],
                                                data['dec_labels_fw_eval'],
                                                data['enc_labels_bw_eval'],
                                                eval_batch_size, shuffle=True)

    data['scaler'] = scaler

    print('|--- Get Correlation Matrix')

    adj_mx = adj_mx_contruction(adj_method=adj_method, data=train_data2d, seq_len=seq_len, adj_dir=dataset_dir,
                                pos_thres=pos_thres, neg_thres=neg_thres)

    print('Number of edges: {}'.format(np.sum(adj_mx)))

    adj_mx = adj_mx.astype('float32')

    data['adj_mx'] = adj_mx

    return data


def create_data_fwbw_lstm(data, seq_len, input_dim, mon_ratio, eps):
    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf,
                               size=data.shape,
                               p=(mon_ratio, 1 - mon_ratio))
    data_x = np.zeros(shape=((data.shape[0] - seq_len - 1) * data.shape[1], seq_len, input_dim), dtype='float32')
    data_y_1 = np.zeros(shape=((data.shape[0] - seq_len - 1) * data.shape[1], seq_len, 1), dtype='float32')
    data_y_2 = np.zeros(shape=((data.shape[0] - seq_len - 1) * data.shape[1], seq_len), dtype='float32')

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(1, _data.shape[0] - seq_len):
            _x = _data[idx: (idx + seq_len), flow]
            _label = _labels[idx: (idx + seq_len), flow]

            data_x[i, :, 0] = _x
            data_x[i, :, 1] = _label

            _y_1 = data[(idx + 1):(idx + seq_len + 1), flow]
            _y_2 = data[(idx - 1):(idx + seq_len - 1), flow]

            data_y_1[i] = np.reshape(_y_1, newshape=(seq_len, 1))
            data_y_2[i] = _y_2
            i += 1

    return data_x, data_y_1, data_y_2


def load_dataset_fwbw_lstm(seq_len, horizon, input_dim, mon_ratio,
                           raw_dataset_dir, day_size, data_size,
                           batch_size, eval_batch_size=None, **kwargs):
    data = {}

    raw_data = np.load(raw_dataset_dir)
    raw_data[raw_data <= 0] = 0.1

    raw_data = raw_data.astype("float32")

    raw_data = raw_data[:int(raw_data.shape[0] * data_size)]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, day_size=day_size)
    test_data2d = test_data2d[0:-day_size * 3]

    print('|--- Normalizing the train set.')
    scaler = StandardScaler(mean=train_data2d.mean(), std=train_data2d.std())
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data2d_norm

    x_train, y_train_1, y_train_2 = create_data_fwbw_lstm(train_data2d_norm, seq_len=seq_len, input_dim=input_dim,
                                                          mon_ratio=mon_ratio, eps=train_data2d_norm.std())
    x_val, y_val_1, y_val_2 = create_data_fwbw_lstm(valid_data2d_norm, seq_len=seq_len, input_dim=input_dim,
                                                    mon_ratio=mon_ratio, eps=train_data2d_norm.std())
    x_eval, y_eval_1, y_eval_2 = create_data_fwbw_lstm(test_data2d_norm, seq_len=seq_len, input_dim=input_dim,
                                                       mon_ratio=mon_ratio, eps=train_data2d_norm.std())

    for cat in ["train", "val", "eval"]:
        _x, _y_1, _y_2 = locals()["x_" + cat], locals()["y_" + cat + '_1'], locals()["y_" + cat + '_2']
        print(cat, "x: ", _x.shape, "y_1:", _y_1.shape, "y_2:", _y_2.shape)

        data['x_' + cat] = _x
        data['y_' + cat + '_1'] = _y_1
        data['y_' + cat + '_2'] = _y_2

    data['scaler'] = scaler

    return data


def create_data_lstm(data, seq_len, input_dim, mon_ratio, eps, horizon=0):
    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf, size=data.shape, p=(mon_ratio, 1 - mon_ratio))
    data_x = np.zeros(shape=((data.shape[0] - seq_len) * data.shape[1], seq_len, input_dim), dtype='float32')
    data_y = np.zeros(shape=((data.shape[0] - seq_len) * data.shape[1], 1), dtype='float32')

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - seq_len):
            _x = _data[idx: idx + seq_len, flow]
            _label = _labels[idx: idx + seq_len, flow]

            data_x[i, :, 0] = _x
            data_x[i, :, 1] = _label

            data_y[i] = data[idx + seq_len, flow]

            i += 1

    return data_x, data_y


def load_dataset_lstm(seq_len, horizon, input_dim, mon_ratio, test_size,
                      raw_dataset_dir, dataset_dir, day_size, data_size, batch_size, eval_batch_size=None, **kwargs):
    raw_data = np.load(raw_dataset_dir)
    raw_data[raw_data <= 0] = 0.1

    # Convert traffic volume from byte to mega-byte
    raw_data = raw_data.astype("float32")

    raw_data = raw_data[:int(raw_data.shape[0] * data_size)]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, day_size=day_size)
    test_data2d = test_data2d[0:-day_size * 3]

    print('|--- Normalizing the train set.')
    data = {}

    scaler = StandardScaler(mean=train_data2d.mean(), std=train_data2d.std())
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data2d_norm

    x_train, y_train = create_data_lstm(train_data2d_norm, seq_len=seq_len, input_dim=input_dim,
                                        mon_ratio=mon_ratio, eps=train_data2d_norm.std())
    x_val, y_val = create_data_lstm(valid_data2d_norm, seq_len=seq_len, input_dim=input_dim,
                                    mon_ratio=mon_ratio, eps=train_data2d_norm.std())
    x_eval, y_eval = create_data_lstm(test_data2d_norm, seq_len=seq_len, input_dim=input_dim,
                                      mon_ratio=mon_ratio, eps=train_data2d_norm.std())

    for cat in ["train", "val", "eval"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)

        data['x_' + cat] = _x
        data['y_' + cat] = _y
    data['scaler'] = scaler

    return data


def create_data_lstm_ed(data, seq_len, input_dim, mon_ratio, eps, horizon=0):
    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf, size=data.shape, p=(mon_ratio, 1 - mon_ratio))
    e_x = np.zeros(shape=((data.shape[0] - seq_len) * data.shape[1], seq_len, input_dim), dtype='float32')
    d_x = np.zeros(shape=((data.shape[0] - seq_len) * data.shape[1], horizon, 1), dtype='float32')
    d_y = np.zeros(shape=((data.shape[0] - seq_len) * data.shape[1], horizon, 1), dtype='float32')

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for flow in range(_data.shape[1]):
        for idx in range(_data.shape[0] - seq_len - horizon):
            _x = _data[idx: idx + seq_len, flow]
            _label = _labels[idx: idx + seq_len, flow]

            e_x[i, :, 0] = _x
            e_x[i, :, 1] = _label

            d_x[i] = np.expand_dims(data[idx + seq_len - 1:idx + seq_len - 1 + horizon, flow], axis=1)

            d_y[i] = np.expand_dims(data[idx + seq_len:idx + seq_len + horizon, flow], axis=1)

            i += 1

    return e_x, d_x, d_y


def load_dataset_lstm_ed(seq_len, horizon, input_dim, mon_ratio, test_size,
                         raw_dataset_dir, dataset_dir, day_size, data_size, batch_size, eval_batch_size=None, **kwargs):
    raw_data = np.load(raw_dataset_dir)
    raw_data[raw_data <= 0] = 0.1

    # Convert traffic volume from byte to mega-byte
    raw_data = raw_data.astype("float32")

    raw_data = raw_data[:int(raw_data.shape[0] * data_size)]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, day_size=day_size)
    test_data2d = test_data2d[0:-day_size * 3]

    print('|--- Normalizing the train set.')
    data = {}

    scaler = StandardScaler(mean=train_data2d.mean(), std=train_data2d.std())
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data2d_norm

    encoder_input_train, decoder_input_train, decoder_target_train = create_data_lstm_ed(train_data2d_norm,
                                                                                         seq_len=seq_len,
                                                                                         input_dim=input_dim,
                                                                                         mon_ratio=mon_ratio,
                                                                                         horizon=horizon,
                                                                                         eps=train_data2d_norm.std())
    encoder_input_val, decoder_input_val, decoder_target_val = create_data_lstm_ed(valid_data2d_norm, seq_len=seq_len,
                                                                                   input_dim=input_dim,
                                                                                   mon_ratio=mon_ratio,
                                                                                   horizon=horizon,
                                                                                   eps=train_data2d_norm.std())
    encoder_input_eval, decoder_input_eval, decoder_target_eval = create_data_lstm_ed(test_data2d_norm, seq_len=seq_len,
                                                                                      input_dim=input_dim,
                                                                                      mon_ratio=mon_ratio,
                                                                                      horizon=horizon,
                                                                                      eps=train_data2d_norm.std())

    for cat in ["train", "val", "eval"]:
        e_x, d_x, d_y = locals()["encoder_input_" + cat], locals()["decoder_input_" + cat], locals()[
            "decoder_target_" + cat]
        print(cat, "e_x: ", e_x.shape, "d_x:", d_x.shape, "d_y:", d_y.shape)

        data["encoder_input_" + cat] = e_x
        data["decoder_input_" + cat] = d_x
        data["decoder_target_" + cat] = d_y
    data['scaler'] = scaler

    return data


def create_data_conv_lstm(data, seq_len, wide, high, channel, mon_ratio, eps):
    _tf = np.array([1.0, 0.0])
    _labels = np.random.choice(_tf, size=data.shape, p=(mon_ratio, 1 - mon_ratio))
    data_x = np.zeros(shape=((data.shape[0] - seq_len), seq_len, wide, high, channel), dtype='float32')
    data_y = np.zeros(shape=((data.shape[0] - seq_len), wide * high), dtype='float32')

    _data = np.copy(data)

    _data[_labels == 0.0] = np.random.uniform(_data[_labels == 0.0] - eps, _data[_labels == 0.0] + eps)

    i = 0
    for idx in range(_data.shape[0] - seq_len):
        _x = _data[idx: idx + seq_len]
        _label = _labels[idx: idx + seq_len]

        _x = np.reshape(_x, newshape=(seq_len, wide, high))
        _label = np.reshape(_label, newshape=(seq_len, wide, high))

        data_x[i, ..., 0] = _x
        data_x[i, ..., 1] = _label

        data_y[i] = data[idx + seq_len]

        i += 1

    return data_x, data_y


def load_dataset_conv_lstm(seq_len, wide, high, channel, mon_ratio,
                           raw_dataset_dir, day_size, data_size,
                           batch_size, eval_batch_size=None, **kwargs):
    data = {}

    raw_data = np.load(raw_dataset_dir)
    raw_data[raw_data <= 0] = 0.1

    raw_data = raw_data.astype("float32")

    raw_data = raw_data[:int(raw_data.shape[0] * data_size)]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, day_size=day_size)
    test_data2d = test_data2d[0:-day_size * 3]

    print('|--- Normalizing the train set.')
    scaler = StandardScaler(mean=train_data2d.mean(), std=train_data2d.std())
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data2d_norm

    x_train, y_train = create_data_conv_lstm(train_data2d_norm, seq_len=seq_len, wide=wide, high=high, channel=channel,
                                             mon_ratio=mon_ratio, eps=train_data2d_norm.std())
    x_val, y_val = create_data_conv_lstm(valid_data2d_norm, seq_len=seq_len, wide=wide, high=high, channel=channel,
                                         mon_ratio=mon_ratio, eps=train_data2d_norm.std())
    x_eval, y_eval = create_data_conv_lstm(test_data2d_norm, seq_len=seq_len, wide=wide, high=high, channel=channel,
                                           mon_ratio=mon_ratio, eps=train_data2d_norm.std())

    for cat in ["train", "val", "eval"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)

        data['x_' + cat] = _x
        data['y_' + cat] = _y

    data['scaler'] = scaler

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


