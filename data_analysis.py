import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from tqdm import tqdm

data_name = 'Geant'
day_size = 96
data = np.load('Dataset/{}/{}.npy'.format(data_name, data_name))


def correlation_matrix(data, seq_len):
    corr_matrices = np.zeros(shape=(data.shape[0] - seq_len, data.shape[1], data.shape[1]), dtype='float32')

    for i in tqdm(range(data.shape[0] - seq_len)):
        data_corr = data[i:i + seq_len]
        df = pd.DataFrame(data_corr, index=range(data_corr.shape[0]),
                          columns=['{}'.format(x + 1) for x in range(data_corr.shape[1])])

        corr_mx = df.corr().values
        corr_mx[np.isnan(corr_mx)] = 0
        corr_mx[np.isinf(corr_mx)] = 0
        corr_matrices[i] = corr_mx

    nan_idx = []
    for i in range(corr_matrices.shape[0]):
        if not np.any(np.isnan(corr_matrices[i])) and not np.any(np.isinf(corr_matrices[i])):
            nan_idx.append(i)

    corr_matrices = corr_matrices[nan_idx]
    print(np.std(corr_matrices, axis=0))
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


if __name__ == '__main__':
    train_size = int((data.shape[0] / day_size) * 0.6)

    corr_mx = correlation_matrix(data[:train_size * day_size], seq_len=int(day_size / 2))
    corr_mx[corr_mx <= 0.0] = 0.0

    corr_mx = (corr_mx - np.min(corr_mx)) / (np.max(corr_mx) - np.min(corr_mx))

    corr_mx[corr_mx > 0.5] = 1.0
    corr_mx[corr_mx <= 0.5] = 0.0
    print(corr_mx.sum())

    path = 'Dataset/{}/adj_mx'.format(data_name)
    np.save(path, corr_mx)


def get_auto_corre():
    for i in tqdm(range(data.shape[1])):
        path = 'Dataset/{}/raw_plot/auto_correlation/'.format(data_name)
        try:
            plot_acf(data[:day_size * 14, i], lags=day_size / 2)
            plt.savefig(path + 'flow_{}.png'.format(i))
            plt.close()
        except:
            import os
            os.makedirs(path)

# for i in tqdm(range(data.shape[1])):
#     path = 'Dataset/{}/raw_plot/'.format(data_name)
#     try:
#         plt.plot(data[:day_size * 10, i]/1e3)
#         plt.savefig(path + 'flow_{}.png'.format(i))
#         plt.close()
#     except:
#         import os
#         os.makedirs(path)
#         plt.plot(data[:day_size * 10, i]/1e3)
#         plt.savefig(path + 'flow_{}.png'.format(i))
#         plt.close()
