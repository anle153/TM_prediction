from statsmodels.tsa.arima_model import ARIMA
import requests, pandas as pd, numpy as np
from Utils.DataHelper import *
from Utils.DataPreprocessing import *
import os
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import arma_order_select_ic
import itertools
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.stattools import adfuller
from pandas import rolling_mean, rolling_std
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import r2_score


# NOT FINISH YET

HOME = os.path.expanduser('~')
# Abilene dataset path.
""" The Abilene dataset contains 

    X           a 2016x2016 matrix of flow volumes
    A           a 30x121 matrix of routing of the 121 flows over 30 edge between adjacent nodes in Abilene network.
    odnames     a 121x1 character vector of OD pair names
    edgenames   a 30x1 character vector of node pairs sharing an edge 
"""
ABILENE_DATASET_PATH = './Dataset/SAND_TM_Estimation_Data.mat'

# Geant dataset path.
""" The Geant dataset contains 

    X: a (10772 x 529) matrix of flow volumes
"""
DATAPATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices'
GEANT_DATASET_PATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices/Geant_dataset.csv'
GEANT_DATASET_NOISE_REMOVED_PATH = DATAPATH + '/Gean_noise_removed.csv'
# RNN CONFIGURATION
INPUT_DIM = 100
HIDDEN_DIM = 300
LOOK_BACK = 26
N_EPOCH = 200
BATCH_SIZE = 1024
WINDOW = LOOK_BACK
EPSILON_P = 0.0005


def test_stationarity(series, eps_p):
    """
    Test the stationarity by taking Dickey-Fuller test
    :param eps_p:
    :param series: (ndarray)
    :return: test_statistic, p_value, lags_used, n_observations, critical_values
    """

    # rolling mean
    rmean = series.rolling(window=7, center=False).mean()
    rstd = series.rolling(window=7, center=False).std()

    # plot statistics
    # orig = plt.plot(series, color='blue', label='Original')
    # mean = plt.plot(rmean, color='red', label='Rolling Mean')
    # std = plt.plot(rstd, color='black', label='Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)
    # plt.close()

    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    if dfoutput['p-value'] > eps_p:
        return False

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

        if dfoutput['Test Statistic'] > value:
            return False

    # print (dfoutput)
    return True


def calculate_q_parameters(data, interval):
    day_size = 24 * (60 / interval)

    flows_acf = []
    nlags = day_size
    for flow_id in range(len(data)):
        print('[calculate_q_parameters] flow %i' %flow_id)

        acf_out, confint = acf(data[flow_id].values, nlags=nlags, alpha=.05)
        for idx in range(1, len(acf_out)):
            if acf_out[idx] >= confint[idx, 1]:
                flows_acf.append(idx)
                break

    _bincount = np.bincount(flows_acf)
    q = np.argmax(_bincount)

    return q


def calculate_p_parameters(data, interval):
    day_size = 24 * (60 / interval)

    flows_pacf = []
    nlags = day_size
    for flow_id in range(len(data)):
        print('[calculate_p_parameters] flow %i' %flow_id)

        pacf_out, confint = pacf(data[flow_id].values, nlags=nlags, alpha=.05)
        for idx in range(len(pacf_out)):
            if pacf_out[idx] > confint[idx]:
                flows_pacf.append(idx)
                break

    _bincount = np.bincount(flows_pacf)
    p = np.argmax(_bincount)

    return p


def series_differncing(series, order):

    last_series_diff = series - series.shift()
    current_series = series.shift()

    for i in range(order - 1):
        series_diff = last_series_diff - (current_series - current_series.shift())
        current_series = current_series.shift()
        last_series_diff = series_diff

    return last_series_diff


def calculate_d_parameter(data):
    flows_diff = []
    for flow_id in range(len(data)):
        print('[calculate_d_parameter] flow %i' %flow_id)
        _d = 0
        test_result = test_stationarity(data[flow_id], eps_p=EPSILON_P)

        while not test_result:
            _d = _d + 1
            print('   --[calculate_d_parameter] flow %i - d = %i' % (flow_id, _d))
            tseries_diff = series_differncing(data[flow_id], order=_d)
            tseries_diff.dropna(inplace=True)
            test_result = test_stationarity(tseries_diff, eps_p=EPSILON_P)

        flows_diff.append(_d)

    _bincount = np.bincount(flows_diff)
    d = np.argmax(_bincount)

    return d


def calculate_arima_parameters(data):
    print('Calculate parameter d')
    # d = calculate_d_parameter(data)
    # print('d = %i' % d)

    print('Calculate parameter q')
    q = calculate_q_parameters(data, interval=5)
    print('q = %i' % q)

    print('Calculate parameter p')
    p = calculate_p_parameters(data, interval=5)
    print('p = %i' % p)

    return p, d, q


def arima(raw_data, dataset_name):

    test_name = 'arima'
    splitting_ratio = [0.7, 0.3]
    look_back = 26
    model_recorded_path = './Model_Recorded/' + dataset_name + '/' + test_name + '/'
    errors = np.empty((0, 4))
    sampling_ratio = 0.3

    seperated_data_set, centers_data_set = mean_std_flows_clustering(raw_data)
    data_set, data_scalers, data_cluster_lens = different_flows_scaling(seperated_data_set[1:],
                                                                        centers_data_set[1:])

    train_set, test_set = prepare_train_test_set(data_set, sampling_itvl=5, splitting_ratio=splitting_ratio)

    training_set_series = []
    for flow_id in range(train_set.shape[1]):
        flow_frame = pd.Series(train_set[:, flow_id])
        training_set_series.append(flow_frame)



    # Calculate p, d, q
    print('Calculate parameters p, d, q')
    p, d, q = calculate_arima_parameters(training_set_series)

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/p%i_d_%i_q_%i/' % (p, d, q)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    TM_prediction = np.empty((test_set.shape[0], 0))
    tf = np.array([True, False])

    measured_matrix = np.empty((test_set.shape[0], 0))

    for flow_id in range(test_set[1]):
        history = [x for x in train_set[:, flow_id]]
        predictions = list()
        measured_flow = np.random.choice(tf, size=(test_set.shape[0], 1), p=[sampling_ratio, 1 - sampling_ratio])

        for ts in range(test_set.shape[0]):
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_set[ts]
            if measured_flow[ts]:
                history.append(obs)
                predictions.append(obs)
            else:
                history.append(yhat)
                predictions.append(yhat)

            # print('predicted=%f, expected=%f' % (yhat, obs))

        measured_matrix = np.concatenate([measured_matrix, measured_flow], axis=1)

        TM_prediction = np.concatenate([TM_prediction, np.array(predictions).reshape((test_set.shape[0], 1))], axis=1)

    pred_tm = different_flows_invert_scaling(TM_prediction, scalers=data_scalers, cluster_lens=data_cluster_lens)
    pred_tm[pred_tm < 0] = 0
    ytrue = different_flows_invert_scaling(data=test_set, scalers=data_scalers,
                                           cluster_lens=data_cluster_lens)


    errors_by_day = calculate_error_ratio_by_day(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix,
                                                 sampling_itvl=5)
    mean_abs_error_by_day = mean_absolute_errors_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)

    rmse_by_day = root_means_squared_error_by_day(y_true=ytrue, y_pred=pred_tm, sampling_itvl=5)

    y3 = ytrue.flatten()
    y4 = pred_tm.flatten()
    a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
    a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)
    pred_confident = r2_score(y3, y4)

    err_rat = error_ratio(y_true=ytrue, y_pred=pred_tm, measured_matrix=measured_matrix)

    error = np.expand_dims(np.array([a_nmae, a_nmse, pred_confident, err_rat]), axis=0)

    errors = np.concatenate([errors, error], axis=0)

    # visualize_results_by_timeslot(y_true=ytrue,
    #                               y_pred=pred_tm,
    #                               measured_matrix=measured_matrix,
    #                               description=test_name + '_sampling_%f' % sampling_ratio,
    #                               saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/',
    #                               ts_plot=288*3)
    #
    visualize_retsult_by_flows(y_true=ytrue,
                               y_pred=pred_tm,
                               sampling_itvl=5,
                               description=test_name + '_sampling_%f' % sampling_ratio,
                               measured_matrix=measured_matrix,
                               saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/',
                               visualized_day=-1)

    print('--- Sampling ratio: %.2f - Means abs errors by day ---' % sampling_ratio)
    print(mean_abs_error_by_day)
    print('--- Sampling ratio: %.2f - RMSE by day ---' % sampling_ratio)
    print(rmse_by_day)
    print('--- Sampling ratio: %.2f - Error ratio by day ---' % sampling_ratio)
    print(errors_by_day)

    plt.title('Means abs errors by day\nSampling: %.2f' % sampling_ratio)
    plt.plot(range(len(mean_abs_error_by_day)), mean_abs_error_by_day)
    plt.xlabel('Day')
    plt.savefig(figures_saving_path + 'Means_abs_errors_by_day_sampling_%.2f.png' % sampling_ratio)
    plt.close()

    plt.title('RMSE by day\nSampling: %.2f' % sampling_ratio)
    plt.plot(range(len(rmse_by_day)), rmse_by_day)
    plt.xlabel('Day')
    plt.savefig(figures_saving_path + 'RMSE_by_day_sampling_%.2f.png' % sampling_ratio)
    plt.close()
    print('ERROR of testing at %.2f sampling' % sampling_ratio)
    print(errors)


def main():
    np.random.seed(10)

    Abilene24s_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24s.csv')
    arima(raw_data=Abilene24s_data, dataset_name='Abilene24s')


if __name__ == '__main__':
    main()
