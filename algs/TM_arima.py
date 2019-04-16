import time

import matplotlib
import pandas as pd

matplotlib.use('Agg')
from statsmodels.tsa.arima_model import ARIMA

from common.DataHelper import *
from Utils.DataPreprocessing import *

from statsmodels.tsa.stattools import acf, pacf
from numpy.linalg import LinAlgError

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
EPSILON_P = 0.000000001
P = 1
D = 0
Q = 0


def test_stationarity(series, eps_p):
    """
    Test the stationarity by taking Dickey-Fuller test
    :param eps_p:
    :param series: (ndarray)
    :return: test_statistic, p_value, lags_used, n_observations, critical_values
    """

    acf_out, confi = acf(series, alpha=.05)
    if acf_out[1] <= -0.5:
        # Overdifferencing - stop differencing
        return -1
    elif (acf_out[1] <= eps_p) and (acf_out[1] > -0.5):
        # Stop differencing
        return 1
    elif acf_out[1] > eps_p:
        # Need more differencing
        return 0


def series_differncing(series, order):
    last_series_diff = series - series.shift()
    current_series = series.shift()

    for i in range(order - 1):
        series_diff = last_series_diff - (current_series - current_series.shift())
        current_series = current_series.shift()
        last_series_diff = series_diff

    return last_series_diff


def calculate_parameters(data):
    _p, _d, _q = 0, 0, 0
    test_result = test_stationarity(data, eps_p=EPSILON_P)
    std = data.std()

    series_stds = [std]
    series_diffs = [data]
    test_results = [test_result]

    while test_result == 0:
        if _d >= 2:
            break

        _d = _d + 1
        _data = data.copy()
        tseries_diff = series_differncing(_data, order=_d)
        tseries_diff.dropna(inplace=True)
        test_result = test_stationarity(tseries_diff, eps_p=EPSILON_P)

        series_diffs.append(tseries_diff)
        series_stds.append(tseries_diff.std())
        test_results.append(test_result)

    series_stds = np.array(series_stds)
    min_stds_idx = np.argmin(series_stds)

    acf_out, confi = acf(series_diffs[min_stds_idx], alpha=.05)
    pacf_out, pacf_confi = pacf(series_diffs[min_stds_idx], alpha=.05)

    if test_results[min_stds_idx] == -1:
        # Overdiferencing case -> Stop differencing and adding MA by looking at the number of lags
        # that cross the confidence interval of acf
        _q = 0
        for idx in range(1, len(acf_out)):
            if (acf_out[idx] < (confi[idx, 1] - acf_out[idx])) and (acf_out[idx] > (confi[idx, 0] - acf_out[idx])):
                break
            _q = _q + 1

        if _q == 0:
            _q = 1
        elif _q > 4:
            _q = 4

    else:
        # Series is stationary -> stop differencing -> AR term is the number of lags that cross the confidence interval
        # of pacf
        _p = 0
        for idx in range(1, len(pacf_out)):
            if (pacf_out[idx] < (pacf_confi[idx, 1] - pacf_out[idx])) and (
                    pacf_out[idx] > (pacf_confi[idx, 0] - pacf_out[idx])):
                break
            _p = _p + 1

        if _p == 0:
            _p = 1
        elif _p > 4:
            _p = 4

    return _p, _d, _q


def calculate_arima_parameters(data):
    p, d, q = calculate_parameters(data)
    print('(p, d, q) = (%i, %i, %i)' % (p, d, q))

    # print('Calculate parameter p')
    # p = calculate_p_parameters(data, interval=5)
    # print('p = %i' % p)
    #
    # print('Calculate parameter q')
    # q = calculate_q_parameters(data, interval=5)
    # print('q = %i' % q)

    return p, d, q


def arima(raw_data, dataset_name, n_timesteps, sampling_ratio, prediction_steps):
    test_name = 'arima'
    splitting_ratio = [0.8, 0.2]

    # look_back data
    look_back_range = [7]
    model_recorded_path = './Model_Recorded/' + dataset_name + '/' + test_name + '/'
    errors = np.empty((0, 5))

    sampling_timesteps_path = 'Sampling_%.3f_timesteps_%i' % (sampling_ratio, n_timesteps)
    model_name = 'arima'
    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/' % \
                  (dataset_name, test_name, sampling_timesteps_path, model_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    test_set = np.copy(test_set[0:-864, :])

    training_set = (train_set - mean_train) / std_train

    testing_set = np.copy(test_set)
    testing_set = (testing_set - mean_train) / std_train

    training_set_series = []
    for flow_id in range(training_set.shape[1]):
        flow_frame = pd.Series(training_set[:, flow_id])
        training_set_series.append(flow_frame)

    TM_prediction = np.empty((testing_set.shape[0], 0))
    tf = np.array([True, False])

    measured_matrix = np.empty((testing_set.shape[0], 0))

    day_size = 24 * (60 / 5)
    p, d, q = 4, 1, 0

    for running_time in range(10, 11, 1):

        ims_pred_tm = np.empty(shape=(testing_set.shape[0], prediction_steps, 0))

        for flow_id in range(testing_set.shape[1]):

            training_set_series[flow_id].dropna(inplace=True)
            flow_train = training_set_series[flow_id].values

            history = [x for x in flow_train.astype(float)]
            predictions = list()

            measured_flow = np.random.choice(tf, size=(testing_set.shape[0], 1), p=[sampling_ratio, 1 - sampling_ratio])

            history = history[-int(day_size * n_timesteps):]

            flow_ims_pred = []

            for ts in range(testing_set.shape[0]):

                print('[ARIMA] Predicting traffic flow %i at time slot %i - Sampling%.3f' % (
                flow_id, ts, sampling_ratio))

                try:
                    p, d, q = calculate_arima_parameters(pd.Series(history).astype(float))
                except LinAlgError as LA:
                    print(LA)
                    print('PASS LinAlgError')
                    pass
                except ValueError as VE:
                    print(VE)
                    print('PASS ValueError')
                    pass

                try:
                    model = ARIMA(history, order=(p, d, q))
                    model_fit = model.fit(disp=0, trend='nc')
                except LinAlgError as LA:
                    print(LA)
                    print('PASS LinAlgError')
                    pass
                except ValueError as VE:
                    print(VE)
                    print('PASS ValueError')
                    pass

                output = model_fit.forecast(steps=prediction_steps)

                flow_ims_pred.append(output[0])

                yhat = output[0][0]
                obs = testing_set[ts, flow_id]

                # Semi-recursive predicting
                if measured_flow[ts]:
                    history.append(obs)
                    predictions.append(obs)
                else:
                    history.append(yhat)
                    predictions.append(yhat)

                history = history[-int(day_size * n_timesteps):]

            measured_matrix = np.concatenate([measured_matrix, measured_flow], axis=1)

            TM_prediction = np.concatenate([TM_prediction, np.array(predictions).reshape((testing_set.shape[0], 1))],
                                           axis=1)
            flow_ims_pred = np.expand_dims(flow_ims_pred, axis=2)
            ims_pred_tm = np.concatenate([ims_pred_tm, flow_ims_pred], axis=2)

        pred_tm = TM_prediction * std_train + mean_train
        ims_pred_tm = ims_pred_tm * std_train + mean_train

        measured_matrix = measured_matrix.astype(bool)

        er = error_ratio(y_true=test_set, y_pred=pred_tm, measured_matrix=measured_matrix)
        r2 = calculate_r2_score(y_true=test_set, y_pred=pred_tm)
        rmse = rmse_tm_prediction(y_true=test_set, y_pred=pred_tm)
        print('ARIMA ERROR RATIO %.3f' % er)
        print('ARIMA RMSE %.3f' % rmse)
        print('ARIMA R2 %.3f' % r2)

        np.save(file=result_path + '[nii]Predicted_tm_running_time_%d' % running_time,
                arr=pred_tm)
        np.save(file=result_path + '[nii]Predicted_measured_matrix_running_time_%d' % running_time,
                arr=measured_matrix)
        np.save(file=result_path + '[nii]Predicted_multistep_tm_running_time_%d' % running_time,
                arr=ims_pred_tm)


def arima_no_ims(raw_data, dataset_name, n_timesteps, sampling_ratio):
    test_name = 'arima'
    splitting_ratio = [0.8, 0.2]

    # look_back data
    look_back_range = [7]
    model_recorded_path = './Model_Recorded/' + dataset_name + '/' + test_name + '/'
    errors = np.empty((0, 5))

    sampling_timesteps_path = 'Sampling_%.3f_timesteps_%i' % (sampling_ratio, n_timesteps)
    model_name = 'arima'
    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/' % \
                  (dataset_name, test_name, sampling_timesteps_path, model_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    test_set = np.copy(test_set[0:-864, :])

    training_set = (train_set - mean_train) / std_train

    testing_set = np.copy(test_set)
    testing_set = (testing_set - mean_train) / std_train

    training_set_series = []
    for flow_id in range(training_set.shape[1]):
        flow_frame = pd.Series(training_set[:, flow_id])
        training_set_series.append(flow_frame)

    TM_prediction = np.empty((testing_set.shape[0], 0))
    tf = np.array([True, False])

    measured_matrix = np.empty((testing_set.shape[0], 0))

    day_size = 24 * (60 / 5)
    p, d, q = 4, 1, 0

    for flow_id in range(testing_set.shape[1]):

        training_set_series[flow_id].dropna(inplace=True)
        flow_train = training_set_series[flow_id].values

        history = [x for x in flow_train.astype(float)]
        predictions = list()

        measured_flow = np.random.choice(tf, size=(testing_set.shape[0], 1), p=[sampling_ratio, 1 - sampling_ratio])

        history = history[-int(day_size * n_timesteps):]

        prediction_time = []

        for ts in range(testing_set.shape[0]):
            start_time = time.time()
            print('[ARIMA] Predicting traffic flow %i at time slot %i - Sampling%.3f' % (flow_id, ts, sampling_ratio))

            try:
                p, d, q = 4, 1, 0
            except LinAlgError as LA:
                print(LA)
                print('PASS LinAlgError')
                pass
            except ValueError as VE:
                print(VE)
                print('PASS ValueError')
                pass

            try:
                model = ARIMA(history, order=(p, d, q))
                model_fit = model.fit(disp=0, trend='nc')
            except LinAlgError as LA:
                print(LA)
                print('PASS LinAlgError')
                pass
            except ValueError as VE:
                print(VE)
                print('PASS ValueError')
                pass

            output = model_fit.forecast()

            yhat = output[0][0]
            obs = testing_set[ts, flow_id]

            # Semi-recursive predicting
            if measured_flow[ts]:
                history.append(obs)
                predictions.append(obs)
            else:
                history.append(yhat)
                predictions.append(yhat)

            history = history[-int(day_size * n_timesteps):]

            prediction_time.append(time.time() - start_time)

            if ts > 1000:
                break

        prediction_time = np.array(prediction_time)
        np.savetxt('[ARIMA]prediciton_time_one_step.csv', prediction_time, delimiter=',')
        return
        measured_matrix = np.concatenate([measured_matrix, measured_flow], axis=1)

    TM_prediction = np.concatenate([TM_prediction, np.array(predictions).reshape((testing_set.shape[0], 1))],
                                   axis=1)
    pred_tm = TM_prediction * std_train + mean_train

    measured_matrix = measured_matrix.astype(bool)

    er = error_ratio(y_true=test_set, y_pred=pred_tm, measured_matrix=measured_matrix)
    r2 = calculate_r2_score(y_true=test_set, y_pred=pred_tm)
    rmse = rmse_tm_prediction(y_true=test_set, y_pred=pred_tm)
    print('ARIMA ERROR RATIO %.3f' % er)
    print('ARIMA RMSE %.3f' % rmse)
    print('ARIMA R2 %.3f' % r2)


# def arima_multistep_tm_prediction(ims_pred_output, ims_tm_pred, flow_id):

def main():
    np.random.seed(10)

    Abilene24 = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24.csv')
    n_timesteps = 28
    sampling_ratio = 0.15

    # arima(raw_data=Abilene24, dataset_name='Abilene24', n_timesteps=n_timesteps, sampling_ratio=sampling_ratio,
    #       prediction_steps=12)
    arima_no_ims(raw_data=Abilene24, dataset_name='Abilene24', n_timesteps=n_timesteps, sampling_ratio=sampling_ratio)


if __name__ == '__main__':
    main()
