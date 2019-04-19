import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')

from common.DataHelper import error_ratio, calculate_r2_score, rmse_tm_prediction
from common.DataPreprocessing import prepare_train_test_set

from pmdarima.arima import auto_arima
from common import Config


def build_auto_arima(data):
    model = auto_arima(data, start_p=1, start_q=1,
                       test='adf',  # use adftest to find optimal 'd'
                       max_p=3, max_q=3,  # maximum p and q
                       m=1,  # frequency of series
                       d=None,  # let model determine 'd'
                       seasonal=False,  # No Seasonality
                       start_P=0,
                       D=0,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    model.fit(data)

    return model


def calculate_ims_tm_test_data(test_data):
    ims_test_set = np.zeros(shape=(test_data.shape[0] - Config.IMS_STEP, Config.IMS_STEP, test_data.shape[1]))

    for ts in range(test_data.shape[0] - Config.IMS_STEP):
        multi_step_test_set = test_data[(ts + Config.LSTM_STEP): (ts + Config.LSTM_STEP + Config.IMS_STEP), :]
        ims_test_set[ts] = multi_step_test_set

    return ims_test_set


def train_test_arima(args, data):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    train_data, valid_data, test_data = prepare_train_test_set(data=data)

    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data = (train_data - mean_train) / std_train
    valid_data = (valid_data - mean_train) / std_train

    test_data_normalized = (test_data - mean_train) / std_train

    training_set_series = []
    for flow_id in range(train_data.shape[1]):
        flow_frame = pd.Series(train_data[:, flow_id])
        training_set_series.append(flow_frame)

    pred_tm = np.zeros((test_data_normalized.shape[0], test_data_normalized.shape[1]))
    tf = np.array([True, False])

    measured_matrix = np.random.choice(tf, size=(test_data_normalized.shape[0], test_data_normalized.shape[1]),
                                       p=[Config.MON_RAIO, 1 - Config.MON_RAIO])

    results_summary = pd.read_csv(Config.RESULTS_PATH + 'sample_results.csv')

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    for running_time in range(Config.TESTING_TIME):

        ims_pred_tm = np.empty(shape=(test_data_normalized.shape[0], Config.IMS_STEP, 0))

        for flow_id in range(test_data_normalized.shape[1]):

            training_set_series[flow_id].dropna(inplace=True)
            flow_train = training_set_series[flow_id].values

            history = [x for x in flow_train.astype(float)]
            predictions = list()

            measured_flow = measured_matrix[:, flow_id]

            # history = history[-int(Config.HISTORY_LENGTH * Config.LSTM_STEP):]

            flow_ims_pred = []

            # Fit all historical data to auto_arima
            model = build_auto_arima(history)

            for ts in range(test_data_normalized.shape[0]):

                output = model.predict(n_periods=Config.IMS_STEP)

                flow_ims_pred.append(output[0])

                yhat = output[0][0]
                obs = test_data_normalized[ts, flow_id]

                # Semi-recursive predicting
                if measured_flow[ts]:
                    history.append(obs)
                    predictions.append(obs)
                else:
                    history.append(yhat)
                    predictions.append(yhat)

            measured_matrix = np.concatenate([measured_matrix, measured_flow], axis=1)

            pred_tm[:, flow_id] = predictions
            flow_ims_pred = np.expand_dims(flow_ims_pred, axis=2)
            ims_pred_tm = np.concatenate([ims_pred_tm, flow_ims_pred], axis=2)

        pred_tm = pred_tm * std_train + mean_train
        ims_pred_tm = ims_pred_tm * std_train + mean_train

        measured_matrix = measured_matrix.astype(bool)

        err.append(error_ratio(y_true=test_data_normalized, y_pred=np.copy(pred_tm), measured_matrix=measured_matrix))
        r2_score.append(calculate_r2_score(y_true=test_data_normalized, y_pred=np.copy(pred_tm)))
        rmse.append(rmse_tm_prediction(y_true=test_data_normalized, y_pred=np.copy(pred_tm)))

        ims_test_set = calculate_ims_tm_test_data(test_data=test_data)

        measured_matrix = np.zeros(shape=ims_test_set.shape)
        err_ims.append(error_ratio(y_pred=ims_pred_tm,
                                   y_true=ims_test_set,
                                   measured_matrix=measured_matrix))

        r2_score_ims.append(calculate_r2_score(y_true=ims_test_set, y_pred=ims_pred_tm))
        rmse_ims.append(rmse_tm_prediction(y_true=ims_test_set, y_pred=ims_pred_tm))

    results_summary['running_time'] = range(Config.TESTING_TIME)
    results_summary['err'] = err
    results_summary['r2_score'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['err_ims'] = err_ims
    results_summary['r2_score_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    results_summary.to_csv(Config.RESULTS_PATH + '{}-{}-{}.csv'.format(data_name, alg_name, tag),
                           index=False)
