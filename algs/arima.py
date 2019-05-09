import pickle

import matplotlib
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from tqdm import tqdm

from common import Config
from common.DataPreprocessing import prepare_train_test_2d, data_scalling
from common.error_utils import error_ratio, calculate_r2_score, calculate_rmse

matplotlib.use('Agg')


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


def ims_tm_test_data(test_data):
    ims_test_set = np.zeros(shape=(test_data.shape[0] - Config.ARIMA_IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.ARIMA_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.ARIMA_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def train_arima(data):
    alg_name = Config.ALG
    tag = Config.TAG
    data_name = Config.DATA_NAME

    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    train_data, test_data = prepare_train_test_2d(data=data, day_size=day_size)

    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data_normalized = (train_data - mean_train) / std_train
    test_data_normalized = (test_data - mean_train) / std_train

    training_set_series = []
    for flow_id in range(train_data_normalized.shape[1]):
        flow_frame = pd.Series(train_data_normalized[:, flow_id])
        training_set_series.append(flow_frame)

    import os
    if not os.path.exists(Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name,
                                                                    alg_name,
                                                                    tag,
                                                                    Config.SCALER)):
        os.makedirs(Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name,
                                                              alg_name,
                                                              tag,
                                                              Config.SCALER))

    for flow_id in tqdm(range(test_data_normalized.shape[1])):
        training_set_series[flow_id].dropna(inplace=True)
        flow_train = training_set_series[flow_id].values

        history = [x for x in flow_train.astype(float)]

        # Fit all historical data to auto_arima
        model = build_auto_arima(history)

        saved_model = open(Config.MODEL_SAVE + '{}-{}-{}-{}/{}.model'.format(data_name,
                                                                             alg_name,
                                                                             tag,
                                                                             Config.SCALER,
                                                                             flow_id), 'wb')
        pickle.dump(model, saved_model, 2)


def test_arima(data):
    alg_name = Config.ALG
    tag = Config.TAG
    data_name = Config.DATA_NAME
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    if not Config.ALL_DATA:
        data = data[0:day_size * Config.NUM_DAYS, :]

    train_data2d, test_data2d = prepare_train_test_2d(data=data, day_size=day_size)
    if 'Abilene' in data_name:
        print('|--- Remove last 3 days in test data.')
        test_data2d = test_data2d[0:-day_size * 3]

    train_data_normalized2d, _, test_data_normalized2d, scalers = data_scalling(train_data2d,
                                                                                [],
                                                                                test_data2d)

    training_set_series = []
    for flow_id in range(train_data_normalized2d.shape[1]):
        flow_frame = pd.Series(train_data_normalized2d[:, flow_id])
        training_set_series.append(flow_frame)

    tf = np.array([True, False])

    results_summary = pd.DataFrame(index=range(Config.ARIMA_TESTING_TIME),
                                   columns=['No.', 'err', 'r2', 'rmse', 'err_ims', 'r2_ims', 'rmse_ims'])

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    import os
    if not os.path.exists(Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name,
                                                                    alg_name,
                                                                    tag,
                                                                    Config.SCALER)):
        os.makedirs(Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name,
                                                              alg_name,
                                                              tag,
                                                              Config.SCALER))

    ims_test_set2d = ims_tm_test_data(test_data=test_data2d)
    measured_matrix_ims = np.zeros(shape=ims_test_set2d.shape)

    pred_tm2d = np.zeros((test_data_normalized2d.shape[0], test_data_normalized2d.shape[1]))
    ims_pred_tm2d = np.zeros(
        (test_data_normalized2d.shape[0] - Config.ARIMA_IMS_STEP + 1, test_data_normalized2d.shape[1]))

    if not os.path.isfile(Config.MODEL_SAVE + '{}-{}-{}-{}/{}.model'.format(data_name,
                                                                            alg_name,
                                                                            tag,
                                                                            Config.SCALER,
                                                                            0)):
        train_arima(data)

    if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name)):
        np.save(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name),
                test_data2d)

    if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER)):
        np.save(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER),
                test_data_normalized2d)

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name,
                                                                      alg_name, tag, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER))

    for running_time in range(Config.ARIMA_TESTING_TIME):
        print('|--- Run time: {}'.format(running_time))

        measured_matrix2d = np.random.choice(tf,
                                             size=(test_data_normalized2d.shape[0], test_data_normalized2d.shape[1]),
                                             p=[Config.ARIMA_MON_RATIO, 1 - Config.ARIMA_MON_RATIO])

        for flow_id in tqdm(range(test_data_normalized2d.shape[1])):
            training_set_series[flow_id].dropna(inplace=True)
            flow_train = training_set_series[flow_id].values

            history = [x for x in flow_train.astype(float)]

            predictions = np.zeros(shape=(test_data_normalized2d.shape[0]))

            measured_flow = measured_matrix2d[:, flow_id]

            flow_ims_pred = np.zeros(shape=(test_data_normalized2d.shape[0] - Config.ARIMA_IMS_STEP + 1))

            # Load trained arima model
            saved_model = open(Config.MODEL_SAVE + '{}-{}-{}-{}/{}.model'.format(data_name,
                                                                                 alg_name,
                                                                                 tag,
                                                                                 Config.SCALER, flow_id), 'rb')
            model = pickle.load(saved_model)

            for ts in range(test_data_normalized2d.shape[0]):

                if (ts % (day_size * Config.ARIMA_UPDATE) == 0) and ts != 0:
                    print('|--- Update arima model at ts: {}'.format(ts))
                    try:
                        model = build_auto_arima(history)
                    except:
                        pass
                if Config.ARIMA_IMS:
                    output = model.predict(n_periods=Config.ARIMA_IMS_STEP)
                    if ts <= (test_data_normalized2d.shape[0] - Config.ARIMA_IMS_STEP):
                        flow_ims_pred[ts] = output[-1]

                else:
                    output = model.predict(n_periods=1)

                yhat = output[0]
                obs = test_data_normalized2d[ts, flow_id]

                # Semi-recursive predicting
                if measured_flow[ts]:
                    history.append(obs)
                    predictions[ts] = obs
                else:
                    history.append(yhat)
                    predictions[ts] = yhat

            pred_tm2d[:, flow_id] = predictions

            if Config.ARIMA_IMS:
                ims_pred_tm2d[:, flow_id] = flow_ims_pred

        np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred_scaled-{}.npy'.format(data_name, alg_name, tag,
                                                                              Config.SCALER, running_time), pred_tm2d)

        pred_tm_invert2d = scalers.inverse_transform(pred_tm2d)

        measured_matrix2d = measured_matrix2d.astype(bool)

        # Calculate error
        err.append(error_ratio(y_true=test_data2d,
                               y_pred=pred_tm_invert2d,
                               measured_matrix=measured_matrix2d))
        r2_score.append(calculate_r2_score(y_true=test_data2d, y_pred=pred_tm_invert2d))
        rmse.append(calculate_rmse(y_true=test_data2d, y_pred=pred_tm_invert2d))

        # Calculate error of ims
        if Config.ARIMA_IMS:
            ims_tm_invert2d = scalers.inverse_transform(ims_pred_tm2d)

            err_ims.append(error_ratio(y_pred=ims_tm_invert2d,
                                       y_true=ims_test_set2d,
                                       measured_matrix=measured_matrix_ims))
            r2_score_ims.append(calculate_r2_score(y_true=ims_test_set2d, y_pred=ims_tm_invert2d))
            rmse_ims.append(calculate_rmse(y_true=ims_test_set2d, y_pred=ims_tm_invert2d))
        else:
            err_ims.append(0)
            r2_score_ims.append(0)
            rmse_ims.append(0)

        print('Result: err\trmse\tr2 \t\t err_ims\trmse_ims\tr2_ims')
        print('        {}\t{}\t{} \t\t {}\t{}\t{}'.format(err[running_time], rmse[running_time], r2_score[running_time],
                                                          err_ims[running_time], rmse_ims[running_time],
                                                          r2_score_ims[running_time]))

        np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred-{}.npy'.format(data_name, alg_name, tag,
                                                                       Config.SCALER, running_time),
                pred_tm_invert2d)
        np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/measure-{}.npy'.format(data_name, alg_name, tag,
                                                                          Config.SCALER, running_time),
                measured_matrix2d)

    results_summary['No.'] = range(Config.ARIMA_TESTING_TIME)
    results_summary['err'] = err
    results_summary['r2'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['err_ims'] = err_ims
    results_summary['r2_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    results_summary.to_csv(Config.RESULTS_PATH +
                           '{}-{}-{}-{}/results.csv'.format(data_name, alg_name, tag, Config.SCALER),
                           index=False)
