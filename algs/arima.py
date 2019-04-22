import pickle

import matplotlib
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima

from common import Config
from common.DataPreprocessing import prepare_train_test_2d
from common.error_utils import error_ratio, calculate_r2_score, calculate_rmse
from tqdm import tqdm

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
    ims_test_set = np.zeros(shape=(test_data.shape[0] - Config.IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.IMS_STEP + 1] = test_data[i]

    return ims_test_set


def train_arima(args, data):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    train_data, test_data = prepare_train_test_2d(data=data)

    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data_normalized = (train_data - mean_train) / std_train
    test_data_normalized = (test_data - mean_train) / std_train

    training_set_series = []
    for flow_id in range(train_data_normalized.shape[1]):
        flow_frame = pd.Series(train_data_normalized[:, flow_id])
        training_set_series.append(flow_frame)

    import os
    if not os.path.exists(Config.MODEL_SAVE + 'arima/'):
        os.makedirs(Config.MODEL_SAVE + 'arima/')

    for flow_id in tqdm(range(test_data_normalized.shape[1])):
        training_set_series[flow_id].dropna(inplace=True)
        flow_train = training_set_series[flow_id].values

        history = [x for x in flow_train.astype(float)]

        # Fit all historical data to auto_arima
        model = build_auto_arima(history)

        saved_model = open(Config.MODEL_SAVE + 'arima/{}-{}-{}-{}'.format(flow_id, data_name, alg_name, tag), 'wb')
        pickle.dump(model, saved_model, 2)


def test_arima(data, args):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    train_data, test_data = prepare_train_test_2d(data=data)

    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data_normalized = (train_data - mean_train) / std_train
    test_data_normalized = (test_data - mean_train) / std_train

    training_set_series = []
    for flow_id in range(train_data_normalized.shape[1]):
        flow_frame = pd.Series(train_data_normalized[:, flow_id])
        training_set_series.append(flow_frame)

    tf = np.array([True, False])

    results_summary = pd.read_csv(Config.RESULTS_PATH + 'sample_results.csv')

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    import os
    if not os.path.exists(Config.MODEL_SAVE + 'arima/'):
        os.makedirs(Config.MODEL_SAVE + 'arima/')

    ims_test_set = ims_tm_test_data(test_data=test_data)
    measured_matrix_ims = np.zeros(shape=ims_test_set.shape)

    pred_tm = np.zeros((test_data_normalized.shape[0], test_data_normalized.shape[1]))
    ims_pred_tm = np.zeros((test_data_normalized.shape[0] - Config.IMS_STEP + 1, test_data_normalized.shape[1]))

    if not os.path.isfile(Config.MODEL_SAVE + 'arima/{}-{}-{}-{}'.format(0, data_name, alg_name, tag)):
        train_arima(args, data)

    for running_time in range(Config.TESTING_TIME):
        print('|--- Run time: {}'.format(running_time))

        measured_matrix = np.random.choice(tf, size=(test_data_normalized.shape[0], test_data_normalized.shape[1]),
                                           p=[Config.MON_RAIO, 1 - Config.MON_RAIO])

        for flow_id in tqdm(range(test_data_normalized.shape[1])):
            training_set_series[flow_id].dropna(inplace=True)
            flow_train = training_set_series[flow_id].values

            history = [x for x in flow_train.astype(float)]

            predictions = np.zeros(shape=(test_data_normalized.shape[0]))

            measured_flow = measured_matrix[:, flow_id]

            flow_ims_pred = np.zeros(shape=(test_data_normalized.shape[0] - Config.IMS_STEP + 1))

            # Load trained arima model
            saved_model = open(Config.MODEL_SAVE + 'arima/{}-{}-{}-{}'.format(flow_id, data_name, alg_name, tag), 'rb')
            model = pickle.load(saved_model)

            for ts in range(test_data_normalized.shape[0]):

                if (ts % 288 == 0) and ts != 0:
                    print('|--- Update arima model at ts: {}'.format(ts))
                    try:
                        model = build_auto_arima(history)
                    except:
                        pass

                output = model.predict(n_periods=Config.IMS_STEP)

                if ts <= (test_data_normalized.shape[0] - Config.IMS_STEP):
                    flow_ims_pred[ts] = output[-1]

                yhat = output[0]
                obs = test_data_normalized[ts, flow_id]

                # Semi-recursive predicting
                if measured_flow[ts]:
                    history.append(obs)
                    predictions[ts] = obs
                else:
                    history.append(yhat)
                    predictions[ts] = yhat

            pred_tm[:, flow_id] = predictions
            ims_pred_tm[:, flow_id] = flow_ims_pred

        pred_tm = pred_tm * std_train + mean_train
        ims_pred_tm = ims_pred_tm * std_train + mean_train

        measured_matrix = measured_matrix.astype(bool)

        # Calculate error
        err.append(error_ratio(y_true=test_data,
                               y_pred=pred_tm,
                               measured_matrix=measured_matrix))
        r2_score.append(calculate_r2_score(y_true=test_data, y_pred=pred_tm))
        rmse.append(calculate_rmse(y_true=test_data, y_pred=pred_tm))

        # Calculate error of ims
        err_ims.append(error_ratio(y_pred=ims_pred_tm,
                                   y_true=ims_test_set,
                                   measured_matrix=measured_matrix_ims))
        r2_score_ims.append(calculate_r2_score(y_true=ims_test_set, y_pred=ims_pred_tm))
        rmse_ims.append(calculate_rmse(y_true=ims_test_set, y_pred=ims_pred_tm))

        print('Result: err\trmse\tr2 \t\t err_ims\trmse_ims\tr2_ims')
        print('        {}\t{}\t{} \t\t {}\t{}\t{}'.format(err[running_time], rmse[running_time], r2_score[running_time],
                                                          err_ims[running_time], rmse_ims[running_time],
                                                          r2_score_ims[running_time]))

    results_summary['running_time'] = range(Config.TESTING_TIME)
    results_summary['err'] = err
    results_summary['r2_score'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['err_ims'] = err_ims
    results_summary['r2_score_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    results_summary.to_csv(Config.RESULTS_PATH + '{}-{}-{}-mon-{}.csv'.format(data_name,
                                                                              alg_name,
                                                                              tag,
                                                                              Config.MON_RAIO),
                           index=False)


def main(args):
    from common import Config
    from common.cmd_utils import common_arg_parser
    import os

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)

    datapath = '/home/anle/TM_prediction/Dataset/Abilene2d.npy'

    if not os.path.isfile(datapath):
        from common.DataHelper import create_abilene_data_2d
        create_abilene_data_2d('/home/anle/AbileneTM-all/')
    data = np.load(datapath)

    test_arima(data, args)

    return


if __name__ == '__main__':
    import sys

    main(sys.argv)
