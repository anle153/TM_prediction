import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor

from common import Config
from common.DataPreprocessing import data_scalling, prepare_train_valid_test_2d, parallel_create_offline_xgb_data
from common.error_utils import calculate_rmse, calculate_r2_score, error_ratio


def ims_tm_test_data(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.XGB_IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.XGB_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.XGB_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def prepare_input_online_prediction(data):
    dataX = np.zeros(shape=(data.shape[1], Config.XGB_STEP))
    for flow_id in range(data.shape[1]):
        x = data[-Config.XGB_STEP:, flow_id]
        dataX[flow_id] = x

    return dataX


def ims_tm_prediction(init_data, model, init_labels):
    multi_steps_tm = np.zeros(shape=(init_data.shape[0] + Config.XGB_IMS_STEP, init_data.shape[1]))
    multi_steps_tm[0:Config.XGB_STEP, :] = init_data

    labels = np.zeros(shape=(init_labels.shape[0] + Config.XGB_IMS_STEP, init_labels.shape[1]))
    labels[0:Config.XGB_STEP, :] = init_labels

    for ts_ahead in range(Config.XGB_IMS_STEP):
        rnn_input = prepare_input_online_prediction(data=multi_steps_tm)
        predictX = model.predict(rnn_input)
        multi_steps_tm[ts_ahead] = predictX.T

    return multi_steps_tm[-1, :]


def predict_xgb(init_data, test_data, model):
    tf_a = np.array([True, False])
    labels = np.ones(shape=init_data.shape)

    tm_pred = np.zeros(shape=(init_data.shape[0] + test_data.shape[0], test_data.shape[1]))

    ims_tm = np.zeros(shape=(test_data.shape[0] - Config.XGB_IMS_STEP + 1, test_data.shape[1]))

    # Predict the TM from time slot look_back
    for ts in tqdm(range(test_data.shape[0])):
        # This block is used for iterated multi-step traffic matrices prediction

        if Config.XGB_IMS and (ts <= test_data.shape[0] - Config.XGB_IMS_STEP):
            ims_tm[ts] = ims_tm_prediction(init_data=tm_pred[ts:ts + Config.XGB_STEP, :],
                                           model=model,
                                           init_labels=labels[ts:ts + Config.XGB_STEP, :])

        # Create 3D input for rnn
        rnn_input = prepare_input_online_prediction(data=tm_pred)

        # Get the TM prediction of next time slot
        predictX = model.predict(rnn_input)

        pred = np.expand_dims(predictX, axis=1)

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf_a,
                                                   size=(test_data.shape[1]),
                                                   p=[Config.XGB_MON_RATIO, 1 - Config.XGB_MON_RATIO]), axis=0)
        labels = np.concatenate([labels, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = pred.T * inv_sampling

        ground_true = test_data[ts, :]

        measured_input = np.expand_dims(ground_true, axis=0) * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shap e[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        tm_pred[ts + Config.XGB_STEP] = new_input

    return tm_pred[Config.XGB_STEP:, :], labels[Config.XGB_STEP:, :], ims_tm


def train_xgboost(data, args):
    print('|-- Run xgboost model training.')

    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')
    train_data_normalized2d, valid_data_normalized2d, _, scalers = data_scalling(train_data2d,
                                                                                 valid_data2d,
                                                                                 test_data2d)
    saving_path = Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    xgb_model = XGBRegressor(n_jobs=Config.XGB_NJOBS)

    trainX, trainY = parallel_create_offline_xgb_data(train_data_normalized2d,
                                                      Config.XGB_STEP,
                                                      Config.XGB_FEATURES,
                                                      Config.XGB_MON_RATIO,
                                                      0.5)

    print('|--- Training model.')

    if os.path.isfile(saving_path + 'xgb.models'):
        xgb_model.load_model(saving_path + 'xgb.models')
    else:
        xgb_model.fit(trainX, trainY)
        xgb_model.save_model(saving_path + 'xgb.models')

    run_test(valid_data2d, valid_data_normalized2d, train_data_normalized2d[-Config.XGB_STEP:],
             xgb_model, scalers, args)


def run_test(test_data2d, test_data_normalized2d, init_data2d, xgb_model, scalers, args):
    print('|--- Run test xgb!')
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    results_summary = pd.DataFrame(index=range(Config.XGB_TESTING_TIME),
                                   columns=['No.', 'err', 'r2', 'rmse', 'err_ims', 'r2_ims', 'rmse_ims'])
    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    ims_test_set = ims_tm_test_data(test_data=test_data2d)
    measured_matrix_ims = np.zeros(shape=ims_test_set.shape)

    if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name)):
        np.save(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name),
                test_data2d)

    if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER)):
        np.save(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER),
                test_data_normalized2d)
    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name,
                                                                      alg_name, tag, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER))

    for i in range(Config.XGB_TESTING_TIME):
        print('|--- Running time: {}'.format(i))
        pred_tm2d, measured_matrix2d, ims_tm2d = predict_xgb(init_data=init_data2d,
                                                             test_data=test_data_normalized2d,
                                                             model=xgb_model)

        np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred_scaled-{}.npy'.format(data_name, alg_name, tag,
                                                                              Config.SCALER, i),
                pred_tm2d)

        pred_tm_invert2d = scalers.inverse_transform(pred_tm2d)

        err.append(error_ratio(y_true=test_data2d, y_pred=pred_tm_invert2d, measured_matrix=measured_matrix2d))
        r2_score.append(calculate_r2_score(y_true=test_data2d, y_pred=pred_tm_invert2d))
        rmse.append(calculate_rmse(y_true=test_data2d, y_pred=pred_tm_invert2d))

        if Config.XGB_IMS:
            ims_tm_invert2d = scalers.inverse_transform(ims_tm2d)

            err_ims.append(error_ratio(y_pred=ims_tm_invert2d,
                                       y_true=ims_test_set,
                                       measured_matrix=measured_matrix_ims))

            r2_score_ims.append(calculate_r2_score(y_true=ims_test_set, y_pred=ims_tm_invert2d))
            rmse_ims.append(calculate_rmse(y_true=ims_test_set, y_pred=ims_tm_invert2d))

        else:
            err_ims.append(0)
            r2_score_ims.append(0)
            rmse_ims.append(0)

        print('Result: err\trmse\tr2 \t\t err_ims\trmse_ims\tr2_ims')
        print('        {}\t{}\t{} \t\t {}\t{}\t{}'.format(err[i], rmse[i], r2_score[i],
                                                          err_ims[i], rmse_ims[i],
                                                          r2_score_ims[i]))

        np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred-{}.npy'.format(data_name, alg_name, tag,
                                                                       Config.SCALER, i),
                pred_tm_invert2d)
        np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/measure-{}.npy'.format(data_name, alg_name, tag,
                                                                          Config.SCALER, i),
                measured_matrix2d)

    results_summary['No.'] = range(Config.XGB_TESTING_TIME)
    results_summary['err'] = err
    results_summary['r2'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['err_ims'] = err_ims
    results_summary['r2_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    results_summary.to_csv(Config.RESULTS_PATH + '{}-{}-{}-{}/results.csv'.format(data_name,
                                                                                  alg_name, tag, Config.SCALER),
                           index=False)
