import os

import numpy as np
from matplotlib import pyplot as plt

from common import Config
from common.error_utils import calculate_r2_score, error_ratio, calculate_rmse


def plot_pred_results(data_name, alg_name, tag, nflows):
    data_raw = np.load(Config.DATA_PATH + '{}.npy'.format(data_name))

    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    # train_set, valid_set, test_set = prepare_train_valid_test_3d(data_raw, day_size=day_size)

    ground_true = np.load(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name))
    if Config.MIN_MAX_SCALER:
        ground_true_scaled = np.load(Config.RESULTS_PATH + 'ground_true_scaled_{}_minmax.npy'.format(data_name))
    else:
        ground_true_scaled = np.load(Config.RESULTS_PATH + 'ground_true_scaled_{}.npy'.format(data_name))

    if 'fwbw-conv-lstm' in alg_name or 'fwbw-convlstm' in alg_name:
        run_time = Config.FWBW_CONV_LSTM_TESTING_TIME
    elif 'conv-lstm' in alg_name or 'convlstm' in alg_name:
        run_time = Config.CONV_LSTM_TESTING_TIME
    elif 'lstm-nn' in alg_name:
        run_time = Config.LSTM_TESTING_TIME
    elif 'arima' in alg_name:
        run_time = Config.ARIMA_TESTING_TIME
    elif 'holt-winter' in alg_name:
        run_time = Config.HOLT_WINTER_TESTING_TIME
    else:
        raise ValueError('Unkown alg!')

    day_x = 1
    day_y = 0

    for i in range(run_time):

        plotted_path_raw = Config.RESULTS_PATH + \
                           '{}-{}-{}-{}/plot_raw/run_{}/'.format(data_name,
                                                                 alg_name,
                                                                 tag,
                                                                 Config.ADDED_RESULT_NAME, i)

        plotted_path_scaled = Config.RESULTS_PATH + \
                              '{}-{}-{}-{}/plot_scaled/run_{}/'.format(data_name,
                                                                       alg_name,
                                                                       tag,
                                                                       Config.ADDED_RESULT_NAME, i)
        if not os.path.exists(plotted_path_raw):
            os.makedirs(plotted_path_raw)
        if not os.path.exists(plotted_path_scaled):
            os.makedirs(plotted_path_scaled)

        pred = np.load(Config.RESULTS_PATH + '{}-{}-{}-{}/pred-{}.npy'.format(data_name, alg_name, tag,
                                                                              Config.ADDED_RESULT_NAME, i))
        pred_scaled = np.load(Config.RESULTS_PATH + '{}-{}-{}-{}/pred_scaled-{}.npy'.format(data_name, alg_name, tag,
                                                                                            Config.ADDED_RESULT_NAME,
                                                                                            i))
        measure_matrix = np.load(Config.RESULTS_PATH + '{}-{}-{}-{}/measure-{}.npy'.format(data_name, alg_name, tag,
                                                                                           Config.ADDED_RESULT_NAME, i))

        plot_flow(plotted_path_raw, ground_true, 'Actual', pred, 'Pred', day_x, day_y, day_size, nflows)
        plot_flow(plotted_path_scaled, ground_true_scaled, 'Actual', pred_scaled, 'Pred', day_x, day_y, day_size,
                  nflows)

        print('|--- Error Ratio: {}'.format(error_ratio(y_true=ground_true,
                                                        y_pred=pred,
                                                        measured_matrix=measure_matrix)))
        print('|--- RMSE: {}'.format(calculate_rmse(y_true=ground_true,
                                                    y_pred=pred)))
        print('|--- R2: {}'.format(calculate_r2_score(y_true=ground_true,
                                                      y_pred=pred)))


def plot_flow(save_path, series1, label1, series2, label2, day_x, day_y, day_size, nflows):
    for flow_x in range(nflows):
        for flow_y in range(nflows):
            plt.plot(range((series1.shape[0] - day_size * day_x), (series1.shape[0] - day_size * day_y)),
                     series1[(series1.shape[0] - day_size * day_x):(series1.shape[0] - day_size * day_y),
                     flow_x, flow_y], label=label1)
            plt.plot(range((series2.shape[0] - day_size * day_x), (series2.shape[0] - day_size * day_y)),
                     series2[(series2.shape[0] - day_size * day_x):(series2.shape[0] - day_size * day_y), flow_x,
                     flow_y],
                     label=label2)
            plt.xlabel('Timestep')
            plt.ylabel('Traffic Load')

            plt.legend()

            plt.savefig(save_path + 'Flow-{}-{}.png'.format(flow_x, flow_y))
            plt.close()
