from matplotlib import pyplot as plt
from common import Config
import numpy as np
import os
from common.error_utils import calculate_r2_score, error_ratio, calculate_rmse
from common.DataPreprocessing import prepare_train_valid_test_3d


def plot_pred_results(data_name, alg_name, tag, nflows, ndays):
    data_raw = np.load(Config.DATA_PATH + '{}.npy'.format(data_name))

    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    # train_set, valid_set, test_set = prepare_train_valid_test_3d(data_raw, day_size=day_size)

    plotted_path = Config.RESULTS_PATH + 'Plotted_results/{}-{}-{}-{}/'.format(data_name,
                                                                               alg_name,
                                                                               tag,
                                                                               Config.ADDED_RESULT_NAME)
    if not os.path.exists(plotted_path):
        os.makedirs(plotted_path)

    test_data = np.load(Config.RESULTS_PATH + '[test-data]{}.npy'.format(data_name))

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

    day_x = 3
    day_y = 2

    for i in range(run_time):
        pred = np.load(Config.RESULTS_PATH + '[pred-{}]{}-{}-{}-{}.npy'.format(i, data_name, alg_name, tag,
                                                                               Config.ADDED_RESULT_NAME))
        measure_matrix = np.load(Config.RESULTS_PATH + '[measure-{}]{}-{}-{}-{}.npy'.format(i, data_name, alg_name, tag,
                                                                                            Config.ADDED_RESULT_NAME))

        for flow_x in range(12):
            for flow_y in range(12):
                plt.plot(range((test_data.shape[0] - day_size * day_x), (test_data.shape[0] - day_size * day_y)),
                         test_data[(test_data.shape[0] - day_size * day_x):(test_data.shape[0] - day_size * day_y),
                         flow_x, flow_y], label='Actual')
                plt.plot(range((pred.shape[0] - day_size * day_x), (pred.shape[0] - day_size * day_y)),
                         pred[(pred.shape[0] - day_size * day_x):(pred.shape[0] - day_size * day_y), flow_x, flow_y],
                         label='Predicted')
                plt.xlabel('Timestep')
                plt.ylabel('Traffic Load')

                plt.legend()

                plt.savefig(plotted_path + 'Flow-{}-{}.png'.format(flow_x, flow_y))
                plt.close()

        # plt.plot(range(9005, 9020),
        #          test_data[9005:9020, 1,
        #          1], label='Actual')
        # plt.plot(range(9005,9020),
        #          pred[9005:9020, 1, 1],
        #          label='Predicted')
        # plt.xlabel('Timestep')
        # plt.ylabel('Traffic Load')
        #
        # plt.legend()
        #
        # plt.savefig(plotted_path + 'Flow-{}-{}.png'.format(1, 1))
        # plt.close()

        # test_data = test_data[(test_data.shape[0] - day_size * day_x):(test_data.shape[0] - day_size * day_y)]
        # pred = pred[(pred.shape[0] - day_size * day_x):(pred.shape[0] - day_size * day_y)]
        # measure_matrix = measure_matrix[
        #                  (measure_matrix.shape[0] - day_size * day_x):(measure_matrix.shape[0] - day_size * day_y)]

        print('|--- Error Ratio: {}'.format(error_ratio(y_true=test_data,
                                                        y_pred=pred,
                                                        measured_matrix=measure_matrix)))
        print('|--- RMSE: {}'.format(calculate_rmse(y_true=test_data,
                                                    y_pred=pred)))
        print('|--- R2: {}'.format(calculate_r2_score(y_true=test_data,
                                                      y_pred=pred)))
