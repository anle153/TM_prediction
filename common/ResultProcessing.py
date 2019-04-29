from matplotlib import pyplot as plt
from common import Config
import numpy as np
import os


def plot_pred_results(data_name, alg_name, tag, nflows, ndays):
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

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

    for i in range(run_time):

        pred = np.load(Config.RESULTS_PATH + '[pred-{}]{}-{}-{}-{}.npy'.format(i, data_name, alg_name, tag,
                                                                               Config.ADDED_RESULT_NAME))

        flows_x = np.random.random_integers(0, test_data.shape[1], size=nflows)
        flows_y = np.random.random_integers(0, test_data.shape[1], size=nflows)

        for j in range(nflows):
            x = flows_x[j]
            y = flows_y[j]
            plt.plot(range(day_size * ndays), test_data[-day_size * ndays:, x, y], label='Actual')
            plt.plot(range(day_size * ndays), pred[-day_size * ndays:, x, y], label='Predicted')
            plt.xlabel('Timestep')
            plt.ylabel('Traffic Load')

            plt.savefig(plotted_path + 'Flow-{}-{}.png'.format(x, y))
            plt.close()
