import numpy as np
from sklearn.metrics import r2_score

from Utils.DataHelper import *


def results_processing(results_path='../Results/Abilene24s/'):
    errors = np.empty((0, 6))

    sampling_ratio = 0.3
    look_back = 26

    arima_prefix = '[arima][sampling_rate_%.2f][look_back_%.2f]' % (sampling_ratio, look_back)
    arima_direct_prefix = '[arima_direct_prediction][sampling_rate_%.2f][look_back_%.2f]' % (sampling_ratio, look_back)
    rnn_prefix = '[forward_backward_rnn_labeled_features][sampling_rate_%.2f][look_back_%.2f]' % (
        sampling_ratio, look_back)
    rnn_cl_prefix = '[forward_backward_rnn_labeled_features][sampling_rate_%.2f][look_back_%.2f][Consecutive_Loss_4]' % (
        sampling_ratio, look_back)

    rnn_normal_cl_prefix = '[normal_rnn_consecutive_loss][sampling_rate_%.2f][look_back_%.2f]' % (
        sampling_ratio, look_back)

    arima_consecutive_loss_prefix = '[arima_consecutive_loss_4][sampling_rate_%.2f][look_back_%.2f]' % (
        sampling_ratio, look_back)

    rnn_normal = '[normal_rnn][sampling_rate_%.2f][look_back_%.2f]' % (
        sampling_ratio, look_back)

    measurement_file_name = 'MeasurementMatrix.csv'
    observation_file_name = 'Observation.csv'
    prediction_file_name = 'Prediction.csv'

    measured_matrix = np.genfromtxt(results_path + rnn_cl_prefix + measurement_file_name, delimiter=',')
    observation = np.genfromtxt(results_path + rnn_cl_prefix + observation_file_name, delimiter=',')
    prediction = np.genfromtxt(results_path + rnn_cl_prefix + prediction_file_name, delimiter=',')

    errors_by_day = calculate_error_ratio_by_day(y_true=observation, y_pred=prediction,
                                                 measured_matrix=measured_matrix,
                                                 sampling_itvl=5)
    mean_abs_error_by_day = mean_absolute_errors_by_day(y_true=observation, y_pred=prediction, sampling_itvl=5)

    rmse_by_day = root_means_squared_error_by_day(y_true=observation, y_pred=prediction, sampling_itvl=5)

    y3 = observation.flatten()
    y4 = prediction.flatten()
    a_nmse = normalized_mean_squared_error(y_true=y3, y_hat=y4)
    a_nmae = normalized_mean_absolute_error(y_true=y3, y_hat=y4)

    mae = mean_abs_error(y_true=observation, y_pred=prediction)
    rmse = np.sqrt(mean_squared_error(y_true=observation.flatten() / 1000.0, y_pred=prediction.flatten() / 1000.0))

    pred_confident = r2_score(y3, y4)

    err_rat = error_ratio(y_true=observation, y_pred=prediction, measured_matrix=measured_matrix)

    error = np.expand_dims(np.array(
        [a_nmae, a_nmse, pred_confident, err_rat, mae, rmse]), axis=0)

    errors = np.concatenate([errors, error], axis=0)

    np.savetxt(results_path + rnn_cl_prefix + 'Errors.csv', errors, delimiter=',')
    np.savetxt(results_path + rnn_cl_prefix + 'Errors_ratio_by_day.csv', errors_by_day, delimiter=',')
    np.savetxt(results_path + rnn_cl_prefix + 'Rmse_by_day.csv', rmse_by_day, delimiter=',')
    np.savetxt(results_path + rnn_cl_prefix + 'Mean_abs_error_by_day.csv', mean_abs_error_by_day, delimiter=',')

    return


def combine_result(results_path='../Results/Abilene24s/'):
    errors = np.empty((0, 6))

    sampling_ratio = 0.3
    look_back = 2

    arima_prefix = '[arima][sampling_rate_%.2f][look_back_%.2f]' % (sampling_ratio, look_back)
    arima_direct_prefix = '[arima_direct_prediction][sampling_rate_%.2f][look_back_%.2f]' % (sampling_ratio, look_back)
    rnn_prefix = '[forward_backward_rnn_labeled_features][sampling_rate_%.2f][look_back_%.2f]' % (
        sampling_ratio, look_back)
    measurement_file_name = 'MeasurementMatrix.csv'
    observation_file_name = 'Observation.csv'
    prediction_file_name = 'Prediction.csv'


def get_flow_by_day(results_path='../Results/Abilene24s/'):
    errors = np.empty((0, 6))

    sampling_ratio = 0.3
    look_back = 26

    arima_prefix = '[arima][sampling_rate_%.2f][look_back_%.2f]' % (sampling_ratio, look_back)
    arima_direct_prefix = '[arima_direct_prediction][sampling_rate_%.2f][look_back_%.2f]' % (sampling_ratio, look_back)
    rnn_prefix = '[forward_backward_rnn_labeled_features][sampling_rate_%.2f][look_back_%.2f]' % (sampling_ratio,
                                                                                                  look_back)
    rnn_normal_prefix = '[normal_rnn][sampling_rate_%.2f][look_back_%.2f]' % (
        sampling_ratio, look_back)

    rnn_normal_direct_predic_prefix = '[forward_backward_rnn_labeled_features][sampling_rate_%.2f][look_back_%.2f]' % (
        sampling_ratio, look_back)

    measurement_file_name = 'MeasurementMatrix.csv'
    observation_file_name = 'Observation.csv'
    prediction_file_name = 'Prediction.csv'

    measured_matrix = np.genfromtxt(results_path + rnn_prefix + measurement_file_name, delimiter=',')
    observation = np.genfromtxt(results_path + rnn_prefix + observation_file_name, delimiter=',')
    prediction = np.genfromtxt(results_path + rnn_prefix + prediction_file_name, delimiter=',')

    day_size = 24 * (60 / 5)

    for day in range(int(prediction.shape[0] / day_size)):
        if day == 2:
            path_flow_by_day = results_path + rnn_prefix + '/' + 'Day_%i/' % day
            if not os.path.exists(path_flow_by_day):
                os.makedirs(path_flow_by_day)

            for flowID in range(observation.shape[1]):
                upperbound = (day + 1) * day_size if (day + 1) * day_size < observation.shape[0] else observation.shape[
                    0]
                y1 = observation[day * day_size:upperbound, flowID]
                y2 = prediction[day * day_size:upperbound, flowID]
                sampling = measured_matrix[day * day_size:upperbound, flowID]
                arg_sampling = np.argwhere(sampling == True)

                # plt.title('Flow %i prediction result' % (flowID))
                # plt.plot(y1, label='Original Data')
                # plt.plot(y2, label='Prediction Data')
                # plt.legend()
                # plt.xlabel('Time')
                # plt.ylabel('Mbps')
                # # Mark the measured data in the predicted data as red start
                # plt.plot(arg_sampling, y2[arg_sampling], 'r*')
                # plt.savefig(path_flow_by_day + '%i.png' % flowID)
                # plt.close()

                y1 = np.expand_dims(y1, axis=1)
                y2 = np.expand_dims(y2, axis=1)
                sampling = np.expand_dims(sampling, axis=1)
                flow = np.concatenate([y1, y2, sampling], axis=1)
                np.savetxt(path_flow_by_day + 'flow_%i.csv' % flowID, flow, delimiter=',')

                measured_traffics = np.reshape(y2[arg_sampling], (y2[arg_sampling].shape[0], y2[arg_sampling].shape[1]))

                measured_points = np.concatenate([arg_sampling, measured_traffics], axis=1)
                np.savetxt(path_flow_by_day + 'flow_%i_measured_points.csv' % flowID, measured_points, delimiter=',')


def get_flow_by_day_direct_predict(results_path='../Results/Abilene24s/'):
    sampling_ratio = 0.3
    look_back = 26

    rnn_normal_direct_predic_prefix = '[normal_rnn_direct_prediction][sampling_rate_%.2f][look_back_%.2f]' % (
        sampling_ratio, look_back)

    observation_file_name = 'Observation.csv'
    prediction_file_name = 'Prediction.csv'

    observation = np.genfromtxt(results_path + rnn_normal_direct_predic_prefix + observation_file_name, delimiter=',')
    prediction = np.genfromtxt(results_path + rnn_normal_direct_predic_prefix + prediction_file_name, delimiter=',')

    day_size = 24 * (60 / 5)

    for day in range(int(prediction.shape[0] / day_size)):
        if day == 2:
            path_flow_by_day = results_path + rnn_normal_direct_predic_prefix + '/' + 'Day_%i/' % day
            if not os.path.exists(path_flow_by_day):
                os.makedirs(path_flow_by_day)

            for flowID in range(observation.shape[1]):
                upperbound = (day + 1) * day_size if (day + 1) * day_size < observation.shape[0] else observation.shape[
                    0]
                y1 = observation[day * day_size:upperbound, flowID]
                y2 = prediction[day * day_size:upperbound, flowID]

                y1 = np.expand_dims(y1, axis=1)
                y2 = np.expand_dims(y2, axis=1)
                flow = np.concatenate([y1, y2], axis=1)
                np.savetxt(path_flow_by_day + 'flow_%i.csv' % flowID, flow, delimiter=',')


if __name__ == '__main__':
    results_path = '../Results/'
    dataset_name = 'Abilene24s/'
    results_processing(results_path=results_path + dataset_name)
