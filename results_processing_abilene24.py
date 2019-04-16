from Utils.DataHelper import *
from Utils.DataPreprocessing import *

HIDDEN_DIM = 100


def get_results(result_path, pred_tm_file, measured_matrix_file, pred_multistep_file):
    pred_tm = np.load(file=result_path + pred_tm_file)
    measured_matrix = np.load(file=result_path + measured_matrix_file)
    pred_multistep_tm = np.load(file=result_path + pred_multistep_file)

    return pred_tm, measured_matrix, pred_multistep_tm


def get_results_path(dataset_name, test_name, sampling_ratio, n_timesteps):
    sampling_timesteps_path = 'Sampling_%.2f_timesteps_%i' % (sampling_ratio, n_timesteps)

    if test_name == 'arima':
        model_name = 'arima'
    else:
        model_name = 'hidden_%i_timesteps_%i_sampling_ratio_%.2f' % (
            HIDDEN_DIM, n_timesteps, sampling_ratio)

    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/' % \
                  (dataset_name, test_name, sampling_timesteps_path, model_name)

    return result_path


def get_multistep_tm_groundtruth(data, n_timesteps, prediction_steps, data_path):
    if not os.path.isfile(data_path + 'Abilene24_multistep_timesteps_%i_predictionsteps_%i.npy'
                          % (n_timesteps, prediction_steps)):

        iterated_multi_step_test_set = np.empty(shape=(0, prediction_steps, data.shape[1]))

        for ts in range(data.shape[0] - n_timesteps - prediction_steps):
            multi_step_test_set = np.copy(data[(ts + n_timesteps): (ts + n_timesteps + prediction_steps), :])
            multi_step_test_set = np.expand_dims(multi_step_test_set, axis=0)
            iterated_multi_step_test_set = np.concatenate([iterated_multi_step_test_set, multi_step_test_set], axis=0)

        np.save(file=data_path + 'Abilene24_multistep_timesteps_%i_predictionsteps_%i.npy'
                     % (n_timesteps, prediction_steps),
                arr=iterated_multi_step_test_set)
    else:
        iterated_multi_step_test_set = np.load(data_path + 'Abilene24_multistep_timesteps_%i_predictionsteps_%i.npy'
                                               % (n_timesteps, prediction_steps))

    return iterated_multi_step_test_set


def calculate_iterated_multi_step_tm_prediction_errors(test_set, n_timesteps,
                                                       prediction_steps):
    iterated_multi_step_test_set = np.empty(shape=(0, prediction_steps, test_set.shape[1]))

    for ts in range(test_set.shape[0] - n_timesteps - prediction_steps):
        multi_step_test_set = np.copy(test_set[(ts + n_timesteps): (ts + n_timesteps + prediction_steps), :])
        multi_step_test_set = np.expand_dims(multi_step_test_set, axis=0)
        iterated_multi_step_test_set = np.concatenate([iterated_multi_step_test_set, multi_step_test_set], axis=0)

    return iterated_multi_step_test_set


def calculate_error(n_timestep, result_path,
                    pred_tm_file, measured_matrix_file, pred_multistep_file, ground_truth_tm,
                    train_set):
    errors = []
    for i in range(10, 20):
        _pred_tm_file = pred_tm_file + '_%i.npy' % i
        _measured_matrix_file = measured_matrix_file + '_%i.npy' % i
        _pred_multistep_file = pred_multistep_file + '_%i.npy' % i

        pred_tm, measured_matrix, pred_multistep_tm = get_results(result_path=result_path,
                                                                  pred_tm_file=_pred_tm_file,
                                                                  measured_matrix_file=_measured_matrix_file,
                                                                  pred_multistep_file=_pred_multistep_file)

        er = error_ratio(y_true=ground_truth_tm, y_pred=pred_tm, measured_matrix=measured_matrix)
        r2_score = calculate_r2_score(y_true=ground_truth_tm, y_pred=pred_tm)
        rmse = rmse_tm_prediction(y_true=ground_truth_tm, y_pred=pred_tm)
        print('|--- ERROR RATIO: %.5f' % er)
        print('|--- RMSE: %.5f' % rmse)
        print('|--- R2: %.5f' % r2_score)

        ground_truth_tm_multistep = calculate_iterated_multi_step_tm_prediction_errors(
            test_set=np.copy(ground_truth_tm),
            n_timesteps=n_timesteps,
            prediction_steps=12)

        measured_matrix_ims = np.zeros(shape=ground_truth_tm_multistep.shape)

        er_ims = error_ratio(y_pred=pred_multistep_tm,
                             y_true=ground_truth_tm_multistep,
                             measured_matrix=measured_matrix_ims)
        r2_score_ims = calculate_r2_score(y_true=ground_truth_tm_multistep, y_pred=pred_multistep_tm)
        rmse_ims = rmse_tm_prediction(y_true=ground_truth_tm_multistep, y_pred=pred_multistep_tm)

        print('|--- ERROR RATIO IMS: %.5f' % er_ims)
        print('|--- RMSE IMS: %.5f' % rmse_ims)
        print('|--- R2 IMS: %.5f' % r2_score_ims)

        errors.append([er, r2_score, rmse, er_ims, r2_score_ims, rmse_ims])

    errors = np.array(errors)
    means = np.expand_dims(np.mean(errors, axis=0), axis=0)
    stds = np.expand_dims(np.std(errors, axis=0), axis=0)

    errors = np.concatenate([errors, means, stds], axis=0)

    np.savetxt(fname=result_path + '[NEW]errors.csv',
               X=errors,
               delimiter=',')


def plot_flow_by_day(n_timestep,
                     result_path,
                     pred_tm_file,
                     measured_matrix_file,
                     pred_multistep_file,
                     ground_truth_tm,
                     train_set):
    flowID_x = 142

    day_size = 288

    pred_tm_file = pred_tm_file + '_1.npy'
    measured_matrix_file = measured_matrix_file + '_1.npy'
    pred_multistep_file = pred_multistep_file + '_1.npy'
    pred_tm, measured_matrix, pred_multistep_tm = get_results(result_path=result_path,
                                                              pred_tm_file=pred_tm_file,
                                                              measured_matrix_file=measured_matrix_file,
                                                              pred_multistep_file=pred_multistep_file)
    ground_truth_tm = np.load(
        '/home/anle/TM_estimation_results/Abilene24/normal_rnn/Sampling_0.30_timesteps_26/hidden_100_timesteps_26_sampling_ratio_0.30/Ground_truth_tm_running_time_0.npy')

    nday = pred_tm.shape[0] / 288

    figure_path = result_path + 'plotting_flows/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    day = 10
    upperbound = (day + 1) * day_size if (day + 1) * day_size < ground_truth_tm.shape[0] else ground_truth_tm.shape[
        0]

    y1 = ground_truth_tm[day * day_size:upperbound, flowID_x]
    y2 = pred_tm[day * day_size:upperbound, flowID_x]

    sampling = measured_matrix[day * day_size:upperbound, flowID_x]
    arg_sampling = np.argwhere(sampling == True)

    plt.title('Flow %i prediction result' % (flowID_x))
    plt.plot(y1, label='Original Data')
    plt.plot(y2, label='Prediction Data')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Mbps')
    # Mark the measured data in the predicted data as red start
    plt.plot(arg_sampling, y2[arg_sampling], 'r*')
    plt.savefig(figure_path + 'Flow_%i_day_%i.png' % (flowID_x, day))
    plt.close()

    y1 = np.expand_dims(y1, axis=1)
    y2 = np.expand_dims(y2, axis=1)
    sampling = np.expand_dims(sampling, axis=1)
    flow = np.concatenate([y1, y2, sampling], axis=1)
    np.savetxt(figure_path + 'Flow_%i_day_%i.csv' % (flowID_x, day), flow, delimiter=',')

    measured_traffics = np.reshape(y2[arg_sampling], (y2[arg_sampling].shape[0], y2[arg_sampling].shape[1]))

    measured_points = np.concatenate([arg_sampling, measured_traffics], axis=1)
    np.savetxt(figure_path + 'flow_%i_day_%i_measured_points.csv' % (flowID_x, day), measured_points, delimiter=',')


if __name__ == '__main__':

    dataset_name = 'Abilene24'
    test_name = 'normal_rnn'
    n_timesteps = 26

    for sampling_ratio in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        result_path = get_results_path(dataset_name=dataset_name,
                                       test_name=test_name,
                                       sampling_ratio=sampling_ratio,
                                       n_timesteps=n_timesteps)

        pred_tm_file = 'Predicted_tm_running_time'
        measured_matrix_file = 'Predicted_measured_matrix_running_time'
        pred_multistep_file = 'Predicted_multistep_tm_running_time'

        # load raw data
        splitting_ratio = [0.8, 0.2]
        data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24.csv')
        train_set, test_set = prepare_train_test_set(data=data,
                                                     sampling_itvl=5,
                                                     splitting_ratio=splitting_ratio)

        test_set = np.copy(test_set[0:-864, :])

        # plot_flow_by_day(n_timestep=n_timesteps,
        #                 result_path=result_path,
        #                 pred_tm_file=pred_tm_file,
        #                 measured_matrix_file=measured_matrix_file,
        #                 pred_multistep_file=pred_multistep_file,
        #                 ground_truth_tm=np.copy(test_set),
        #                 train_set=train_set)
        calculate_error(n_timestep=n_timesteps,
                        result_path=result_path,
                        pred_tm_file=pred_tm_file,
                        measured_matrix_file=measured_matrix_file,
                        pred_multistep_file=pred_multistep_file,
                        ground_truth_tm=np.copy(test_set),
                        train_set=train_set)
