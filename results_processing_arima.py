from Utils.DataPreprocessing import *

from common.DataHelper import *

HIDDEN_DIM = 100


def get_results(result_path, pred_tm_file, measured_matrix_file, pred_multistep_file):
    pred_tm = np.load(file=result_path + pred_tm_file)
    measured_matrix = np.load(file=result_path + measured_matrix_file)
    pred_multistep_tm = np.load(file=result_path + pred_multistep_file)

    return pred_tm, measured_matrix, pred_multistep_tm


def get_results_path(dataset_name, test_name, sampling_ratio, n_timesteps):
    sampling_timesteps_path = 'Sampling_%.2f_timesteps_%i' % (sampling_ratio, n_timesteps)

    model_name = 'arima'

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


def calculate_error(test_name, n_timestep, result_path,
                    pred_tm_file, measured_matrix_file, pred_multistep_file, ground_truth_tm,
                    train_set):
    errors = []
    for i in range(0, 1):
        # _pred_tm_file = pred_tm_file + '_ly.npy'
        # _measured_matrix_file = measured_matrix_file + '_ly.npy'
        # _pred_multistep_file = pred_multistep_file + '_ly.npy'

        _pred_tm_file = pred_tm_file + '_%i.npy' % i
        _measured_matrix_file = measured_matrix_file + '_%i.npy' % i
        _pred_multistep_file = pred_multistep_file + '_%i.npy' % i

        # _pred_tm_file = '[24s]' + pred_tm_file + '_%i.npy' % i
        # _measured_matrix_file = '[24s]' + measured_matrix_file + '_%i.npy' % i
        # _pred_multistep_file ='[24s]' +  pred_multistep_file + '_%i.npy' % i

        pred_tm, measured_matrix, pred_multistep_tm = get_results(result_path=result_path,
                                                                  pred_tm_file=_pred_tm_file,
                                                                  measured_matrix_file=_measured_matrix_file,
                                                                  pred_multistep_file=_pred_multistep_file)

        # if ground_truth_tm.shape[0] > pred_tm.shape[0]:
        #     ground_truth_tm = ground_truth_tm[0:pred_tm.shape[0], :]

        er = error_ratio(y_true=ground_truth_tm, y_pred=pred_tm, measured_matrix=measured_matrix)
        r2_score = calculate_r2_score(y_true=ground_truth_tm, y_pred=pred_tm)
        rmse = rmse_tm_prediction(y_true=ground_truth_tm, y_pred=pred_tm)
        print('ERROR RATIO: %.5f' % er)
        print('RMSE: %.5f' % rmse)
        print('R2: %.5f' % r2_score)

        ground_truth_tm_multistep = get_multistep_tm_groundtruth(
            data=ground_truth_tm, prediction_steps=12, n_timesteps=n_timestep,
            data_path=HOME + '/TM_estimation_dataset/Abilene24/')
        # ground_truth_tm_multistep = np.load(
        #     '/home/anle/TM_estimation_dataset/Abilene24/normal_rnn/Ground_truth_multistep_prediciton_12.npy')

        measured_matrix_ims = np.zeros(shape=ground_truth_tm_multistep.shape)

        pred_multistep_tm = pred_multistep_tm[0:ground_truth_tm_multistep.shape[0], :, :]

        er_ims = error_ratio(y_pred=pred_multistep_tm,
                             y_true=ground_truth_tm_multistep,
                             measured_matrix=measured_matrix_ims)
        r2_score_ims = calculate_r2_score(y_true=ground_truth_tm_multistep, y_pred=pred_multistep_tm)
        rmse_ims = rmse_tm_prediction(y_true=ground_truth_tm_multistep, y_pred=pred_multistep_tm)

        print('ERROR RATIO IMS: %.5f' % er_ims)
        print('RMSE IMS: %.5f' % rmse_ims)
        print('R2 IMS: %.5f' % r2_score_ims)

        errors.append([er, r2_score, rmse, er_ims, r2_score_ims, rmse_ims])

    errors = np.array(errors)
    means = np.expand_dims(np.mean(errors, axis=0), axis=0)
    stds = np.expand_dims(np.std(errors, axis=0), axis=0)

    errors = np.concatenate([errors, means, stds], axis=0)

    np.savetxt(fname=result_path + 'errors.csv',
               X=errors,
               delimiter=',')


def plot_flow_by_day(test_name, n_timestep, result_path,
                     pred_tm_file, measured_matrix_file, pred_multistep_file, ground_truth_tm,
                     train_set):
    flowID_x = 11
    flowID_y = 10

    day_size = 288

    pred_tm_file = pred_tm_file + '_0.npy'
    measured_matrix_file = measured_matrix_file + '_0.npy'
    pred_multistep_file = pred_multistep_file + '_0.npy'
    pred_tm, measured_matrix, pred_multistep_tm = get_results(result_path=result_path,
                                                              pred_tm_file=pred_tm_file,
                                                              measured_matrix_file=measured_matrix_file,
                                                              pred_multistep_file=pred_multistep_file)
    nday = pred_tm.shape[0] / 288

    figure_path = result_path + 'plotting_flows/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # for flowID_x in range(pred_tm.shape[1]):
    #     for flowID_y in range(pred_tm.shape[2]):

    day = 10
    upperbound = (day + 1) * day_size if (day + 1) * day_size < ground_truth_tm.shape[0] else ground_truth_tm.shape[
        0]

    y1 = ground_truth_tm[day * day_size:upperbound, flowID_x, flowID_y]
    y2 = pred_tm[day * day_size:upperbound, flowID_x, flowID_y]

    y1 = y1 * 1000
    y2 = y2 * 1000

    sampling = measured_matrix[day * day_size:upperbound, flowID_x, flowID_y]
    arg_sampling = np.argwhere(sampling == True)

    plt.title('Flow %i_%i prediction result' % (flowID_x, flowID_y))
    plt.plot(y1, label='Original Data')
    plt.plot(y2, label='Prediction Data')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Mbps')
    # Mark the measured data in the predicted data as red start
    plt.plot(arg_sampling, y2[arg_sampling], 'r*')
    plt.savefig(figure_path + 'Flow_%i_%i_day_%i.png' % (flowID_x, flowID_y, day))
    plt.close()

    y1 = np.expand_dims(y1, axis=1)
    y2 = np.expand_dims(y2, axis=1)
    sampling = np.expand_dims(sampling, axis=1)
    flow = np.concatenate([y1, y2, sampling], axis=1)
    np.savetxt(figure_path + 'Flow_%i_%i_day_%i.csv' % (flowID_x, flowID_y, day), flow, delimiter=',')

    measured_traffics = np.reshape(y2[arg_sampling], (y2[arg_sampling].shape[0], y2[arg_sampling].shape[1]))

    measured_points = np.concatenate([arg_sampling, measured_traffics], axis=1)
    np.savetxt(figure_path + 'flow_%i_%i_day_%i_measured_points.csv' % (flowID_x, flowID_y, day), measured_points,
               delimiter=',')


if __name__ == '__main__':
    dataset_name = 'Abilene24'
    test_name = 'arima'
    sampling_ratio = 0.20
    n_timesteps = 28

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

    total_times = data.shape[0]
    data = data[0:total_times / 2, :]

    train_set, test_set = prepare_train_test_set(data=data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    test_set = np.copy(test_set[0:-864, :])

    calculate_error(test_name=test_name,
                    n_timestep=n_timesteps,
                    result_path=result_path,
                    pred_tm_file=pred_tm_file,
                    measured_matrix_file=measured_matrix_file,
                    pred_multistep_file=pred_multistep_file,
                    ground_truth_tm=test_set,
                    train_set=train_set)
    # plot_flow_by_day(test_name=test_name,
    #                 n_timestep=n_timesteps,
    #                 result_path=result_path,
    #                 pred_tm_file=pred_tm_file,
    #                 measured_matrix_file=measured_matrix_file,
    #                 pred_multistep_file=pred_multistep_file,
    #                 ground_truth_tm=test_set,
    #                 train_set=train_set)
