import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import time
from RNN import *
from Utils.DataHelper import *
from Utils.DataPreprocessing import *
from multiprocessing import cpu_count

# PATH CONFIGURATION
FIGURE_DIR = './figures/'
MODEL_RECORDED = './Model_Recorded/'

# DATASET CONFIGURATION
DATASET = ['Geant', 'Geant_noise_removed', 'Abilene', 'Abilene_noise_removed']
GEANT = 0
GEANT_NOISE_REMOVED = 1
ABILENE = 2
ABILENE_NOISE_REMOVE = 3

HOME = os.path.expanduser('~')

# Abilene dataset path.
""" The Abilene dataset contains 
    
    X           a 2016x2016 matrix of flow volumes
    A           a 30x121 matrix of routing of the 121 flows over 30 edge between adjacent nodes in Abilene network.
    odnames     a 121x1 character vector of OD pair names
    edgenames   a 30x1 character vector of node pairs sharing an edge 
"""
ABILENE_DATASET_PATH = './Dataset/SAND_TM_Estimation_Data.mat'

# Geant dataset path.
""" The Geant dataset contains 

    X: a (10772 x 529) matrix of flow volumes
"""
DATAPATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices'
GEANT_DATASET_PATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices/Geant_dataset.csv'
GEANT_DATASET_NOISE_REMOVED_PATH = DATAPATH + '/Gean_noise_removed.csv'
# RNN CONFIGURATION
INPUT_DIM = 100
HIDDEN_DIM = 100
LOOK_BACK = 26
N_EPOCH = 50
BATCH_SIZE = 256

# TRAINING MODES
SPLIT_TRAINING_SET_MODE = 0
HIDDEN_DIM_MODE = 1
BATCH_SIZE_MODE = 2
LOOK_BACK_MODE = 3

TRAINING_MODES = ['Split_dataset', 'hidden_layers', 'batch_size', 'lookback']

# Scaler mode
STANDARDSCALER = 0
MINMAX = 1
NORMALIZER = 2
DATANORMALIZER = 3
SCALER = ['StandardScaler', 'MinMax', 'SKNormalizer', 'MeanStdScale']


def prepare_input_online_prediction(data, n_timesteps, labels):
    labels = labels.astype(int)
    dataX = np.empty(shape=(0, n_timesteps, 2))
    for flow_id in range(data.shape[1]):
        x = data[-n_timesteps:, flow_id]
        label = labels[-n_timesteps:, flow_id]

        sample = np.array([x, label]).T
        sample = np.expand_dims(sample, axis=0)
        dataX = np.concatenate([dataX, sample], axis=0)

    return dataX


def iterated_multi_step_tm_prediction(ret_tm, rnn_model,
                                      n_timesteps,
                                      prediction_steps,
                                      iterated_multi_steps_tm,
                                      labels):
    multi_steps_tm = np.copy(ret_tm[-n_timesteps:, :])

    measured_matrix = np.copy(labels)

    for ts_ahead in range(prediction_steps):
        rnn_input = prepare_input_online_prediction(data=multi_steps_tm,
                                                    n_timesteps=n_timesteps,
                                                    labels=measured_matrix)
        predictX = rnn_model.predict(rnn_input)
        pred = np.expand_dims(predictX[:, -1, 0], axis=0)

        sampling = np.zeros(shape=(1, pred.shape[1]))
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)

        multi_steps_tm = np.concatenate([multi_steps_tm, pred], axis=0)

    multi_steps_tm = multi_steps_tm[n_timesteps:, :]
    multi_steps_tm = np.expand_dims(multi_steps_tm, axis=0)

    iterated_multi_steps_tm = np.concatenate([iterated_multi_steps_tm, multi_steps_tm], axis=0)

    return iterated_multi_steps_tm


def predict_normal_rnn(test_set, n_timesteps, model, sampling_ratio, prediction_steps):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)
    n_days = int(test_set.shape[0] / day_size) if (test_set.shape[0] % day_size) == 0 \
        else int(test_set.shape[0] / day_size) + 1

    # Initialize the first input for RNN to predict the TM at time slot look_back
    ret_tm = np.copy(test_set[0:n_timesteps, :])
    # Results TM
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.ones(shape=(ret_tm.shape[0], ret_tm.shape[1]))

    iterated_multi_steps_tm = np.empty(shape=(0, prediction_steps, ret_tm.shape[1]))

    # Predict the TM from time slot look_back
    for ts in range(0, test_set.shape[0] - n_timesteps, 1):
        # This block is used for iterated multi-step traffic matrices prediction

        # print('|--- Timesteps %d' % ts)

        if ts < test_set.shape[0] - n_timesteps - prediction_steps:
            iterated_multi_steps_tm = iterated_multi_step_tm_prediction(ret_tm=ret_tm,
                                                                        rnn_model=model,
                                                                        prediction_steps=prediction_steps,
                                                                        n_timesteps=n_timesteps,
                                                                        iterated_multi_steps_tm=iterated_multi_steps_tm,
                                                                        labels=measured_matrix)

        # Create 3D input for rnn
        rnn_input = prepare_input_online_prediction(data=ret_tm, n_timesteps=n_timesteps, labels=measured_matrix)

        # Get the TM prediction of next time slot
        predictX = model.predict(rnn_input)

        pred = np.expand_dims(predictX[:, -1, 0], axis=1)

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf,
                                                   size=(test_set.shape[1]),
                                                   p=[sampling_ratio, 1 - sampling_ratio]), axis=0)
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = pred.T * inv_sampling

        ground_true = np.copy(test_set[ts + n_timesteps, :])

        measured_input = np.expand_dims(ground_true, axis=0) * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=0)

        # Print error ratio each timestep prediction
        # print('|--- Timestep: %i, error ratio %.4f' %(ts+n_timesteps,
        #                                               error_ratio(y_true=test_set[ts+n_timesteps, :],
        #                                                           y_pred=ret_tm[ts+n_timesteps, :],
        #                                                           measured_matrix=measured_matrix[ts+n_timesteps, :])))

    return ret_tm, measured_matrix, iterated_multi_steps_tm


def predict_normal_rnn_no_multistep(test_set, n_timesteps, model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)
    # n_days = int(test_set.shape[0] / day_size) if (test_set.shape[0] % day_size) == 0 \
    #     else int(test_set.shape[0] / day_size) + 1

    # Initialize the first input for RNN to predict the TM at time slot look_back
    ret_tm = np.copy(test_set[0:n_timesteps, :])
    # Results TM
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.ones(shape=(ret_tm.shape[0], ret_tm.shape[1]))

    prediction_time = []

    # Predict the TM from time slot look_back
    for ts in range(0, test_set.shape[0] - n_timesteps, 1):
        # This block is used for iterated multi-step traffic matrices prediction

        # print('|--- Timesteps %d' % ts)
        start_prediction_time = time.time()

        # Create 3D input for rnn
        rnn_input = prepare_input_online_prediction(data=ret_tm, n_timesteps=n_timesteps, labels=measured_matrix)

        # Get the TM prediction of next time slot
        predictX = model.predict(rnn_input)

        pred = np.expand_dims(predictX[:, -1, 0], axis=1)

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf,
                                                   size=(test_set.shape[1]),
                                                   p=[sampling_ratio, 1 - sampling_ratio]), axis=0)
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = pred.T * inv_sampling

        ground_true = np.copy(test_set[ts + n_timesteps, :])

        measured_input = np.expand_dims(ground_true, axis=0) * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=0)

        # Print error ratio each timestep prediction
        # print('|--- Timestep: %i, error ratio %.4f' %(ts+n_timesteps,
        #                                               error_ratio(y_true=test_set[ts+n_timesteps, :],
        #                                                           y_pred=ret_tm[ts+n_timesteps, :],
        #                                                           measured_matrix=measured_matrix[ts+n_timesteps, :])))
        prediction_time.append(time.time() - start_prediction_time)
    prediction_time = np.array(prediction_time)
    np.savetxt('[LSTM]prediction_time_one_step.csv', prediction_time, delimiter=',')

    return ret_tm, measured_matrix


def traffic_prediction_consecutive_loss(test_set, look_back, rnn_model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)
    n_days = int(test_set.shape[0] / day_size) if (test_set.shape[0] % day_size) == 0 \
        else int(test_set.shape[0] / day_size) + 1

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :]
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - look_back, 1):
        date = int(tslot / day_size)

        # Create 3D input for rnn
        rnn_input = prepare_input_online_predict(pred_tm=ret_tm, look_back=look_back)
        rnn_input = np.reshape(rnn_input, (rnn_input.shape[0], rnn_input.shape[1], 1))

        # Get the TM prediction of next time slot
        predictX = rnn_model.predict(rnn_input)

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)
        # boolean array(1 x n_flows):for choosing value from predicted data

        # For Consecutive loss
        if (tslot > (date * day_size + 120)) and (tslot <= (date * day_size + 120 + 12 * 4)):
            sampling = np.zeros(shape=(1, measured_matrix.shape[1]), dtype=bool)
            print('Consecutive loss')
        else:
            sampling = np.expand_dims(np.random.choice(tf,
                                                       size=(test_set.shape[1]),
                                                       p=[sampling_ratio, 1 - sampling_ratio]), axis=0)

        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = predictX.T * inv_sampling
        measured_input = test_set[tslot + look_back, :] * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=0)

    return ret_tm, measured_matrix


def normal_rnn(raw_data, dataset_name='Abilene24s', n_timesteps=26):
    test_name = 'normal_rnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'
    nprocs = cpu_count()
    random_eps = 1
    sampling_ratio = 0.40

    if not os.path.isfile(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                    sampling_ratio, n_timesteps) + dataset_name + '_trainX.npy'):
        if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps)):
            os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps))

        print('|--- Splitting train-test set')
        train_set, test_set = prepare_train_test_set(data=raw_data,
                                                     sampling_itvl=5,
                                                     splitting_ratio=splitting_ratio)

        print("|--- Create XY set.")
        mean_train = np.mean(train_set)
        std_train = np.std(train_set)

        training_set = (train_set - mean_train) / std_train

        trainX, trainY = parallel_create_xy_set_by_random(training_set, n_timesteps, sampling_ratio, random_eps,
                                                          nprocs - 1)

        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps) + dataset_name + '_trainX.npy',
            trainX)
        np.save(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps) + dataset_name + '_trainY.npy',
            trainY)
    else:

        print(
            "|---  Load xy set from " + HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps))

        trainX = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps) + dataset_name + '_trainX.npy')
        trainY = np.load(
            HOME + '/TM_estimation_dataset/' + dataset_name + '/Sampling_%.2f_timesteps_%i/' % (
                sampling_ratio, n_timesteps) + dataset_name + '_trainY.npy')

    print("|--- Creating RNN model")

    rnn = RNN(
        saving_path=model_recorded_path + 'hidden_%i_timesteps_%i_sampling_ratio_%.2f/' % (
            HIDDEN_DIM, n_timesteps, sampling_ratio),
        raw_data=raw_data,
        look_back=n_timesteps,
        n_epoch=N_EPOCH,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        check_point=True)

    rnn.seq2seq_model_construction(n_timesteps=n_timesteps, n_features=2)

    if os.path.isfile(path=rnn.saving_path + 'model.json'):
        rnn.load_model_from_disk(model_json_file='model.json',
                                 model_weight_file='model.h5')

    else:
        print('[%s]---Compile model. Saving path %s --- ' % (test_name, rnn.saving_path))
        from_epoch = rnn.load_model_from_check_point(weights_file_type='hdf5')

        rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

        if from_epoch > 0:
            print('[%s]--- Continue training model from epoch %i --- ' % (test_name, from_epoch))

            training_history = rnn.model.fit(trainX,
                                             trainY,
                                             epochs=rnn.n_epoch,
                                             batch_size=rnn.batch_size,
                                             initial_epoch=from_epoch,
                                             validation_split=0.25,
                                             callbacks=rnn.callbacks_list)
            rnn.plot_model_metrics(training_history,
                                   plot_prefix_name='Metrics')
        else:

            training_history = rnn.model.fit(trainX,
                                             trainY,
                                             epochs=rnn.n_epoch,
                                             batch_size=rnn.batch_size,
                                             validation_split=0.25,
                                             callbacks=rnn.callbacks_list)

            rnn.plot_model_metrics(training_history,
                                   plot_prefix_name='Metrics')

    print(rnn.model.summary())

    return


def normal_rnn_test_loop(test_set, testing_set, rnn,
                         epoch, n_timesteps, sampling_ratio,
                         std_train, mean_train, n_running_time,
                         results_path):
    err_ratio_temp = []
    err_ratio_ims_temp = []

    for running_time in range(15, 20, 1):
        rnn.load_model_from_check_point(_from_epoch=epoch, weights_file_type='hdf5')
        rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

        _testing_set = np.copy(testing_set)
        _test_set = np.copy(test_set)
        print('|--- Epoch %i - Run time: %i' % (epoch, running_time))

        prediction_steps = 12

        pred_tm, measured_matrix, iterated_multi_step_pred_tm = predict_normal_rnn(test_set=_testing_set,
                                                                                   n_timesteps=n_timesteps,
                                                                                   model=rnn.model,
                                                                                   sampling_ratio=sampling_ratio,
                                                                                   prediction_steps=prediction_steps)

        pred_tm = pred_tm * std_train + mean_train
        measured_matrix = measured_matrix.astype(bool)

        err = error_ratio(y_true=_test_set, y_pred=pred_tm, measured_matrix=measured_matrix)
        r2_score = calculate_r2_score(y_true=_test_set, y_pred=pred_tm)
        rmse = rmse_tm_prediction(y_true=_test_set / 1000, y_pred=pred_tm / 1000)

        err_ratio_temp.append(err)
        print('|--- ERROR RATIO: %.5f' % err)
        print('|--- RMSE: %.5f' % rmse)
        print('|--- R2: %.5f' % r2_score)

        np.save(file=results_path + 'Predicted_tm_running_time_%d' % running_time,
                arr=pred_tm)
        np.save(file=results_path + 'Predicted_measured_matrix_running_time_%d' % running_time,
                arr=measured_matrix)
        np.save(file=results_path + 'Ground_truth_tm_running_time_%d' % running_time,
                arr=_test_set)

        iterated_multi_step_pred_tm = iterated_multi_step_pred_tm * std_train + mean_train

        iterated_multi_step_test_set = calculate_iterated_multi_step_tm_prediction_errors(
            iterated_multi_step_pred_tm=iterated_multi_step_pred_tm,
            test_set=_test_set,
            n_timesteps=n_timesteps,
            prediction_steps=prediction_steps)

        measured_matrix = np.zeros(shape=iterated_multi_step_test_set.shape)

        err_ims = error_ratio(y_pred=iterated_multi_step_pred_tm,
                              y_true=iterated_multi_step_test_set,
                              measured_matrix=measured_matrix)

        r2_score_ims = calculate_r2_score(y_true=iterated_multi_step_test_set, y_pred=iterated_multi_step_pred_tm)
        rmse_ims = rmse_tm_prediction(y_true=iterated_multi_step_test_set / 1000,
                                      y_pred=iterated_multi_step_pred_tm / 1000)

        err_ratio_ims_temp.append(err_ims)

        print('|--- ERROR RATIO: %.5f' % err_ims)
        print('|--- RMSE: %.5f' % rmse_ims)
        print('|--- R2: %.5f' % r2_score_ims)

        # Save tm prediction

        np.save(file=results_path + 'Predicted_multistep_tm_running_time_%d' % running_time,
                arr=iterated_multi_step_pred_tm)
        np.save(file=results_path + 'Ground_truth_multistep_tm_running_time_%d' % running_time,
                arr=iterated_multi_step_test_set)

    return err_ratio_temp, err_ratio_ims_temp


def normal_rnn_test_loop_no_multistep(test_set, testing_set, rnn,
                                      epoch, n_timesteps, sampling_ratio,
                                      std_train, mean_train, n_running_time,
                                      results_path):
    results_path = results_path + 'test/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    err_ratio_temp = []

    for running_time in range(n_running_time):
        rnn.load_model_from_check_point(_from_epoch=epoch, weights_file_type='hdf5')
        rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

        _testing_set = np.copy(testing_set)
        _test_set = np.copy(test_set)
        print('|--- Epoch %i - Run time: %i' % (epoch, running_time))

        pred_tm, measured_matrix = predict_normal_rnn_no_multistep(test_set=np.copy(testing_set),
                                                                   n_timesteps=n_timesteps,
                                                                   model=rnn.model,
                                                                   sampling_ratio=sampling_ratio)

        pred_tm = pred_tm * std_train + mean_train
        measured_matrix = measured_matrix.astype(bool)

        er = error_ratio(y_true=np.copy(test_set),
                         y_pred=np.copy(pred_tm),
                         measured_matrix=measured_matrix)

        err_ratio_temp.append(error_ratio(y_true=np.copy(test_set),
                                          y_pred=np.copy(pred_tm),
                                          measured_matrix=measured_matrix))
        r2_score = calculate_r2_score(y_true=np.copy(test_set), y_pred=np.copy(pred_tm))
        rmse = rmse_tm_prediction(y_true=np.copy(test_set), y_pred=np.copy(pred_tm))

        print('|--- er: %.3f --- rmse: %.3f --- r2: %.3f' % (er, rmse, r2_score))

        # Save tm prediction
        # np.save(file=results_path + 'Predicted_tm_running_time_%d' % running_time,
        #         arr=pred_tm)
        # np.save(file=results_path + 'Predicted_measured_matrix_running_time_%d' % running_time,
        #         arr=measured_matrix)
        # np.save(file=results_path + 'Ground_truth_tm_running_time_%d' % running_time,
        #         arr=_test_set)

    return err_ratio_temp


def calculate_iterated_multi_step_tm_prediction_errors(iterated_multi_step_pred_tm, test_set, n_timesteps,
                                                       prediction_steps):
    iterated_multi_step_test_set = np.empty(shape=(0, prediction_steps, test_set.shape[1]))

    for ts in range(test_set.shape[0] - n_timesteps - prediction_steps):
        multi_step_test_set = np.copy(test_set[(ts + n_timesteps): (ts + n_timesteps + prediction_steps), :])
        multi_step_test_set = np.expand_dims(multi_step_test_set, axis=0)
        iterated_multi_step_test_set = np.concatenate([iterated_multi_step_test_set, multi_step_test_set], axis=0)

    np.save(file='/home/anle/TM_estimation_dataset/Abilene24/normal_rnn/Ground_truth_multistep_prediciton_12',
            arr=iterated_multi_step_test_set)

    return iterated_multi_step_test_set


def normal_rnn_test(raw_data, dataset_name, n_timesteps=26, with_epoch=0, sampling_ratio=0.10):
    test_name = 'normal_rnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    new_test_set = np.copy(test_set[0:-864, :])

    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    testing_set = np.copy(new_test_set)
    testing_set = (testing_set - mean_train) / std_train

    copy_testing_set = np.copy(testing_set)
    copy_test_set = np.copy(new_test_set)

    model_name = 'hidden_%i_timesteps_%i_sampling_ratio_%.2f' % (
        HIDDEN_DIM, n_timesteps, sampling_ratio)

    sampling_timesteps = 'Sampling_%.2f_timesteps_%d' % (sampling_ratio, n_timesteps)

    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/' % (
        dataset_name, test_name, sampling_timesteps, model_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    rnn = RNN(saving_path=model_recorded_path + '%s/' % model_name,
              raw_data=raw_data,
              look_back=n_timesteps,
              n_epoch=N_EPOCH,
              batch_size=BATCH_SIZE,
              hidden_dim=HIDDEN_DIM,
              check_point=True)
    rnn.seq2seq_model_construction(n_timesteps=n_timesteps, n_features=2)

    if with_epoch != 0:
        n_running_time = 5

        err_ratio_temp, err_ratio_ims_temp = normal_rnn_test_loop(test_set=copy_test_set,
                                                                  testing_set=copy_testing_set,
                                                                  rnn=rnn,
                                                                  epoch=with_epoch,
                                                                  n_timesteps=n_timesteps,
                                                                  sampling_ratio=sampling_ratio,
                                                                  std_train=std_train,
                                                                  mean_train=mean_train,
                                                                  n_running_time=n_running_time,
                                                                  results_path=result_path)

        err_ratio_temp = np.array(err_ratio_temp)
        err_ratio_temp = np.reshape(err_ratio_temp, newshape=(n_running_time, 1))
        err_ratio = np.mean(err_ratio_temp)
        err_ratio_std = np.std(err_ratio_temp)

        print('Error_mean: %.5f - Error_std: %.5f' % (err_ratio, err_ratio_std))
        print('|-------------------------------------------------------')

        results = np.empty(shape=(n_running_time, 0))
        epochs = np.arange(0, n_running_time)
        epochs = np.reshape(epochs, newshape=(n_running_time, 1))
        results = np.concatenate([results, epochs], axis=1)
        results = np.concatenate([results, err_ratio_temp], axis=1)

        # Save results:
        print('|--- Results have been saved at %s' % (result_path + 'Epoch_%i_n_running_time_%i.csv' %
                                                      (with_epoch, n_running_time)))

        np.savetxt(fname=result_path + 'Epoch_%i_n_running_time_%i.csv' % (with_epoch, n_running_time),
                   X=results, delimiter=',')

        err_ratio_ims_temp = np.array(err_ratio_ims_temp)
        err_ratio_ims_temp = np.reshape(err_ratio_ims_temp, newshape=(n_running_time, 1))
        err_ratio_ims = np.mean(err_ratio_ims_temp)
        err_ratio_ims_std = np.std(err_ratio_ims_temp)

        print('Error_mean: %.5f - Error_std: %.5f' % (err_ratio_ims, err_ratio_ims_std))
        print('|-------------------------------------------------------')

        results_ims = np.empty(shape=(n_running_time, 0))
        results_ims = np.concatenate([results_ims, epochs], axis=1)
        results_ims = np.concatenate([results_ims, err_ratio_ims_temp], axis=1)

        # Save results:
        print('|--- Results have been saved at %s' % (result_path + '|--- [IMS]Epoch_%i_n_running_time_%i.csv' %
                                                      (with_epoch, n_running_time)))

        np.savetxt(fname=result_path + '[IMS]Epoch_%i_n_running_time_%i.csv' % (with_epoch, n_running_time),
                   X=results_ims, delimiter=',')


def normal_rnn_test_no_ims(raw_data, dataset_name, n_timesteps=26, with_epoch=0, sampling_ratio=0.10):
    test_name = 'normal_rnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    new_test_set = np.copy(test_set[0:-864, :])

    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    testing_set = np.copy(new_test_set)
    testing_set = (testing_set - mean_train) / std_train

    copy_testing_set = np.copy(testing_set)
    copy_test_set = np.copy(new_test_set)

    model_name = 'hidden_%i_timesteps_%i_sampling_ratio_%.2f' % (
        HIDDEN_DIM, n_timesteps, sampling_ratio)

    sampling_timesteps = 'Sampling_%.2f_timesteps_%d' % (sampling_ratio, n_timesteps)

    result_path = HOME + '/TM_estimation_results/%s/%s/%s/%s/' % (
        dataset_name, test_name, sampling_timesteps, model_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    rnn = RNN(saving_path=model_recorded_path + '%s/' % model_name,
              raw_data=raw_data,
              look_back=n_timesteps,
              n_epoch=N_EPOCH,
              batch_size=BATCH_SIZE,
              hidden_dim=HIDDEN_DIM,
              check_point=True)
    rnn.seq2seq_model_construction(n_timesteps=n_timesteps, n_features=2)

    if with_epoch != 0:
        n_running_time = 1

        err_ratio_temp = normal_rnn_test_loop_no_multistep(test_set=copy_test_set,
                                                           testing_set=copy_testing_set,
                                                           rnn=rnn,
                                                           epoch=with_epoch,
                                                           n_timesteps=n_timesteps,
                                                           sampling_ratio=sampling_ratio,
                                                           std_train=std_train,
                                                           mean_train=mean_train,
                                                           n_running_time=n_running_time,
                                                           results_path=result_path)

        # err_ratio_temp = np.array(err_ratio_temp)
        # err_ratio_temp = np.reshape(err_ratio_temp, newshape=(n_running_time, 1))
        # err_ratio = np.mean(err_ratio_temp)
        # err_ratio_std = np.std(err_ratio_temp)
        #
        # print('Error_mean: %.5f - Error_std: %.5f' % (err_ratio, err_ratio_std))
        # print('|-------------------------------------------------------')
        #
        # results = np.empty(shape=(n_running_time, 0))
        # epochs = np.arange(0, n_running_time)
        # epochs = np.reshape(epochs, newshape=(n_running_time, 1))
        # results = np.concatenate([results, epochs], axis=1)
        # results = np.concatenate([results, err_ratio_temp], axis=1)

        # Save results:
        # print('|--- Results have been saved at %s' % (result_path + 'Epoch_%i_n_running_time_%i.csv' %
        #                                               (with_epoch, n_running_time)))

        # np.savetxt(fname=result_path + 'Epoch_%i_n_running_time_%i.csv' % (with_epoch, n_running_time),
        #            X=results, delimiter=',')


if __name__ == "__main__":
    Abilene24 = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24.csv')

    sampling_ratio = 0.1

    with tf.device('/device:GPU:1'):
        # normal_rnn(raw_data=Abilene24, dataset_name='Abilene24', n_timesteps=26)
        normal_rnn_test_no_ims(raw_data=Abilene24, dataset_name='Abilene24', n_timesteps=26, with_epoch=50,
                               sampling_ratio=sampling_ratio)
