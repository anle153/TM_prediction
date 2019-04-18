import time

import pandas as pd
import tensorflow as tf

from Models.RNN_LSTM import lstm
from common.DataHelper import *
from common.DataPreprocessing import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


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
                                      iterated_multi_steps_tm,
                                      labels):
    multi_steps_tm = np.copy(ret_tm[-Config.LSTM_STEP:, :])

    measured_matrix = np.copy(labels)

    for ts_ahead in range(Config.IMS_STEP):
        rnn_input = prepare_input_online_prediction(data=multi_steps_tm,
                                                    n_timesteps=Config.LSTM_STEP,
                                                    labels=measured_matrix)
        predictX = rnn_model.predict(rnn_input)
        pred = np.expand_dims(predictX[:, -1, 0], axis=0)

        sampling = np.zeros(shape=(1, pred.shape[1]))
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)

        multi_steps_tm = np.concatenate([multi_steps_tm, pred], axis=0)

    multi_steps_tm = multi_steps_tm[Config.LSTM_STEP:, :]
    multi_steps_tm = np.expand_dims(multi_steps_tm, axis=0)

    iterated_multi_steps_tm = np.concatenate([iterated_multi_steps_tm, multi_steps_tm], axis=0)

    return iterated_multi_steps_tm


def predict_lstm_nn(test_data, model):

    # Initialize the first input for RNN to predict the TM at time slot look_back
    ret_tm = np.copy(test_data[0:Config.LSTM_STEP, :])
    # Results TM
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.ones(shape=(ret_tm.shape[0], ret_tm.shape[1]))

    iterated_multi_steps_tm = np.empty(shape=(0, Config.IMS_STEP, ret_tm.shape[1]))

    # Predict the TM from time slot look_back
    for ts in range(0, test_data.shape[0] - Config.LSTM_STEP, 1):
        # This block is used for iterated multi-step traffic matrices prediction

        if ts < test_data.shape[0] - Config.LSTM_STEP - Config.IMS_STEP:
            iterated_multi_steps_tm = iterated_multi_step_tm_prediction(ret_tm=ret_tm,
                                                                        rnn_model=model,
                                                                        iterated_multi_steps_tm=iterated_multi_steps_tm,
                                                                        labels=measured_matrix)

        # Create 3D input for rnn
        rnn_input = prepare_input_online_prediction(data=ret_tm, n_timesteps=Config.LSTM_STEP, labels=measured_matrix)

        # Get the TM prediction of next time slot
        predictX = model.predict(rnn_input)

        pred = np.expand_dims(predictX[:, -1, 0], axis=1)

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf,
                                                   size=(test_data.shape[1]),
                                                   p=[Config.MON_RAIO, 1 - Config.MON_RAIO]), axis=0)
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = pred.T * inv_sampling

        ground_true = np.copy(test_data[ts + Config.LSTM_STEP, :])

        measured_input = np.expand_dims(ground_true, axis=0) * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=0)

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


def build_model(args, input_shape):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    net = lstm(input_shape=input_shape,
               hidden=Config.LSTM_HIDDEN_UNIT,
               drop_out=Config.LSTM_DROPOUT,
               alg_name=alg_name, tag=tag, check_point=True,
               saving_path=Config.MODEL_SAVE + '{}-{}-{}/fw/'.format(data_name, alg_name, tag))

    return net


def train_lstm_nn(data, args):
    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_test_set(data=data)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data = (train_data - mean_train) / std_train
    valid_data = (valid_data - mean_train) / std_train
    test_data = (test_data - mean_train) / std_train

    input_shape = (Config.LSTM_STEP, Config.LSTM_FEATURES)

    lstm_net = build_model(args, input_shape)

    lstm_net.seq2seq_model_construction()

    if os.path.isfile(path=lstm_net.saving_path + 'model.json'):
        lstm_net.load_model_from_check_point(_from_epoch=Config.BEST_CHECKPOINT, weights_file_type='hdf5')

    else:
        print('|---Compile model. Saving path {} --- '.format(lstm_net.saving_path))
        from_epoch = lstm_net.load_model_from_check_point(weights_file_type='hdf5')

        if from_epoch > 0:

            training_history = lstm_net.model.fit_generator(
                generator_lstm_nn_train_data(data=train_data,
                                             input_shape=input_shape,
                                             mon_ratio=Config.MON_RAIO,
                                             eps=0.5,
                                             batch_size=Config.BATCH_SIZE),
                epochs=Config.N_EPOCH,
                steps_per_epoch=Config.NUM_ITER,
                initial_epoch=from_epoch,
                validation_data=generator_lstm_nn_train_data(valid_data, input_shape, Config.MON_RAIO, 0.5,
                                                             Config.BATCH_SIZE),
                validation_steps=int(Config.NUM_ITER * 0.2),
                callbacks=lstm_net.callbacks_list,
                use_multiprocessing=True, workers=2, max_queue_size=1024
            )
        else:

            training_history = lstm_net.model.fit_generator(
                generator_lstm_nn_train_data(data=train_data,
                                             input_shape=input_shape,
                                             mon_ratio=Config.MON_RAIO,
                                             eps=0.5,
                                             batch_size=Config.BATCH_SIZE),
                epochs=Config.N_EPOCH,
                steps_per_epoch=Config.NUM_ITER,
                validation_data=generator_lstm_nn_train_data(valid_data, input_shape, Config.MON_RAIO, 0.5,
                                                             Config.BATCH_SIZE),
                validation_steps=int(Config.NUM_ITER * 0.2),
                callbacks=lstm_net.callbacks_list,
                use_multiprocessing=True, workers=2, max_queue_size=1024
            )

        if training_history is not None:
            lstm_net.plot_training_history(training_history)
    print('---------------------------------LSTM_NET SUMMARY---------------------------------')
    print(lstm_net.model.summary())

    return


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


def test_fwbw_conv_lstm(data, args):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_test_set_3d(data=data)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data = (train_data - mean_train) / std_train
    valid_data = (valid_data - mean_train) / std_train
    test_data_normalized = (test_data - mean_train) / std_train

    print("|--- Create FWBW_CONVLSTM model.")
    input_shape = (Config.LSTM_STEP,
                   Config.CNN_WIDE, Config.CNN_HIGH, Config.CNN_CHANNEL)

    lstm_net = build_model(args, input_shape)

    results_summary = pd.read_csv(Config.RESULTS_PATH + '{}-{}-{}.csv'.format(data_name, alg_name, tag))

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    for i in range(Config.TESTING_TIME):
        tm_labels, iterated_multi_steps_tm = predict_lstm_nn(test_data=test_data_normalized,
                                                             model=lstm_net.model)

        pred_tm = tm_labels[:, :, :, 0]
        measured_matrix = tm_labels[:, :, :, 1]

        pred_tm = pred_tm * std_train + mean_train

        err.append(error_ratio(y_true=test_data_normalized, y_pred=np.copy(pred_tm), measured_matrix=measured_matrix))
        r2_score.append(calculate_r2_score(y_true=test_data_normalized, y_pred=np.copy(pred_tm)))
        rmse.append(rmse_tm_prediction(y_true=test_data_normalized, y_pred=np.copy(pred_tm)))

        iterated_multi_steps_tm = iterated_multi_steps_tm * std_train + mean_train

        iterated_multi_step_test_set = calculate_lstm_iterated_multi_step_tm_prediction_errors(test_set=test_data)

        measured_matrix = np.zeros(shape=iterated_multi_step_test_set.shape)
        err_ims.append(error_ratio(y_pred=iterated_multi_steps_tm,
                                   y_true=iterated_multi_step_test_set,
                                   measured_matrix=measured_matrix))

        r2_score_ims.append(calculate_r2_score(y_true=iterated_multi_step_test_set, y_pred=iterated_multi_steps_tm))
        rmse_ims.append(rmse_tm_prediction(y_true=iterated_multi_step_test_set, y_pred=iterated_multi_steps_tm))

    results_summary['running_time'] = range(Config.TESTING_TIME)
    results_summary['err'] = err
    results_summary['r2_score'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['err_ims'] = err_ims
    results_summary['r2_score_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    results_summary.to_csv(Config.RESULTS_PATH + '{}-{}-{}.csv'.format(data_name, alg_name, tag),
                           index=False)

    return
