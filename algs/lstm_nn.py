import os

import numpy as np
import pandas as pd
import tensorflow as tf

from Models.RNN_LSTM import lstm
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_2d, generator_lstm_nn_train_data
from common.error_utils import error_ratio, calculate_r2_score, calculate_rmse
from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def prepare_input_online_prediction(data, labels):
    labels = labels.astype(int)
    dataX = np.zeros(shape=(data.shape[1], Config.LSTM_STEP, 2))
    for flow_id in range(data.shape[1]):
        x = data[-Config.LSTM_STEP:, flow_id]
        label = labels[-Config.LSTM_STEP:, flow_id]

        sample = np.array([x, label]).T
        dataX[flow_id] = sample

    return dataX


def ims_tm_prediction(init_data, model, init_labels):
    multi_steps_tm = np.zeros(shape=(init_data.shape[0] + Config.IMS_STEP, init_data.shape[1]))
    multi_steps_tm[0:Config.LSTM_STEP, :] = init_data

    labels = np.zeros(shape=(init_labels.shape[0] + Config.IMS_STEP, init_labels.shape[1]))
    labels[0:Config.LSTM_STEP, :] = init_labels

    for ts_ahead in range(Config.IMS_STEP):
        rnn_input = prepare_input_online_prediction(data=multi_steps_tm,
                                                    labels=labels)
        predictX = model.predict(rnn_input)
        multi_steps_tm[ts_ahead] = predictX[:, -1, 0].T

    return multi_steps_tm[-1, :]


def predict_lstm_nn(init_data, test_data, model):

    tf = np.array([True, False])
    labels = np.ones(shape=init_data.shape)

    tm_pred = np.zeros(shape=(init_data.shape[0] + test_data.shape[0], test_data.shape[1]))

    ims_tm = np.zeros(shape=(test_data.shape[0] - Config.IMS_STEP + 1, test_data.shape[1]))

    # Predict the TM from time slot look_back
    for ts in tqdm(range(test_data.shape[0])):
        # This block is used for iterated multi-step traffic matrices prediction

        if ts <= test_data.shape[0] - Config.IMS_STEP:
            ims_tm[ts] = ims_tm_prediction(init_data=tm_pred[ts:ts + Config.LSTM_STEP:, :],
                                           model=model,
                                           init_labels=labels[ts:ts + Config.LSTM_STEP:, :])

        # Create 3D input for rnn
        rnn_input = prepare_input_online_prediction(data=tm_pred, labels=labels)

        # Get the TM prediction of next time slot
        predictX = model.predict(rnn_input)

        pred = np.expand_dims(predictX[:, -1, 0], axis=1)

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf,
                                                   size=(test_data.shape[1]),
                                                   p=[Config.MON_RAIO, 1 - Config.MON_RAIO]), axis=0)
        labels = np.concatenate([labels, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = pred.T * inv_sampling

        ground_true = np.copy(test_data[ts, :])

        measured_input = np.expand_dims(ground_true, axis=0) * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shap e[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        tm_pred[ts + Config.LSTM_STEP] = new_input

    return tm_pred[Config.LSTM_STEP:, :], labels[Config.LSTM_STEP:, :], ims_tm


def build_model(args, input_shape):
    print('|--- Build models.')
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    net = lstm(input_shape=input_shape,
               hidden=Config.LSTM_HIDDEN_UNIT,
               drop_out=Config.LSTM_DROPOUT,
               alg_name=alg_name, tag=tag, check_point=True,
               saving_path=Config.MODEL_SAVE + '{}-{}-{}/'.format(data_name, alg_name, tag))

    if 'deep' in alg_name:
        net.seq2seq_deep_model_construction(n_layers=3)
    else:
        net.seq2seq_model_construction()

    return net


def train_lstm_nn(data, args):
    print('|-- Run model training.')
    gpu = args.gpu

    if gpu is None:
        gpu = 0

    with tf.device('/device:GPU:{}'.format(gpu)):

        print('|--- Splitting train-test set.')
        train_data, valid_data, test_data = prepare_train_valid_test_2d(data=data)
        print('|--- Normalizing the train set.')
        mean_train = np.mean(train_data)
        std_train = np.std(train_data)
        train_data_normalized = (train_data - mean_train) / std_train
        valid_data_normalized = (valid_data - mean_train) / std_train
        # test_data_normalized = (test_data - mean_train) / std_train

        input_shape = (Config.LSTM_STEP, Config.LSTM_FEATURES)

        lstm_net = build_model(args, input_shape)

        if os.path.isfile(path=lstm_net.checkpoints_path + 'weights-{:02d}.hdf5'.format(Config.N_EPOCH)):
            lstm_net.load_model_from_check_point(_from_epoch=Config.BEST_CHECKPOINT, weights_file_type='hdf5')

        else:
            print('|---Compile model. Saving path {} --- '.format(lstm_net.saving_path))
            from_epoch = lstm_net.load_model_from_check_point(weights_file_type='hdf5')

            if from_epoch > 0:

                training_history = lstm_net.model.fit_generator(
                    generator_lstm_nn_train_data(data=train_data_normalized,
                                                 input_shape=input_shape,
                                                 mon_ratio=Config.MON_RAIO,
                                                 eps=0.5,
                                                 batch_size=Config.BATCH_SIZE),
                    epochs=Config.N_EPOCH,
                    steps_per_epoch=Config.NUM_ITER,
                    initial_epoch=from_epoch,
                    validation_data=generator_lstm_nn_train_data(valid_data_normalized,
                                                                 input_shape, Config.MON_RAIO, 0.5,
                                                                 Config.BATCH_SIZE),
                    validation_steps=int(Config.NUM_ITER * 0.2),
                    callbacks=lstm_net.callbacks_list,
                    use_multiprocessing=True, workers=4, max_queue_size=1024
                )
            else:

                training_history = lstm_net.model.fit_generator(
                    generator_lstm_nn_train_data(data=train_data_normalized,
                                                 input_shape=input_shape,
                                                 mon_ratio=Config.MON_RAIO,
                                                 eps=0.5,
                                                 batch_size=Config.BATCH_SIZE),
                    epochs=Config.N_EPOCH,
                    steps_per_epoch=Config.NUM_ITER,
                    validation_data=generator_lstm_nn_train_data(valid_data_normalized,
                                                                 input_shape, Config.MON_RAIO, 0.5,
                                                                 Config.BATCH_SIZE),
                    validation_steps=int(Config.NUM_ITER * 0.2),
                    callbacks=lstm_net.callbacks_list,
                    use_multiprocessing=True, workers=4, max_queue_size=1024
                )

            if training_history is not None:
                lstm_net.plot_training_history(training_history)
        print('---------------------------------LSTM_NET SUMMARY---------------------------------')
        print(lstm_net.model.summary())

    return


def calculate_iterated_multi_step_tm_prediction_errors(test_set):
    ims_test_set = np.empty(shape=(0, Config.IMS_STEP, test_set.shape[1]))

    for ts in range(test_set.shape[0] - Config.LSTM_STEP - Config.IMS_STEP):
        multi_step_test_set = np.copy(test_set[(ts + Config.LSTM_STEP): (ts + Config.LSTM_STEP + Config.IMS_STEP), :])
        multi_step_test_set = np.expand_dims(multi_step_test_set, axis=0)
        ims_test_set = np.concatenate([ims_test_set, multi_step_test_set], axis=0)

    return ims_test_set


def load_trained_model(args, input_shape, best_ckp):
    print('|--- Load trained model')
    lstm_net = build_model(args, input_shape)
    lstm_net.model.load_weights(lstm_net.checkpoints_path + "weights-{:02d}.hdf5".format(best_ckp))
    return lstm_net


def test_lstm_nn(data, args):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_valid_test_2d(data=data)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    # train_data_normalized = (train_data - mean_train) / std_train
    valid_data_normalized = (valid_data - mean_train) / std_train
    test_data_normalized = (test_data - mean_train) / std_train

    print("|--- Create FWBW_CONVLSTM model.")
    input_shape = (Config.LSTM_STEP, Config.LSTM_FEATURES)

    lstm_net = load_trained_model(args, input_shape, Config.LSTM_BEST_CHECKPOINT)

    results_summary = pd.read_csv(Config.RESULTS_PATH + 'sample_results.csv')

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    for i in range(Config.TESTING_TIME):
        print('|--- Running time: {}'.format(i))
        pred_tm, measured_matrix, ims_tm = predict_lstm_nn(init_data=valid_data_normalized[-Config.LSTM_STEP:, :],
                                                           test_data=test_data_normalized,
                                                           model=lstm_net.model)

        pred_tm = pred_tm * std_train + mean_train

        err.append(error_ratio(y_true=test_data_normalized, y_pred=np.copy(pred_tm), measured_matrix=measured_matrix))
        r2_score.append(calculate_r2_score(y_true=test_data_normalized, y_pred=np.copy(pred_tm)))
        rmse.append(calculate_rmse(y_true=test_data_normalized, y_pred=np.copy(pred_tm)))

        ims_tm = ims_tm * std_train + mean_train

        ims_test_set = calculate_iterated_multi_step_tm_prediction_errors(test_set=test_data)

        measured_matrix = np.zeros(shape=ims_test_set.shape)
        err_ims.append(error_ratio(y_pred=ims_tm,
                                   y_true=ims_test_set,
                                   measured_matrix=measured_matrix))

        r2_score_ims.append(calculate_r2_score(y_true=ims_test_set, y_pred=ims_tm))
        rmse_ims.append(calculate_rmse(y_true=ims_test_set, y_pred=ims_tm))

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
