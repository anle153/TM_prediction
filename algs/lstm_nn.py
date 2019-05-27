import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from Models.RNN_LSTM import lstm
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_2d, create_offline_lstm_nn_data, data_scalling
from common.error_utils import error_ratio, calculate_r2_score, calculate_rmse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def plot_test_data(prefix, raw_data, pred, current_data):
    saving_path = Config.RESULTS_PATH + 'plot_check_lstm/'

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    from matplotlib import pyplot as plt
    for flow_x in range(raw_data.shape[1]):
        plt.plot(raw_data[:, flow_x], label='Actual')
        plt.plot(pred[:, flow_x], label='Pred_fw')
        plt.plot(current_data[:, flow_x], label='Current_pred')

        plt.legend()
        plt.savefig(saving_path + '{}_flow_{:02d}.png'.format(prefix, flow_x))
        plt.close()


def prepare_input_online_prediction(data, labels):
    labels = labels.astype(int)
    dataX = np.zeros(shape=(data.shape[1], Config.LSTM_STEP, 2))
    for flow_id in range(data.shape[1]):
        x = data[:, flow_id]
        label = labels[:, flow_id]

        sample = np.array([x, label]).T
        dataX[flow_id] = sample

    return dataX


def ims_tm_prediction(init_data, model, init_labels):
    multi_steps_tm = np.zeros(shape=(init_data.shape[0] + Config.LSTM_IMS_STEP, init_data.shape[1]))
    multi_steps_tm[0:Config.LSTM_STEP, :] = init_data

    labels = np.zeros(shape=(init_labels.shape[0] + Config.LSTM_IMS_STEP, init_labels.shape[1]))
    labels[0:Config.LSTM_STEP, :] = init_labels

    for ts_ahead in range(Config.LSTM_IMS_STEP):
        rnn_input = prepare_input_online_prediction(data=multi_steps_tm[ts_ahead:ts_ahead + Config.LSTM_STEP],
                                                    labels=labels[ts_ahead:ts_ahead + Config.LSTM_STEP])
        predictX = model.predict(rnn_input)
        multi_steps_tm[ts_ahead] = predictX[:, -1, 0].T

    return multi_steps_tm[-1, :]


def predict_lstm_nn(init_data, test_data, model):
    tf_a = np.array([1.0, 0.0])
    labels = np.zeros(shape=(init_data.shape[0] + test_data.shape[0], test_data.shape[1]))

    tm_pred = np.zeros(shape=(init_data.shape[0] + test_data.shape[0], test_data.shape[1]))

    ims_tm = np.zeros(shape=(test_data.shape[0] - Config.LSTM_IMS_STEP + 1, test_data.shape[1]))

    tm_pred[0:init_data.shape[0]] = init_data
    labels[0:init_data.shape[0]] = np.ones(shape=init_data.shape)

    raw_data = np.zeros(shape=(init_data.shape[0] + test_data.shape[0], test_data.shape[1]))

    raw_data[0:init_data.shape[0]] = init_data
    raw_data[init_data.shape[0]:] = test_data

    # Predict the TM from time slot look_back
    for ts in tqdm(range(test_data.shape[0])):
        # This block is used for iterated multi-step traffic matrices prediction

        if Config.LSTM_IMS and (ts <= test_data.shape[0] - Config.LSTM_IMS_STEP):
            ims_tm[ts] = ims_tm_prediction(init_data=tm_pred[ts:ts + Config.LSTM_STEP, :],
                                           model=model,
                                           init_labels=labels[ts:ts + Config.LSTM_STEP, :])

        # Create 3D input for rnn
        rnn_input = prepare_input_online_prediction(data=tm_pred[ts: ts + Config.LSTM_STEP],
                                                    labels=labels[ts: ts + Config.LSTM_STEP])

        # Get the TM prediction of next time slot
        predictX = model.predict(rnn_input)

        pred = predictX[:, -1, 0]

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.random.choice(tf_a, size=(test_data.shape[1]),
                                    p=[Config.LSTM_MON_RAIO, 1 - Config.LSTM_MON_RAIO])

        labels[ts + Config.LSTM_STEP] = sampling
        # invert of sampling: for choosing value from the original data
        inv_sampling = 1.0 - sampling
        pred_input = pred * inv_sampling

        ground_true = test_data[ts]

        measured_input = ground_true * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        tm_pred[ts + Config.LSTM_STEP] = new_input

    return tm_pred[Config.LSTM_STEP:, :], labels[Config.LSTM_STEP:, :], ims_tm


def build_model(input_shape):
    print('|--- Build models.')
    alg_name = Config.ALG
    tag = Config.TAG
    data_name = Config.DATA_NAME

    net = lstm(input_shape=input_shape,
               hidden=Config.LSTM_HIDDEN_UNIT,
               drop_out=Config.LSTM_DROPOUT,
               alg_name=alg_name, tag=tag, check_point=True,
               saving_path=Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER))

    if Config.LSTM_DEEP:
        net.seq2seq_deep_model_construction(n_layers=Config.LSTM_DEEP_NLAYERS)
    else:
        net.seq2seq_model_construction()

    return net


def train_lstm_nn(data, experiment):
    print('|-- Run model training.')
    gpu = Config.GPU

    params = Config.set_comet_params_lstm_nn()

    data_name = Config.DATA_NAME
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

    input_shape = (Config.LSTM_STEP, Config.LSTM_FEATURES)

    with tf.device('/device:GPU:{}'.format(gpu)):
        lstm_net = build_model(input_shape)

    if not Config.LSTM_VALID_TEST:
        if os.path.isfile(path=lstm_net.checkpoints_path + 'weights-{:02d}.hdf5'.format(Config.LSTM_N_EPOCH)):
            lstm_net.load_model_from_check_point(_from_epoch=Config.LSTM_BEST_CHECKPOINT)

        else:
            print('|---Compile model. Saving path {} --- '.format(lstm_net.saving_path))
            from_epoch = lstm_net.load_model_from_check_point()
            # -------------------------------- Create offline training and validating dataset --------------------------
            print('|--- Create offline train set for lstm-nn!')
            trainX, trainY = create_offline_lstm_nn_data(train_data_normalized2d, input_shape, Config.LSTM_MON_RAIO,
                                                         train_data_normalized2d.std())
            print('|--- Create offline valid set for lstm-nn!')
            validX, validY = create_offline_lstm_nn_data(valid_data_normalized2d, input_shape, Config.LSTM_MON_RAIO,
                                                         train_data_normalized2d.std())
            # ----------------------------------------------------------------------------------------------------------

            if from_epoch > 0:
                print('|--- Continue training.')
                training_history = lstm_net.model.fit(x=trainX,
                                                      y=trainY,
                                                      batch_size=Config.LSTM_BATCH_SIZE,
                                                      epochs=Config.LSTM_N_EPOCH,
                                                      callbacks=lstm_net.callbacks_list,
                                                      validation_data=(validX, validY),
                                                      shuffle=True,
                                                      initial_epoch=from_epoch,
                                                      verbose=2)
            else:
                print('|--- Training new model.')

                training_history = lstm_net.model.fit(x=trainX,
                                                      y=trainY,
                                                      batch_size=Config.LSTM_BATCH_SIZE,
                                                      epochs=Config.LSTM_N_EPOCH,
                                                      callbacks=lstm_net.callbacks_list,
                                                      validation_data=(validX, validY),
                                                      shuffle=True,
                                                      verbose=2)

            if training_history is not None:
                lstm_net.plot_training_history(training_history)
                experiment.log_parameters(params)
    else:
        lstm_net.model.load_weights(
            lstm_net.checkpoints_path + "weights-{:02d}.hdf5".format(Config.LSTM_BEST_CHECKPOINT))
    print('---------------------------------LSTM_NET SUMMARY---------------------------------')
    print(lstm_net.model.summary())

    run_test(experiment, valid_data2d, valid_data_normalized2d, train_data_normalized2d[-Config.LSTM_STEP:],
             lstm_net, params, scalers)

    return


def ims_tm_test_data(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.LSTM_IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.LSTM_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.LSTM_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def load_trained_model(input_shape, best_ckp):
    print('|--- Load trained model')
    lstm_net = build_model(input_shape)
    lstm_net.model.load_weights(lstm_net.checkpoints_path + "weights-{:02d}.hdf5".format(best_ckp))
    return lstm_net


def test_lstm_nn(data, experiment):
    print('|-- Run model testing.')
    gpu = Config.GPU

    params = Config.set_comet_params_lstm_nn()

    data_name = Config.DATA_NAME
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    if not Config.ALL_DATA:
        data = data[0:Config.NUM_DAYS * day_size]

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    if 'Abilene' in data_name:
        print('|--- Remove last 3 days in test data.')
        test_data2d = test_data2d[0:-day_size * 3]

    print('|--- Normalizing the train set.')
    _, valid_data_normalized2d, test_data_normalized2d, scalers = data_scalling(train_data2d,
                                                                                valid_data2d,
                                                                                test_data2d)

    print("|--- Create LSTM model.")
    input_shape = (Config.LSTM_STEP, Config.LSTM_FEATURES)

    with tf.device('/device:GPU:{}'.format(gpu)):

        lstm_net = load_trained_model(input_shape, Config.LSTM_BEST_CHECKPOINT)

    run_test(experiment, test_data2d, test_data_normalized2d, valid_data_normalized2d[-Config.LSTM_STEP:],
             lstm_net, params, scalers)
    return


def run_test(experiment, test_data2d, test_data_normalized2d, init_data2d, lstm_net, params, scalers):
    alg_name = Config.ALG
    tag = Config.TAG
    data_name = Config.DATA_NAME

    results_summary = pd.DataFrame(index=range(Config.LSTM_TESTING_TIME),
                                   columns=['No.', 'err', 'r2', 'rmse', 'err_ims', 'r2_ims', 'rmse_ims'])

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    ims_test_set = ims_tm_test_data(test_data=test_data2d)
    measured_matrix_ims = np.zeros(shape=ims_test_set.shape)

    # if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name)):
    #     np.save(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name),
    #             test_data2d)
    #
    # if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER)):
    #     np.save(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER),
    #             test_data_normalized2d)

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name,
                                                                      alg_name, tag, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER))

    for i in range(Config.LSTM_TESTING_TIME):
        print('|--- Running time: {}'.format(i))
        pred_tm2d, measured_matrix2d, ims_tm2d = predict_lstm_nn(init_data=init_data2d,
                                                                 test_data=test_data_normalized2d,
                                                                 model=lstm_net.model)

        # np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred_scaled-{}.npy'.format(data_name, alg_name, tag,
        #                                                                       Config.SCALER, i),
        #         pred_tm2d)

        pred_tm_invert2d = scalers.inverse_transform(pred_tm2d)

        err.append(error_ratio(y_true=test_data2d, y_pred=pred_tm_invert2d, measured_matrix=measured_matrix2d))
        r2_score.append(calculate_r2_score(y_true=test_data2d, y_pred=pred_tm_invert2d))
        rmse.append(calculate_rmse(y_true=test_data2d / 1000000, y_pred=pred_tm_invert2d / 1000000))

        if Config.LSTM_IMS:
            ims_tm_invert2d = scalers.inverse_transform(ims_tm2d)

            err_ims.append(error_ratio(y_pred=ims_tm_invert2d,
                                       y_true=ims_test_set,
                                       measured_matrix=measured_matrix_ims))

            r2_score_ims.append(calculate_r2_score(y_true=ims_test_set, y_pred=ims_tm_invert2d))
            rmse_ims.append(calculate_rmse(y_true=ims_test_set / 1000000, y_pred=ims_tm_invert2d / 1000000))

        else:
            err_ims.append(0)
            r2_score_ims.append(0)
            rmse_ims.append(0)

        print('Result: err\trmse\tr2 \t\t err_ims\trmse_ims\tr2_ims')
        print('        {}\t{}\t{} \t\t {}\t{}\t{}'.format(err[i], rmse[i], r2_score[i],
                                                          err_ims[i], rmse_ims[i],
                                                          r2_score_ims[i]))

        # np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred-{}.npy'.format(data_name, alg_name, tag,
        #                                                                Config.SCALER, i),
        #         pred_tm_invert2d)
        # np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/measure-{}.npy'.format(data_name, alg_name, tag,
        #                                                                   Config.SCALER, i),
        #         measured_matrix2d)

    results_summary['No.'] = range(Config.LSTM_TESTING_TIME)
    results_summary['err'] = err
    results_summary['r2'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['err_ims'] = err_ims
    results_summary['r2_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    results_summary.to_csv(Config.RESULTS_PATH + '{}-{}-{}-{}/results.csv'.format(data_name,
                                                                                  alg_name, tag, Config.SCALER),
                           index=False)
    print('avg_err: {} - avg_rmse: {} - avg_r2: {}'.format(np.mean(np.array(err)),
                                                           np.mean(np.array(rmse)),
                                                           np.mean(np.array(r2_score))))
