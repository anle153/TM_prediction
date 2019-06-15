import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from Models.ConvLSTM_model import ConvLSTM
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_2d, create_offline_convlstm_data_fix_ratio, data_scalling
from common.error_utils import error_ratio, calculate_r2_score, \
    calculate_rmse, calculate_mape

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def plot_test_data(prefix, raw_data, pred, current_data):
    saving_path = Config.RESULTS_PATH + 'plot_check_conv-lstm/'

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    from matplotlib import pyplot as plt
    for flow_x in range(raw_data.shape[1]):
        for flow_y in range(raw_data.shape[2]):
            plt.plot(raw_data[:, flow_x, flow_y], label='Actual')
            plt.plot(pred[:, flow_x, flow_y], label='Pred')
            plt.plot(current_data[:, flow_x, flow_y, 0], label='Current_pred')

            plt.legend()
            plt.savefig(saving_path + '{}_flow_{:02d}-{:02d}.png'.format(prefix, flow_x, flow_y))
            plt.close()


def ims_tm_prediction(init_data_labels, conv_lstm_model):
    multi_steps_tm = np.zeros(shape=(init_data_labels.shape[0] + Config.CONV_LSTM_IMS_STEP,
                                     init_data_labels.shape[1], init_data_labels.shape[2], init_data_labels.shape[3]))

    multi_steps_tm[0:init_data_labels.shape[0], :, :, :] = init_data_labels

    for ts_ahead in range(Config.CONV_LSTM_IMS_STEP):
        rnn_input = multi_steps_tm[-Config.CONV_LSTM_STEP:, :, :, :]  # shape(timesteps, od, od , 2)

        rnn_input = np.expand_dims(rnn_input, axis=0)  # shape(1, timesteps, od, od , 2)

        predictX = conv_lstm_model.predict(rnn_input)  # shape(1, timesteps, od, od , 1)

        predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, od, od , 1)
        predictX = np.squeeze(predictX, axis=3)  # shape(timesteps, od, od)

        predict_tm = predictX[-1, :, :]

        sampling = np.zeros(shape=(Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH, 1))

        # Calculating the true value for the TM
        new_input = predict_tm

        # Concaternating the new tm to the final results
        # Shape = (12, 12, 2)
        new_input = np.concatenate([np.expand_dims(new_input, axis=2), sampling], axis=2)
        multi_steps_tm[ts_ahead + Config.CONV_LSTM_STEP] = new_input  # Shape = (timestep, 12, 12, 2)

    return multi_steps_tm[-1, :, :, 0]


def predict_conv_lstm(initial_data, test_data, conv_lstm_model):
    tf_a = np.array([1.0, 0.0])

    init_labels = np.ones(shape=initial_data.shape)

    tm_pred = np.zeros(
        shape=(initial_data.shape[0] + test_data.shape[0], initial_data.shape[1], initial_data.shape[2]))
    tm_pred[0:initial_data.shape[0]] = initial_data

    labels = np.zeros(
        shape=(init_labels.shape[0] + test_data.shape[0], init_labels.shape[1], init_labels.shape[2]))
    labels[0:init_labels.shape[0]] = init_labels

    ims_tm = np.zeros(
        shape=(test_data.shape[0] - Config.CONV_LSTM_IMS_STEP + 1, test_data.shape[1], test_data.shape[2]))
    raw_data = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1], test_data.shape[2]))

    raw_data[0:initial_data.shape[0]] = initial_data
    raw_data[initial_data.shape[0]:] = test_data

    for ts in tqdm(range(test_data.shape[0])):
        # if Config.CONV_LSTM_IMS and (ts <= test_data.shape[0] - Config.CONV_LSTM_IMS_STEP):
        #     ims_tm[ts] = ims_tm_prediction(init_data_labels=tm_labels[ts:ts + Config.CONV_LSTM_STEP, :, :, :],
        #                                    conv_lstm_model=conv_lstm_model)

        rnn_input = np.zeros(
            shape=(Config.CONV_LSTM_STEP, Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH, Config.CONV_LSTM_CHANNEL))

        rnn_input[:, :, :, 0] = tm_pred[ts:(ts + Config.CONV_LSTM_STEP)]
        rnn_input[:, :, :, 1] = labels[ts:(ts + Config.CONV_LSTM_STEP)]

        rnn_input = np.expand_dims(rnn_input, axis=0)

        predictX = conv_lstm_model.predict(rnn_input)  # shape(1, timesteps, #nflows)

        predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, #nflows)
        predict_tm = predictX[-1]

        predict_tm = np.reshape(predict_tm, newshape=(Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH))

        # Selecting next monitored flows randomly
        sampling = np.random.choice(tf_a, size=(Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH),
                                    p=(Config.CONV_LSTM_MON_RAIO, 1.0 - Config.CONV_LSTM_MON_RAIO))
        inv_sampling = 1.0 - sampling

        # Calculating the true value for the TM
        new_tm = predict_tm * inv_sampling + test_data[ts] * sampling

        # Concaternating the new tm to the final results
        tm_pred[ts + Config.CONV_LSTM_STEP] = new_tm
        labels[ts + Config.CONV_LSTM_STEP] = sampling

    return tm_pred[Config.CONV_LSTM_STEP:], labels[Config.CONV_LSTM_STEP:], ims_tm


def build_model(input_shape):
    print('|--- Build models.')

    conv_lstm_net = ConvLSTM(input_shape=input_shape,
                             cnn_layers=Config.CONV_LSTM_LAYERS,
                             a_filters=Config.CONV_LSTM_FILTERS,
                             a_strides=Config.CONV_LSTM_STRIDES,
                             dropouts=Config.CONV_LSTM_DROPOUTS,
                             kernel_sizes=Config.CONV_LSTM_KERNEL_SIZE,
                             rnn_dropouts=Config.CONV_LSTM_RNN_DROPOUTS,
                             alg_name=Config.ALG,
                             tag=Config.TAG,
                             check_point=True,
                             saving_path=Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(Config.DATA_NAME, Config.ALG,
                                                                                   Config.TAG,
                                                                                   Config.SCALER))
    print(conv_lstm_net.model.summary())
    conv_lstm_net.plot_models()
    return conv_lstm_net


def load_trained_models(input_shape, best_ckp):
    print('|--- Load trained model')
    conv_lstm_net = build_model(input_shape)
    conv_lstm_net.model.load_weights(conv_lstm_net.checkpoints_path + "weights-{:02d}.hdf5".format(best_ckp))

    return conv_lstm_net


def train_conv_lstm(data, experiment):
    print('|-- Run model training.')

    if Config.DATA_NAME == Config.DATA_SETS[0]:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')

    train_data_normalized2d, valid_data_normalized2d, _, scalers = data_scalling(train_data2d,
                                                                                 valid_data2d,
                                                                                 test_data2d)

    train_data_normalized = np.reshape(np.copy(train_data_normalized2d), newshape=(train_data_normalized2d.shape[0],
                                                                                   Config.CONV_LSTM_WIDE,
                                                                                   Config.CONV_LSTM_HIGH))
    valid_data_normalized = np.reshape(np.copy(valid_data_normalized2d), newshape=(valid_data_normalized2d.shape[0],
                                                                                   Config.CONV_LSTM_WIDE,
                                                                                   Config.CONV_LSTM_HIGH))

    input_shape = (Config.CONV_LSTM_STEP,
                   Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH, Config.CONV_LSTM_CHANNEL)

    with tf.device('/device:GPU:{}'.format(Config.GPU)):
        conv_lstm_net = build_model(input_shape)

    if not Config.CONV_LSTM_VALID_TEST or \
            not os.path.isfile(
                conv_lstm_net.checkpoints_path + 'weights-{:02d}.hdf5'.format(
                    Config.CONV_LSTM_BEST_CHECKPOINT)):

        print('|--- Compile model. Saving path %s --- ' % conv_lstm_net.saving_path)

        # -------------------------------- Create offline training and validating dataset --------------------------
        print('|--- Create offline train set for conv_lstm net!')

        trainX, trainY = create_offline_convlstm_data_fix_ratio(train_data_normalized,
                                                                input_shape, Config.CONV_LSTM_MON_RAIO,
                                                                train_data_normalized.std(), 3)
        print('|--- Create offline valid set for conv_lstm net!')

        validX, validY = create_offline_convlstm_data_fix_ratio(valid_data_normalized,
                                                                input_shape, Config.CONV_LSTM_MON_RAIO,
                                                                train_data_normalized.std(), 1)
        # ----------------------------------------------------------------------------------------------------------

        # Load model check point
        from_epoch = conv_lstm_net.load_model_from_check_point()
        if from_epoch > 0:
            print('|--- Continue training model from epoch %i --- ' % from_epoch)
            training_history = conv_lstm_net.model.fit(x=trainX,
                                                       y=trainY,
                                                       batch_size=Config.CONV_LSTM_BATCH_SIZE,
                                                       epochs=Config.CONV_LSTM_N_EPOCH,
                                                       callbacks=conv_lstm_net.callbacks_list,
                                                       validation_data=(validX, validY),
                                                       shuffle=True,
                                                       initial_epoch=from_epoch,
                                                       verbose=2)
        else:
            print('|--- Training new model.')
            training_history = conv_lstm_net.model.fit(x=trainX,
                                                       y=trainY,
                                                       batch_size=Config.CONV_LSTM_BATCH_SIZE,
                                                       epochs=Config.CONV_LSTM_N_EPOCH,
                                                       callbacks=conv_lstm_net.callbacks_list,
                                                       validation_data=(validX, validY),
                                                       shuffle=True,
                                                       verbose=2)

        # Plot the training history
        if training_history is not None:
            conv_lstm_net.plot_training_history(training_history)
    else:
        print('|--- Test valid set')
        conv_lstm_net.load_model_from_check_point(_from_epoch=Config.CONV_LSTM_BEST_CHECKPOINT)
    print(conv_lstm_net.model.summary())

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                      Config.ALG, Config.TAG, Config.SCALER)):
        os.makedirs(
            Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME, Config.ALG, Config.TAG, Config.SCALER))

    results_summary = pd.DataFrame(index=range(Config.CONV_LSTM_TESTING_TIME),
                                   columns=['No.', 'mape, ''err', 'r2', 'rmse', 'mape_ims', 'err_ims', 'r2_ims',
                                            'rmse_ims'])

    results_summary = run_test(valid_data2d, valid_data_normalized2d, conv_lstm_net, scalers, results_summary)

    results_summary.to_csv(Config.RESULTS_PATH +
                           '{}-{}-{}-{}/Valid_results.csv'.format(Config.DATA_NAME, Config.ALG, Config.TAG,
                                                                  Config.SCALER),
                           index=False)
    return


def ims_tm_test_data(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.LSTM_IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.LSTM_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.LSTM_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def test_conv_lstm(data, experiment):
    print('|-- Run model testing.')
    if Config.DATA_NAME == Config.DATA_SETS[0]:
        day_size = Config.ABILENE_DAY_SIZE
        assert Config.CONV_LSTM_WIDE == 12
        assert Config.CONV_LSTM_HIGH == 12
    else:
        day_size = Config.GEANT_DAY_SIZE
        assert Config.CONV_LSTM_WIDE == 23
        assert Config.CONV_LSTM_HIGH == 23

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')

    if Config.DATA_NAME == Config.DATA_SETS[0]:
        print('|--- Remove last 3 days in test data.')
        test_data2d = test_data2d[0:-day_size * 3]

    _, valid_data_normalized2d, test_data_normalized2d, scalers = data_scalling(train_data2d,
                                                                                valid_data2d,
                                                                                test_data2d)

    input_shape = (Config.CONV_LSTM_STEP,
                   Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH, Config.CONV_LSTM_CHANNEL)

    conv_lstm_net = load_trained_models(input_shape, Config.CONV_LSTM_BEST_CHECKPOINT)

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                      Config.ALG, Config.TAG, Config.SCALER)):
        os.makedirs(
            Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME, Config.ALG, Config.TAG, Config.SCALER))

    results_summary = pd.DataFrame(index=range(Config.CONV_LSTM_TESTING_TIME),
                                   columns=['No.', 'mape, ''err', 'r2', 'rmse', 'mape_ims', 'err_ims', 'r2_ims',
                                            'rmse_ims'])

    run_test(test_data2d, test_data_normalized2d, conv_lstm_net, scalers, results_summary)
    results_summary.to_csv(Config.RESULTS_PATH +
                           '{}-{}-{}-{}/Test_results.csv'.format(Config.DATA_NAME, Config.ALG, Config.TAG,
                                                                 Config.SCALER),
                           index=False)

    return


def prepare_test_set(test_data2d, test_data_normalized2d):
    if Config.DATA_NAME == Config.DATA_SETS[0]:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    idx = np.random.random_integers(Config.CONV_LSTM_STEP, test_data2d.shape[0] - day_size * 2 - 10)

    test_data_normalize = np.copy(test_data_normalized2d[idx:idx + day_size * 2])
    init_data_normalize = np.copy(test_data_normalized2d[idx - Config.CONV_LSTM_STEP: idx])
    test_data = test_data2d[idx:idx + day_size * 2]

    return test_data_normalize, init_data_normalize, test_data


def run_test(test_data2d, test_data_normalized2d, conv_lstm_net, scalers, results_summary):
    mape, err, r2_score, rmse = [], [], [], []
    mape_ims, err_ims, r2_score_ims, rmse_ims = [], [], [], []

    for i in range(Config.CONV_LSTM_TESTING_TIME):
        print('|--- Run time {}'.format(i))
        test_data_normalize, init_data_normalize, test_data = prepare_test_set(test_data2d, test_data_normalized2d)

        init_data_normalize = np.reshape(init_data_normalize, newshape=(init_data_normalize.shape[0],
                                                                        Config.CONV_LSTM_WIDE,
                                                                        Config.CONV_LSTM_HIGH))
        test_data_normalize = np.reshape(test_data_normalize, newshape=(test_data_normalize.shape[0],
                                                                        Config.CONV_LSTM_WIDE,
                                                                        Config.CONV_LSTM_HIGH))
        measured_matrix_ims2d = np.zeros((test_data.shape[0] - Config.CONV_LSTM_IMS_STEP + 1,
                                          Config.CONV_LSTM_WIDE * Config.CONV_LSTM_HIGH))

        pred_tm, measured_matrix, ims_tm = predict_conv_lstm(initial_data=init_data_normalize,
                                                             test_data=test_data_normalize,
                                                             conv_lstm_model=conv_lstm_net.model)

        pred_tm2d = np.reshape(np.copy(pred_tm), newshape=(pred_tm.shape[0], pred_tm.shape[1] * pred_tm.shape[2]))
        measured_matrix2d = np.reshape(np.copy(measured_matrix),
                                       newshape=(measured_matrix.shape[0],
                                                 measured_matrix.shape[1] * measured_matrix.shape[2]))

        pred_tm_invert2d = scalers.inverse_transform(pred_tm2d)
        mape.append(calculate_mape(y_true=test_data, y_pred=pred_tm_invert2d))

        err.append(error_ratio(y_true=test_data, y_pred=pred_tm_invert2d, measured_matrix=measured_matrix2d))
        r2_score.append(calculate_r2_score(y_true=test_data, y_pred=pred_tm_invert2d))
        rmse.append(calculate_rmse(y_true=test_data / 1000000, y_pred=pred_tm_invert2d / 1000000))

        if Config.CONV_LSTM_IMS:
            ims_tm2d = np.reshape(np.copy(ims_tm), newshape=(ims_tm.shape[0], ims_tm.shape[1] * ims_tm.shape[2]))
            ims_tm_invert2d = scalers.inverse_transform(ims_tm2d)
            ims_ytrue2d = ims_tm_test_data(test_data=test_data)

            mape_ims.append(calculate_mape(y_true=ims_ytrue2d, y_pred=ims_tm_invert2d))

            err_ims.append(error_ratio(y_pred=ims_tm_invert2d,
                                       y_true=ims_ytrue2d,
                                       measured_matrix=measured_matrix_ims2d))

            r2_score_ims.append(calculate_r2_score(y_true=ims_ytrue2d, y_pred=ims_tm_invert2d))
            rmse_ims.append(calculate_rmse(y_true=ims_ytrue2d / 1000000, y_pred=ims_tm_invert2d / 1000000))
        else:
            err_ims.append(0)
            r2_score_ims.append(0)
            rmse_ims.append(0)
            mape_ims.append(0)

        print('Result: mape\terr\trmse\tr2 \t\t mape_ims\terr_ims\trmse_ims\tr2_ims')
        print('        {}\t{}\t{}\t{} \t\t {}\t{}\t{}\t{}'.format(mape[i], err[i], rmse[i], r2_score[i],
                                                                  mape_ims[i], err_ims[i], rmse_ims[i],
                                                                  r2_score_ims[i]))

    results_summary['No.'] = range(Config.CONV_LSTM_TESTING_TIME)
    results_summary['mape'] = mape
    results_summary['err'] = err
    results_summary['r2'] = r2_score
    results_summary['rmse'] = rmse
    results_summary['mape_ims'] = mape_ims
    results_summary['err_ims'] = err_ims
    results_summary['r2_ims'] = r2_score_ims
    results_summary['rmse_ims'] = rmse_ims

    print('Test: {}-{}-{}-{}'.format(Config.DATA_NAME, Config.ALG, Config.TAG, Config.SCALER))

    print('avg_mape: {} - avg_err: {} - avg_rmse: {} - avg_r2: {}'.format(np.mean(np.array(mape)),
                                                                          np.mean(np.array(err)),
                                                                          np.mean(np.array(rmse)),
                                                                          np.mean(np.array(r2_score))))

    return results_summary
