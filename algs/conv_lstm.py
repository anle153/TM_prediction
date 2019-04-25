import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from Models.ConvLSTM_model import ConvLSTM
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_3d, generator_convlstm_train_data, \
    generator_convlstm_train_data_fix_ratio, create_offline_convlstm_valid_set_fix_ratio
from common.error_utils import calculate_consecutive_loss_3d, recovery_loss_3d, error_ratio, calculate_r2_score, \
    calculate_rmse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def ims_tm_prediction(init_data_labels, conv_lstm_model):
    multi_steps_tm = np.zeros(shape=(init_data_labels.shape[0] + Config.IMS_STEP,
                                     init_data_labels.shape[1], init_data_labels.shape[2], init_data_labels.shape[3]))

    multi_steps_tm[0:init_data_labels.shape[0], :, :, :] = init_data_labels

    for ts_ahead in range(Config.IMS_STEP):
        rnn_input = multi_steps_tm[-Config.LSTM_STEP:, :, :, :]  # shape(timesteps, od, od , 2)

        rnn_input = np.expand_dims(rnn_input, axis=0)  # shape(1, timesteps, od, od , 2)

        predictX = conv_lstm_model.predict(rnn_input)  # shape(1, timesteps, od, od , 1)

        predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, od, od , 1)
        predictX = np.squeeze(predictX, axis=3)  # shape(timesteps, od, od)

        predict_tm = predictX[-1, :, :]

        sampling = np.zeros(shape=(Config.CNN_WIDE, Config.CNN_HIGH, 1))

        # Calculating the true value for the TM
        new_input = predict_tm

        # Concaternating the new tm to the final results
        # Shape = (12, 12, 2)
        new_input = np.concatenate([np.expand_dims(new_input, axis=2), sampling], axis=2)
        multi_steps_tm[ts_ahead + Config.LSTM_STEP] = new_input  # Shape = (timestep, 12, 12, 2)

    return multi_steps_tm[-1, :, :, 0]


def predict_conv_lstm(initial_data, test_data, conv_lstm_model):
    init_labels = np.ones((initial_data.shape[0], initial_data.shape[1], initial_data.shape[2]))

    tm_labels = np.zeros(
        shape=(initial_data.shape[0] + test_data.shape[0], initial_data.shape[1], initial_data.shape[2], 2))
    tm_labels[0:initial_data.shape[0], :, :, 0] = initial_data
    tm_labels[0:init_labels.shape[0], :, :, 1] = init_labels

    ims_tm = np.zeros(shape=(test_data.shape[0] - Config.IMS_STEP + 1, test_data.shape[1], test_data.shape[2]))

    for ts in tqdm(range(test_data.shape[0])):

        if ts <= test_data.shape[0] - Config.IMS_STEP:
            ims_tm[ts] = ims_tm_prediction(init_data_labels=tm_labels[ts:ts + Config.LSTM_STEP, :, :, :],
                                           conv_lstm_model=conv_lstm_model)

        rnn_input = tm_labels[ts:(ts + Config.LSTM_STEP), :, :, :]  # shape(timesteps, od, od , 2)

        rnn_input = np.expand_dims(rnn_input, axis=0)  # shape(1, timesteps, od, od , 2)

        predictX = conv_lstm_model.predict(rnn_input)  # shape(1, timesteps, od, od , 1)

        predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, od, od , 1)
        predictX = np.squeeze(predictX, axis=3)  # shape(timesteps, od, od)

        predict_tm = predictX[-1, :, :]

        # Selecting next monitored flows randomly
        sampling = np.random.choice(tf, size=(12, 12), p=(Config.MON_RAIO, 1 - Config.MON_RAIO))
        inv_sampling = np.invert(sampling)

        pred_tm = predict_tm * inv_sampling
        corrected_data = test_data[ts, :, :]
        ground_truth = corrected_data * sampling

        # Calculating the true value for the TM
        new_tm = pred_tm + ground_truth

        # Concaternating the new tm to the final results
        new_tm = np.concatenate([np.expand_dims(new_tm, axis=2), np.expand_dims(sampling, axis=2)], axis=2)
        tm_labels[ts + Config.LSTM_STEP] = new_tm  # Shape = (timestep, 12, 12, 2)

    return tm_labels[Config.LSTM_STEP:, :, :, :], ims_tm


def build_model(args, input_shape):
    print('|--- Build models.')
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    conv_lstm_net = ConvLSTM(input_shape=input_shape,
                             cnn_layers=Config.CNN_LAYERS,
                             a_filters=Config.FILTERS,
                             a_strides=Config.STRIDES,
                             dropouts=Config.DROPOUTS,
                             kernel_sizes=Config.KERNEL_SIZE,
                             rnn_dropouts=Config.RNN_DROPOUTS,
                             alg_name=alg_name,
                             tag=tag,
                             check_point=True,
                             saving_path=Config.MODEL_SAVE + '{}-{}-{}/'.format(data_name, alg_name, tag))

    return conv_lstm_net


def load_trained_models(args, input_shape, best_ckp):
    print('|--- Load trained model')
    conv_lstm_net = build_model(args, input_shape)
    conv_lstm_net.model.load_weights(conv_lstm_net.checkpoints_path + "weights-{:02d}.hdf5".format(best_ckp))

    return conv_lstm_net


def train_conv_lstm(data, args):
    print('|-- Run model training.')
    gpu = args.gpu

    data_name = args.data_name
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    with tf.device('/device:GPU:{}'.format(gpu)):

        print('|--- Splitting train-test set.')
        train_data, valid_data, test_data = prepare_train_valid_test_3d(data=data, day_size=day_size)
        print('|--- Normalizing the train set.')
        mean_train = np.mean(train_data)
        std_train = np.std(train_data)
        train_data_normalized = (train_data - mean_train) / std_train
        valid_data_normalized = (valid_data - mean_train) / std_train
        # test_data = (test_data - mean_train) / std_train

        input_shape = (Config.LSTM_STEP,
                       Config.CNN_WIDE, Config.CNN_HIGH, Config.CNN_CHANNEL)

        conv_lstm_net = build_model(args, input_shape)

        if not os.path.isfile(conv_lstm_net.saving_path + 'validX.npy'):
            validX, validY = create_offline_convlstm_valid_set_fix_ratio(valid_data_normalized,
                                                                         input_shape, Config.MON_RAIO, 0.5)
            np.save(conv_lstm_net.saving_path + 'validX.npy', validX)
            np.save(conv_lstm_net.saving_path + 'validY.npy', validY)
        else:
            validX = np.load(conv_lstm_net.saving_path + 'validX.npy')
            validY = np.load(conv_lstm_net.saving_path + 'validY.npy')

        if Config.MON_RAIO is not None:
            generator_train_data = generator_convlstm_train_data_fix_ratio
        else:
            generator_train_data = generator_convlstm_train_data

        if os.path.isfile(path=conv_lstm_net.checkpoints_path + 'weights-{:02d}-0.00.hdf5'.format(Config.N_EPOCH)):
            print('|--- Model exist!')
            conv_lstm_net.load_model_from_check_point(_from_epoch=Config.CONV_LSTM_BEST_CHECKPOINT)
        else:
            print('|--- Compile model. Saving path %s --- ' % conv_lstm_net.saving_path)

            # Load model check point
            from_epoch = conv_lstm_net.load_model_from_check_point()
            if from_epoch > 0:
                print('|--- Continue training model from epoch %i --- ' % from_epoch)
                training_history = conv_lstm_net.model.fit_generator(
                    generator_train_data(train_data_normalized,
                                         input_shape,
                                         Config.MON_RAIO,
                                         0.5,
                                         Config.BATCH_SIZE),
                    epochs=Config.N_EPOCH,
                    steps_per_epoch=Config.NUM_ITER,
                    initial_epoch=from_epoch,
                    validation_data=(validX, validY),
                    callbacks=conv_lstm_net.callbacks_list,
                    use_multiprocessing=True,
                    workers=2,
                    max_queue_size=1024)
            else:
                print('|--- Training new model.')

                training_history = conv_lstm_net.model.fit_generator(
                    generator_train_data(train_data_normalized,
                                         input_shape,
                                         Config.MON_RAIO,
                                         0.5,
                                         Config.BATCH_SIZE),
                    epochs=Config.N_EPOCH,
                    steps_per_epoch=Config.NUM_ITER,
                    validation_data=(validX, validY),
                    callbacks=conv_lstm_net.callbacks_list,
                    use_multiprocessing=True,
                    workers=2,
                    max_queue_size=1028)

            # Plot the training history
            if training_history is not None:
                conv_lstm_net.plot_training_history(training_history)

        print('---------------------------------CONV_LSTM_NET SUMMARY---------------------------------')
        print(conv_lstm_net.model.summary())

    return


def ims_tm_ytrue(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.IMS_STEP + 1, test_data.shape[1], test_data.shape[2]))

    for i in range(Config.IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.IMS_STEP + 1] = test_data[i]

    return ims_test_set


def test_conv_lstm(data, args):
    print('|-- Run model testing.')
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_valid_test_3d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    # train_data_normalized = (train_data - mean_train) / std_train
    valid_data_normalized = (valid_data - mean_train) / std_train
    test_data_normalized = (test_data - mean_train) / std_train

    input_shape = (Config.LSTM_STEP,
                   Config.CNN_WIDE, Config.CNN_HIGH, Config.CNN_CHANNEL)

    conv_lstm_net = load_trained_models(args, input_shape, Config.CONV_LSTM_BEST_CHECKPOINT)

    results_summary = pd.read_csv(Config.RESULTS_PATH + 'sample_results.csv')

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    measured_matrix_ims = np.zeros((test_data.shape[0] - Config.IMS_STEP + 1, Config.CNN_WIDE, Config.CNN_HIGH))

    for i in range(Config.TESTING_TIME):
        print('|--- Run time {}'.format(i))

        tm_labels, ims_tm = predict_conv_lstm(
            initial_data=valid_data_normalized[-Config.LSTM_STEP:, :, :],
            test_data=test_data_normalized,
            conv_lstm_model=conv_lstm_net.model)

        pred_tm = tm_labels[:, :, :, 0]
        measured_matrix = tm_labels[:, :, :, 1]

        pred_tm = pred_tm * std_train + mean_train

        err.append(error_ratio(y_true=test_data, y_pred=pred_tm, measured_matrix=measured_matrix))
        r2_score.append(calculate_r2_score(y_true=test_data, y_pred=pred_tm))
        rmse.append(calculate_rmse(y_true=test_data, y_pred=pred_tm))

        ims_tm = ims_tm * std_train + mean_train
        ims_ytrue = ims_tm_ytrue(test_data=test_data)

        err_ims.append(error_ratio(y_pred=ims_tm,
                                   y_true=ims_ytrue,
                                   measured_matrix=measured_matrix_ims))

        r2_score_ims.append(calculate_r2_score(y_true=ims_ytrue, y_pred=ims_tm))
        rmse_ims.append(calculate_rmse(y_true=ims_ytrue, y_pred=ims_tm))

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
