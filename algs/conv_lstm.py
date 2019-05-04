import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from Models.ConvLSTM_model import ConvLSTM
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_3d, create_offline_convlstm_data_fix_ratio
from common.error_utils import error_ratio, calculate_r2_score, \
    calculate_rmse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


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
    tf_a = np.array([True, False])

    init_labels = np.ones((initial_data.shape[0], initial_data.shape[1], initial_data.shape[2]))

    tm_labels = np.zeros(
        shape=(initial_data.shape[0] + test_data.shape[0], initial_data.shape[1], initial_data.shape[2], 2))
    tm_labels[0:initial_data.shape[0], :, :, 0] = initial_data
    tm_labels[0:init_labels.shape[0], :, :, 1] = init_labels

    ims_tm = np.zeros(
        shape=(test_data.shape[0] - Config.CONV_LSTM_IMS_STEP + 1, test_data.shape[1], test_data.shape[2]))

    for ts in tqdm(range(test_data.shape[0])):

        if Config.CONV_LSTM_IMS and (ts <= test_data.shape[0] - Config.CONV_LSTM_IMS_STEP):
            ims_tm[ts] = ims_tm_prediction(init_data_labels=tm_labels[ts:ts + Config.CONV_LSTM_STEP, :, :, :],
                                           conv_lstm_model=conv_lstm_model)

        rnn_input = tm_labels[ts:(ts + Config.CONV_LSTM_STEP), :, :, :]  # shape(timesteps, od, od , 2)

        rnn_input = np.expand_dims(rnn_input, axis=0)  # shape(1, timesteps, od, od , 2)

        predictX = conv_lstm_model.predict(rnn_input)  # shape(1, timesteps, od, od , 1)

        predictX = np.squeeze(predictX, axis=0)  # shape(timesteps, od, od , 1)
        predictX = np.squeeze(predictX, axis=3)  # shape(timesteps, od, od)

        predict_tm = predictX[-1, :, :]

        # Selecting next monitored flows randomly
        sampling = np.random.choice(tf_a, size=(test_data.shape[1], test_data.shape[2]),
                                    p=(Config.CONV_LSTM_MON_RAIO, 1 - Config.CONV_LSTM_MON_RAIO))
        inv_sampling = np.invert(sampling)

        pred_tm = predict_tm * inv_sampling
        corrected_data = test_data[ts, :, :]
        ground_truth = corrected_data * sampling

        # Calculating the true value for the TM
        new_tm = pred_tm + ground_truth

        # Concaternating the new tm to the final results
        new_tm = np.concatenate([np.expand_dims(new_tm, axis=2), np.expand_dims(sampling, axis=2)], axis=2)
        tm_labels[ts + Config.CONV_LSTM_STEP] = new_tm  # Shape = (timestep, 12, 12, 2)

    return tm_labels[Config.CONV_LSTM_STEP:, :, :, :], ims_tm


def build_model(args, input_shape):
    print('|--- Build models.')
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    conv_lstm_net = ConvLSTM(input_shape=input_shape,
                             cnn_layers=Config.CONV_LSTM_LAYERS,
                             a_filters=Config.CONV_LSTM_FILTERS,
                             a_strides=Config.CONV_LSTM_STRIDES,
                             dropouts=Config.CONV_LSTM_DROPOUTS,
                             kernel_sizes=Config.CONV_LSTM_KERNEL_SIZE,
                             rnn_dropouts=Config.CONV_LSTM_RNN_DROPOUTS,
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


def train_conv_lstm(data, experiment, args):
    print('|-- Run model training.')
    gpu = args.gpu

    params = Config.set_comet_params_conv_lstm()

    data_name = args.data_name
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_valid_test_3d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')

    scalers = {
        'min_train': 0,
        'max_train': 0,
        'mean_train': 0,
        'std_train': 0,
    }

    if Config.MIN_MAX_SCALER:
        scalers['min_train'] = np.min(train_data)
        scalers['max_train'] = np.max(train_data)
        train_data_normalized = (train_data - scalers['min_train']) / (scalers['max_train'] - scalers['min_train'])
        valid_data_normalized = (valid_data - scalers['min_train']) / (scalers['max_train'] - scalers['min_train'])
    else:
        scalers['mean_train'] = np.mean(train_data)
        scalers['std_train'] = np.std(train_data)
        train_data_normalized = (train_data - scalers['mean_train']) / scalers['std_train']
        valid_data_normalized = (valid_data - scalers['mean_train']) / scalers['std_train']

    input_shape = (Config.CONV_LSTM_STEP,
                   Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH, Config.CONV_LSTM_CHANNEL)

    with tf.device('/device:GPU:{}'.format(gpu)):
        conv_lstm_net = build_model(args, input_shape)

    # -------------------------------- Create offline training and validating dataset ------------------------------
    if not os.path.isfile(conv_lstm_net.saving_path + 'trainX.npy'):
        print('|--- Create offline train set for conv_lstm net!')

        trainX, trainY = create_offline_convlstm_data_fix_ratio(train_data_normalized,
                                                                input_shape, Config.CONV_LSTM_MON_RAIO, 0.5)
        np.save(conv_lstm_net.saving_path + 'trainX.npy', trainX)
        np.save(conv_lstm_net.saving_path + 'trainY.npy', trainY)
    else:
        trainX = np.load(conv_lstm_net.saving_path + 'trainX.npy')
        trainY = np.load(conv_lstm_net.saving_path + 'trainY.npy')

    if not os.path.isfile(conv_lstm_net.saving_path + 'validX.npy'):
        print('|--- Create offline valid set for conv_lstm net!')

        validX, validY = create_offline_convlstm_data_fix_ratio(valid_data_normalized,
                                                                input_shape, Config.CONV_LSTM_MON_RAIO, 0.5)
        np.save(conv_lstm_net.saving_path + 'validX.npy', validX)
        np.save(conv_lstm_net.saving_path + 'validY.npy', validY)
    else:
        validX = np.load(conv_lstm_net.saving_path + 'validX.npy')
        validY = np.load(conv_lstm_net.saving_path + 'validY.npy')
    # --------------------------------------------------------------------------------------------------------------

    with experiment.train():
        if os.path.isfile(path=conv_lstm_net.checkpoints_path + 'weights-{:02d}.hdf5'.format(Config.CONV_LSTM_N_EPOCH)):
            print('|--- Model exist!')
            conv_lstm_net.load_model_from_check_point(_from_epoch=Config.CONV_LSTM_BEST_CHECKPOINT)
        else:
            print('|--- Compile model. Saving path %s --- ' % conv_lstm_net.saving_path)

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
                                                           initial_epoch=from_epoch)
            else:
                print('|--- Training new model.')

                training_history = conv_lstm_net.model.fit(x=trainX,
                                                           y=trainY,
                                                           batch_size=Config.CONV_LSTM_BATCH_SIZE,
                                                           epochs=Config.CONV_LSTM_N_EPOCH,
                                                           callbacks=conv_lstm_net.callbacks_list,
                                                           validation_data=(validX, validY),
                                                           shuffle=True,
                                                           initial_epoch=from_epoch)

            # Plot the training history
            if training_history is not None:
                conv_lstm_net.plot_training_history(training_history)

        print('---------------------------------CONV_LSTM_NET SUMMARY---------------------------------')
        print(conv_lstm_net.model.summary())

        experiment.log_parameters(params)

    return


def ims_tm_ytrue(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.CONV_LSTM_IMS_STEP + 1, test_data.shape[1], test_data.shape[2]))

    for i in range(Config.CONV_LSTM_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.CONV_LSTM_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def test_conv_lstm(data, experiment, args):
    print('|-- Run model testing.')
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
        assert Config.CONV_LSTM_WIDE == 12
        assert Config.CONV_LSTM_HIGH == 12
    else:
        day_size = Config.GEANT_DAY_SIZE
        assert Config.CONV_LSTM_WIDE == 23
        assert Config.CONV_LSTM_HIGH == 23

    if not Config.ALL_DATA:
        data = data[0:Config.NUM_DAYS * day_size]

    print('|--- Splitting train-test set.')
    train_data, valid_data, test_data = prepare_train_valid_test_3d(data=data, day_size=day_size)
    if 'Abilene' in data_name:
        print('|--- Remove last 3 days in test data.')
        test_data = test_data[0:-day_size * 3]

    print('|--- Normalizing the train set.')
    min_train, max_train, mean_train, std_train = 0, 0, 0, 0
    if Config.MIN_MAX_SCALER:
        min_train = np.min(train_data)
        max_train = np.max(train_data)
        valid_data_normalized = (valid_data - min_train) / (max_train - min_train)
        test_data_normalized = (test_data - min_train) / (max_train - min_train)
    else:
        mean_train = np.mean(train_data)
        std_train = np.std(train_data)
        valid_data_normalized = (valid_data - mean_train) / std_train
        test_data_normalized = (test_data - mean_train) / std_train

    input_shape = (Config.CONV_LSTM_STEP,
                   Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH, Config.CONV_LSTM_CHANNEL)

    conv_lstm_net = load_trained_models(args, input_shape, Config.CONV_LSTM_BEST_CHECKPOINT)


    return


def run_test(experiment, test_data, test_data_normalized, init_data, conv_lstm_net, params, scalers, args,
             save_results=False):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    results_summary = pd.DataFrame(index=range(Config.CONV_LSTM_TESTING_TIME),
                                   columns=['No.', 'err', 'r2', 'rmse', 'err_ims', 'r2_ims', 'rmse_ims'])

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    measured_matrix_ims = np.zeros(
        (test_data.shape[0] - Config.CONV_LSTM_IMS_STEP + 1, Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH))

    if save_results:
        if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name)):
            np.save(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name),
                    test_data)

        if Config.MIN_MAX_SCALER:
            if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_scaled_{}_minmax.npy'.format(data_name)):
                print(())
                np.save(Config.RESULTS_PATH + 'ground_true_scaled_{}_minmax.npy'.format(data_name),
                        test_data_normalized)
        else:
            if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_scaled_{}.npy'.format(data_name)):
                print(())
                np.save(Config.RESULTS_PATH + 'ground_true_scaled_{}.npy'.format(data_name),
                        test_data_normalized)

    with experiment.test():
        for i in range(Config.CONV_LSTM_TESTING_TIME):
            print('|--- Run time {}'.format(i))

            tm_labels, ims_tm = predict_conv_lstm(
                initial_data=init_data,
                test_data=test_data_normalized,
                conv_lstm_model=conv_lstm_net.model)

            pred_tm = tm_labels[:, :, :, 0]
            measured_matrix = tm_labels[:, :, :, 1]

            np.save(Config.RESULTS_PATH + '[pred_scaled-{}]{}-{}-{}-{}.npy'.format(i, data_name, alg_name, tag,
                                                                                   Config.ADDED_RESULT_NAME),
                    pred_tm)

            if Config.MIN_MAX_SCALER:
                pred_tm_invert = pred_tm * (scalers['max_train'] - scalers['min_train']) + scalers['min_train']
            else:
                pred_tm_invert = pred_tm * scalers['std_train'] + scalers['mean_train']

            err.append(error_ratio(y_true=test_data, y_pred=pred_tm_invert, measured_matrix=measured_matrix))
            r2_score.append(calculate_r2_score(y_true=test_data, y_pred=pred_tm_invert))
            rmse.append(calculate_rmse(y_true=test_data, y_pred=pred_tm_invert))

            if Config.CONV_LSTM_IMS:
                if Config.MIN_MAX_SCALER:
                    ims_tm_invert = ims_tm * (scalers['max_train'] - scalers['min_train']) + scalers['min_train']
                else:
                    ims_tm_invert = ims_tm * scalers['std_train'] + scalers['mean_train']

                ims_ytrue = ims_tm_ytrue(test_data=test_data)

                err_ims.append(error_ratio(y_pred=ims_tm_invert,
                                           y_true=ims_ytrue,
                                           measured_matrix=measured_matrix_ims))

                r2_score_ims.append(calculate_r2_score(y_true=ims_ytrue, y_pred=ims_tm_invert))
                rmse_ims.append(calculate_rmse(y_true=ims_ytrue, y_pred=ims_tm_invert))
            else:
                err_ims.append(0)
                r2_score_ims.append(0)
                rmse_ims.append(0)

            np.save(Config.RESULTS_PATH + '[pred-{}]{}-{}-{}-{}.npy'.format(i, data_name, alg_name, tag,
                                                                            Config.ADDED_RESULT_NAME),
                    pred_tm_invert)
            np.save(Config.RESULTS_PATH + '[measure-{}]{}-{}-{}-{}.npy'.format(i, data_name, alg_name, tag,
                                                                               Config.ADDED_RESULT_NAME),
                    measured_matrix)

            print('Result: err\trmse\tr2 \t\t err_ims\trmse_ims\tr2_ims')
            print('        {}\t{}\t{} \t\t {}\t{}\t{}'.format(err[i], rmse[i], r2_score[i],
                                                              err_ims[i], rmse_ims[i],
                                                              r2_score_ims[i]))

        results_summary['No.'] = range(Config.CONV_LSTM_TESTING_TIME)
        results_summary['err'] = err
        results_summary['r2'] = r2_score
        results_summary['rmse'] = rmse
        results_summary['err_ims'] = err_ims
        results_summary['r2_ims'] = r2_score_ims
        results_summary['rmse_ims'] = rmse_ims

        results_summary.to_csv(Config.RESULTS_PATH + '{}-{}-{}-{}.csv'.format(data_name,
                                                                              alg_name, tag, Config.ADDED_RESULT_NAME),
                               index=False)

        metrics = {
            'err': results_summary['err'],
            'rmse': results_summary['rmse'],
            'r2': results_summary['r2'],
            'err_ims': results_summary['err_ims'],
            'rmse_ims': results_summary['rmse_ims'],
            'r2_ims': results_summary['rmse_ims'],
        }

        experiment.log_metrics(metrics)
        experiment.log_parameters(params)

    return
