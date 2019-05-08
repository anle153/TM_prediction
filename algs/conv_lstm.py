import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from Models.ConvLSTM_model import ConvLSTM
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_2d, create_offline_convlstm_data_fix_ratio, data_scalling
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
                             saving_path=Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name, alg_name, tag,
                                                                                   Config.SCALER))

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
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')

    train_data_normalized2d, valid_data_normalized2d, _, scalers = data_scalling(train_data2d,
                                                                                 valid_data2d,
                                                                                 test_data2d)

    train_data_normalized = np.reshape(np.copy(train_data_normalized2d), newshape=(train_data_normalized2d.shape[0],
                                                                                   Config.FWBW_CONV_LSTM_WIDE,
                                                                                   Config.FWBW_CONV_LSTM_HIGH))
    valid_data_normalized = np.reshape(np.copy(valid_data_normalized2d), newshape=(valid_data_normalized2d.shape[0],
                                                                                   Config.FWBW_CONV_LSTM_WIDE,
                                                                                   Config.FWBW_CONV_LSTM_HIGH))

    input_shape = (Config.CONV_LSTM_STEP,
                   Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH, Config.CONV_LSTM_CHANNEL)

    with tf.device('/device:GPU:{}'.format(gpu)):
        conv_lstm_net = build_model(args, input_shape)

    # -------------------------------- Create offline training and validating dataset ------------------------------
    print('|--- Create offline train set for conv_lstm net!')

    trainX, trainY = create_offline_convlstm_data_fix_ratio(train_data_normalized,
                                                            input_shape, Config.CONV_LSTM_MON_RAIO, 0.5)
    print('|--- Create offline valid set for conv_lstm net!')

    validX, validY = create_offline_convlstm_data_fix_ratio(valid_data_normalized,
                                                            input_shape, Config.CONV_LSTM_MON_RAIO, 0.5)
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

    # run_test(experiment, test_data, test_data_normalized, init_data, fw_net, bw_net, params, scalers, args)
    run_test(experiment, valid_data2d, valid_data_normalized2d, train_data_normalized2d[-Config.FWBW_CONV_LSTM_STEP:],
             conv_lstm_net, params, scalers, args)
    return


def ims_tm_test_data(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.LSTM_IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.LSTM_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.LSTM_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def test_conv_lstm(data, experiment, args):
    print('|-- Run model testing.')
    params = Config.set_comet_params_conv_lstm()
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
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=data, day_size=day_size)
    print('|--- Normalizing the train set.')

    if 'Abilene' in data_name:
        print('|--- Remove last 3 days in test data.')
        test_data2d = test_data2d[0:-day_size * 3]

    _, valid_data_normalized2d, test_data_normalized2d, scalers = data_scalling(train_data2d,
                                                                                valid_data2d,
                                                                                test_data2d)

    input_shape = (Config.CONV_LSTM_STEP,
                   Config.CONV_LSTM_WIDE, Config.CONV_LSTM_HIGH, Config.CONV_LSTM_CHANNEL)

    conv_lstm_net = load_trained_models(args, input_shape, Config.CONV_LSTM_BEST_CHECKPOINT)
    run_test(experiment, test_data2d, test_data_normalized2d, valid_data_normalized2d[-Config.FWBW_CONV_LSTM_STEP:],
             conv_lstm_net, params, scalers, args)

    return


def run_test(experiment, test_data2d, test_data_normalized2d, init_data2d, conv_lstm_net, params, scalers, args):
    alg_name = args.alg
    tag = args.tag
    data_name = args.data_name

    results_summary = pd.DataFrame(index=range(Config.CONV_LSTM_TESTING_TIME),
                                   columns=['No.', 'err', 'r2', 'rmse', 'err_ims', 'r2_ims', 'rmse_ims'])

    err, r2_score, rmse = [], [], []
    err_ims, r2_score_ims, rmse_ims = [], [], []

    measured_matrix_ims2d = np.zeros((test_data2d.shape[0] - Config.FWBW_CONV_LSTM_IMS_STEP + 1,
                                      Config.FWBW_CONV_LSTM_WIDE * Config.FWBW_CONV_LSTM_HIGH))

    if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name)):
        np.save(Config.RESULTS_PATH + 'ground_true_{}.npy'.format(data_name),
                test_data2d)

    if not os.path.isfile(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER)):
        np.save(Config.RESULTS_PATH + 'ground_true_scaled_{}_{}.npy'.format(data_name, Config.SCALER),
                test_data_normalized2d)

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name,
                                                                      alg_name, tag, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER))

    with experiment.test():
        for i in range(Config.CONV_LSTM_TESTING_TIME):
            print('|--- Run time {}'.format(i))
            init_data = np.reshape(init_data2d, newshape=(init_data2d.shape[0],
                                                          Config.FWBW_CONV_LSTM_WIDE,
                                                          Config.FWBW_CONV_LSTM_HIGH))
            test_data_normalized = np.reshape(test_data_normalized2d, newshape=(test_data_normalized2d.shape[0],
                                                                                Config.FWBW_CONV_LSTM_WIDE,
                                                                                Config.FWBW_CONV_LSTM_HIGH))

            tm_labels, ims_tm = predict_conv_lstm(
                initial_data=init_data,
                test_data=test_data_normalized,
                conv_lstm_model=conv_lstm_net.model)

            pred_tm = tm_labels[:, :, :, 0]
            measured_matrix = tm_labels[:, :, :, 1]

            pred_tm2d = np.reshape(np.copy(pred_tm), newshape=(pred_tm.shape[0], pred_tm.shape[1] * pred_tm.shape[2]))
            measured_matrix2d = np.reshape(np.copy(measured_matrix),
                                           newshape=(measured_matrix.shape[0],
                                                     measured_matrix.shape[1] * measured_matrix.shape[2]))
            np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred_scaled-{}.npy'.format(data_name, alg_name, tag,
                                                                                  Config.SCALER, i),
                    pred_tm2d)

            pred_tm_invert2d = scalers.inverse_transform(pred_tm2d)

            err.append(error_ratio(y_true=test_data2d, y_pred=pred_tm_invert2d, measured_matrix=measured_matrix2d))
            r2_score.append(calculate_r2_score(y_true=test_data2d, y_pred=pred_tm_invert2d))
            rmse.append(calculate_rmse(y_true=test_data2d / 1000000, y_pred=pred_tm_invert2d / 1000000))

            if Config.CONV_LSTM_IMS:
                ims_tm2d = np.reshape(np.copy(ims_tm), newshape=(ims_tm.shape[0], ims_tm.shape[1] * ims_tm.shape[2]))

                ims_tm_invert2d = scalers.inverse_transform(ims_tm2d)

                ims_ytrue2d = ims_tm_test_data(test_data=test_data2d)

                err_ims.append(error_ratio(y_pred=ims_tm_invert2d,
                                           y_true=ims_ytrue2d,
                                           measured_matrix=measured_matrix_ims2d))

                r2_score_ims.append(calculate_r2_score(y_true=ims_ytrue2d, y_pred=ims_tm_invert2d))
                rmse_ims.append(calculate_rmse(y_true=ims_ytrue2d / 1000000, y_pred=ims_tm_invert2d / 1000000))
            else:
                err_ims.append(0)
                r2_score_ims.append(0)
                rmse_ims.append(0)

            np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/pred-{}.npy'.format(data_name, alg_name, tag,
                                                                           Config.SCALER, i),
                    pred_tm_invert2d)
            np.save(Config.RESULTS_PATH + '{}-{}-{}-{}/measure-{}.npy'.format(data_name, alg_name, tag,
                                                                              Config.SCALER, i),
                    measured_matrix2d)

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

        results_summary.to_csv(Config.RESULTS_PATH + '{}-{}-{}-{}/results.csv'.format(data_name,
                                                                                      alg_name, tag, Config.SCALER),
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
