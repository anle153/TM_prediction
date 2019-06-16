import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from Models.fwbw_LSTM import fwbw_lstm_model
from common import Config
from common.DataPreprocessing import prepare_train_valid_test_2d, data_scalling, create_offline_fwbw_lstm
from common.error_utils import error_ratio, calculate_r2_score, calculate_rmse, calculate_mape

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def plot_test_data(prefix, raw_data, pred_fw, pred_bw, current_data):
    saving_path = Config.RESULTS_PATH + 'plot_check_fwbw_lstm/'

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    from matplotlib import pyplot as plt
    for flow_id in range(raw_data.shape[1]):
        plt.plot(raw_data[:, flow_id], label='Actual')
        plt.plot(pred_fw[:, flow_id], label='Pred_fw')
        plt.plot(pred_bw[:, flow_id], label='Pred_bw')
        plt.plot(current_data[:, flow_id], label='Current_pred')

        plt.legend()
        plt.savefig(saving_path + '{}_flow_{:02d}.png'.format(prefix, flow_id))
        plt.close()


def prepare_input_online_prediction(data, labels):
    labels = labels.astype(int)
    dataX = np.zeros(shape=(data.shape[1], Config.FWBW_LSTM_STEP, Config.FWBW_LSTM_FEATURES))
    for flow_id in range(data.shape[1]):
        x = data[:, flow_id]
        label = labels[:, flow_id]

        sample = np.array([x, label]).T
        dataX[flow_id] = sample

    return dataX


def calculate_forward_backward_loss(measured_block, pred_forward, pred_backward, rnn_input):
    l_fw, l_bw = [], []
    for flow_id in range(rnn_input.shape[0]):
        idx_fw = measured_block[flow_id, 1:]

        l_fw.append(error_ratio(y_true=rnn_input[flow_id, 1:][idx_fw == 1.0],
                                y_pred=pred_forward[flow_id, :-1][idx_fw == 1.0],
                                measured_matrix=np.zeros(idx_fw[idx_fw == 1.0].shape)))
        idx_bw = measured_block[flow_id, 0:-1]

        l_bw.append(error_ratio(y_true=rnn_input[flow_id, :-1][idx_bw == 1.0],
                                y_pred=pred_backward[flow_id, 1:][idx_bw == 1.0],
                                measured_matrix=np.zeros(idx_bw[idx_bw == 1.0].shape)))

    l_fw = np.array(l_fw)
    l_fw[l_fw == 0.] = np.max(l_fw)
    l_bw = np.array(l_bw)
    l_bw[l_bw == 0.] = np.max(l_bw)

    return l_fw, l_bw


def calculate_confident_factors(measured_block, forward_loss, backward_loss):
    eps = 0.0001

    labels = measured_block.astype(int)  # shape = (#nflows, #steps)

    measured_count = np.sum(labels, axis=1).astype(float)  # shape = (#nflows,)
    _eta = measured_count / Config.FWBW_LSTM_STEP

    _eta[_eta == 0.0] = eps

    alpha = 1.0 - _eta  # shape = (#nflows,)
    alpha = np.tile(np.expand_dims(alpha, axis=1), (1, Config.FWBW_LSTM_STEP))  # shape = (#nflows, #steps)

    rho = np.zeros((measured_block.shape[0], Config.FWBW_LSTM_STEP))
    mu = np.zeros((measured_block.shape[0], Config.FWBW_LSTM_STEP))
    for j in range(0, Config.FWBW_LSTM_STEP):
        _rho = (np.sum(measured_block[:, j:], axis=1)) / float(Config.FWBW_LSTM_STEP - j)
        _mu = (np.sum(measured_block[:, :(j + 1)], axis=1)) / float(j + 1)
        rho[:, j] = _rho
        mu[:, j] = _mu

    forward_loss = np.tile(np.expand_dims(forward_loss, axis=1), (1, Config.FWBW_LSTM_STEP))
    backward_loss = np.tile(np.expand_dims(backward_loss, axis=1), (1, Config.FWBW_LSTM_STEP))

    beta = (backward_loss + mu) * (1.0 - alpha) / (forward_loss + backward_loss + mu + rho)

    gamma = (forward_loss + rho) * (1.0 - alpha) / (forward_loss + backward_loss + mu + rho)

    return alpha, beta, gamma


def data_correction(rnn_input, pred_forward, pred_backward, labels):
    rnn_input = np.copy(rnn_input).T  # shape = (#step, #nflows)
    labels = np.copy(labels)

    forward_loss, backward_loss = calculate_forward_backward_loss(measured_block=measured_block,
                                                                  pred_forward=pred_forward,
                                                                  pred_backward=pred_backward,
                                                                  rnn_input=rnn_input)

    alpha, beta, gamma = calculate_confident_factors(measured_block=measured_block,
                                                     forward_loss=forward_loss,
                                                     backward_loss=backward_loss)

    considered_forward = pred_forward[:, :-2]
    considered_backward = pred_backward[:, 2:]
    considered_rnn_input = rnn_input[:, 1:-1]

    alpha = alpha[:, 1:-1]
    beta = beta[:, 1:-1]
    gamma = gamma[:, 1:-1]

    updated_rnn_input = considered_backward * gamma + considered_forward * beta + considered_rnn_input * alpha

    sampling_measured_matrix = measured_block[:, 1:-1]
    inv_sampling_measured_matrix = 1.0 - sampling_measured_matrix
    bidirect_rnn_pred_value = updated_rnn_input * inv_sampling_measured_matrix

    tm_pred[1:-1, :] = \
        tm_pred[1:-1, :] * sampling_measured_matrix.T + bidirect_rnn_pred_value.T

    return tm_pred


def predict_fwbw_lstm(initial_data, test_data, model):
    tf_a = np.array([1.0, 0.0])
    labels = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1]))

    tm_pred = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1]))
    tm_pred_no_updated = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1]))

    tm_pred[0:initial_data.shape[0]] = initial_data
    labels[0:initial_data.shape[0]] = np.ones(shape=initial_data.shape)
    tm_pred_no_updated[0:initial_data.shape[0]] = initial_data

    ims_tm = np.zeros(shape=(test_data.shape[0] - Config.FWBW_LSTM_IMS_STEP + 1, test_data.shape[1]))

    raw_data = np.zeros(shape=(initial_data.shape[0] + test_data.shape[0], test_data.shape[1]))

    raw_data[0:initial_data.shape[0]] = initial_data
    raw_data[initial_data.shape[0]:] = test_data

    # Predict the TM from time slot look_back
    for ts in tqdm(range(test_data.shape[0])):

        # Create 3D input for rnn
        rnn_input = prepare_input_online_prediction(data=tm_pred[ts: ts + Config.FWBW_LSTM_STEP],
                                                    labels=labels[ts: ts + Config.FWBW_LSTM_STEP])

        rnn_input_wo_corr = prepare_input_online_prediction(data=tm_pred_no_updated[ts: ts + Config.FWBW_LSTM_STEP],
                                                            labels=labels[ts: ts + Config.FWBW_LSTM_STEP])

        # Get the TM prediction of next time slot
        fw_outputs, bw_outputs = model.predict(rnn_input)
        _fw_outputs, _ = model.predict(rnn_input)

        # Data Correction
        updated_data = corr_data.T
        _labels = labels[ts + 1:ts + Config.FWBW_LSTM_STEP - 1]
        updated_data = updated_data * (1 - _labels)
        tm_pred[ts + 1:ts + Config.FWBW_LSTM_STEP - 1] = tm_pred[
                                                         ts + 1:ts + Config.FWBW_LSTM_STEP - 1] * _labels + updated_data

        # _err_2 = error_ratio(y_pred=tm_pred[0:ts + Config.FWBW_CONV_LSTM_STEP],
        #                      y_true=raw_data[0:ts + Config.FWBW_CONV_LSTM_STEP],
        #                      measured_matrix=labels[0: ts + Config.FWBW_CONV_LSTM_STEP])

        # Sampling data
        sampling = np.random.choice(tf_a, size=(test_data.shape[1]),
                                    p=[Config.FWBW_LSTM_MON_RAIO, 1 - Config.FWBW_LSTM_MON_RAIO])

        labels[ts + Config.FWBW_LSTM_STEP] = sampling  # Store sampling to labels
        inv_sampling = 1.0 - sampling  # invert of sampling: for choosing value from the original data

        pred_input = pred_data.T * inv_sampling
        ground_true = test_data[ts]
        measured_input = ground_true * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        tm_pred[ts + Config.FWBW_LSTM_STEP] = new_input
        tm_pred_no_updated[ts + Config.FWBW_LSTM_STEP] = new_input

    _err_1 = error_ratio(y_pred=tm_pred[Config.FWBW_CONVLSTM_STEP:],
                         y_true=raw_data[Config.FWBW_CONVLSTM_STEP:],
                         measured_matrix=labels[Config.FWBW_CONVLSTM_STEP:])
    _err_2 = error_ratio(y_pred=tm_pred_no_updated[Config.FWBW_CONVLSTM_STEP:],
                         y_true=raw_data[Config.FWBW_CONVLSTM_STEP:],
                         measured_matrix=labels[Config.FWBW_CONVLSTM_STEP:])

    print('Err_w_updated: {} -- Err_wo_updated: {}'.format(_err_1, _err_2))

    return tm_pred[Config.FWBW_LSTM_STEP:], labels[Config.FWBW_LSTM_STEP:], ims_tm


def build_model(input_shape):
    print('|--- Build models fwbw lstm.')
    alg_name = Config.ALG
    tag = Config.TAG
    data_name = Config.DATA_NAME

    # lstm forward model
    fwbw_net = fwbw_lstm_model(input_shape=input_shape,
                               hidden=Config.FWBW_LSTM_HIDDEN_UNIT,
                               drop_out=Config.FWBW_LSTM_DROPOUT,
                               alg_name=alg_name, tag=tag, check_point=True,
                               saving_path=Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name, alg_name, tag,
                                                                                     Config.SCALER))
    fwbw_net.construct_fwbw_lstm()
    print(fwbw_net.model.summary())
    fwbw_net.plot_models()
    return fwbw_net


def load_trained_models(input_shape, fw_ckp):
    fwbw_net = build_model(input_shape)
    print('|--- Load trained model from: {}'.format(fwbw_net.checkpoints_path))
    fwbw_net.model.load_weights(fwbw_net.checkpoints_path + "weights-{:02d}.hdf5".format(fw_ckp))

    return fwbw_net


def train_fwbw_lstm(data, experiment):
    print('|-- Run model training fwbw_lstm.')

    params = Config.set_comet_params_fwbw_lstm()

    gpu = Config.GPU

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

    input_shape = (Config.FWBW_LSTM_STEP, Config.FWBW_LSTM_FEATURES)

    with tf.device('/device:GPU:{}'.format(gpu)):
        fwbw_net = build_model(input_shape)

    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------Training fw model-------------------------------------------------

    if not Config.FWBW_LSTM_VALID_TEST or \
            not os.path.isfile(
                fwbw_net.checkpoints_path + 'weights-{:02d}.hdf5'.format(Config.FWBW_LSTM_BEST_CHECKPOINT)):
        print('|--- Compile model. Saving path %s --- ' % fwbw_net.saving_path)
        # -------------------------------- Create offline training and validating dataset --------------------------

        print('|--- Create offline train set for forward net!')

        trainX, trainY_1, trainY_2 = create_offline_fwbw_lstm(train_data_normalized2d,
                                                              input_shape, Config.FWBW_LSTM_MON_RAIO,
                                                              train_data_normalized2d.std())
        print('|--- Create offline valid set for forward net!')

        validX, validY_1, validY_2 = create_offline_fwbw_lstm(valid_data_normalized2d,
                                                              input_shape, Config.FWBW_LSTM_MON_RAIO,
                                                              train_data_normalized2d.std())

        # Load model check point
        from_epoch = fwbw_net.load_model_from_check_point()
        if from_epoch > 0:
            print('|--- Continue training forward model from epoch %i --- ' % from_epoch)
            training_fw_history = fwbw_net.model.fit(x=trainX,
                                                     y={'fw_output': trainY_1, 'bw_output': trainY_2},
                                                     batch_size=Config.FWBW_LSTM_BATCH_SIZE,
                                                     epochs=Config.FWBW_LSTM_N_EPOCH,
                                                     callbacks=fwbw_net.callbacks_list,
                                                     validation_data=(
                                                         validX, {'fw_output': validY_1, 'bw_output': validY_2}),
                                                     shuffle=True,
                                                     initial_epoch=from_epoch,
                                                     verbose=2)
        else:
            print('|--- Training new forward model.')

            training_fw_history = fwbw_net.model.fit(x=trainX,
                                                     y={'fw_output': trainY_1, 'bw_output': trainY_2},
                                                     batch_size=Config.FWBW_LSTM_BATCH_SIZE,
                                                     epochs=Config.FWBW_LSTM_N_EPOCH,
                                                     callbacks=fwbw_net.callbacks_list,
                                                     validation_data=(
                                                         validX, {'fw_output': validY_1, 'bw_output': validY_2}),
                                                     shuffle=True,
                                                     verbose=2)

        # Plot the training history
        if training_fw_history is not None:
            fwbw_net.plot_training_history(training_fw_history)

    else:
        fwbw_net.load_model_from_check_point(_from_epoch=Config.FWBW_LSTM_BEST_CHECKPOINT)
    # --------------------------------------------------------------------------------------------------------------
    run_test(valid_data2d, valid_data_normalized2d, train_data_normalized2d[-Config.FWBW_LSTM_STEP:],
             fwbw_net, params, scalers)
    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                      Config.ALG, Config.TAG, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                Config.ALG, Config.TAG, Config.SCALER))
    results_summary = pd.DataFrame(index=range(Config.FWBW_CONV_LSTM_TESTING_TIME),
                                   columns=['No.', 'mape, ''err', 'r2', 'rmse', 'mape_ims', 'err_ims', 'r2_ims',
                                            'rmse_ims'])

    results_summary = run_test(valid_data2d, valid_data_normalized2d, fwbw_net, scalers, results_summary)

    results_summary.to_csv(Config.RESULTS_PATH +
                           '{}-{}-{}-{}/Valid_results.csv'.format(Config.DATA_NAME, Config.ALG, Config.TAG,
                                                                  Config.SCALER),
                           index=False)

    return


def ims_tm_test_data(test_data):
    ims_test_set = np.zeros(
        shape=(test_data.shape[0] - Config.FWBW_LSTM_IMS_STEP + 1, test_data.shape[1]))

    for i in range(Config.FWBW_LSTM_IMS_STEP - 1, test_data.shape[0], 1):
        ims_test_set[i - Config.FWBW_LSTM_IMS_STEP + 1] = test_data[i]

    return ims_test_set


def test_fwbw_lstm(data, experiment):
    print('|-- Run model testing.')
    gpu = Config.GPU

    data_name = Config.DATA_NAME
    if 'Abilene' in data_name:
        day_size = Config.ABILENE_DAY_SIZE
    else:
        day_size = Config.GEANT_DAY_SIZE

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
    input_shape = (Config.FWBW_LSTM_STEP, Config.FWBW_LSTM_FEATURES)

    with tf.device('/device:GPU:{}'.format(gpu)):
        fwbw_net = load_trained_models(input_shape, Config.FWBW_LSTM_BEST_CHECKPOINT)

    if not os.path.exists(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                      Config.ALG, Config.TAG, Config.SCALER)):
        os.makedirs(Config.RESULTS_PATH + '{}-{}-{}-{}/'.format(Config.DATA_NAME,
                                                                Config.ALG, Config.TAG, Config.SCALER))
    results_summary = pd.DataFrame(index=range(Config.FWBW_CONV_LSTM_TESTING_TIME),
                                   columns=['No.', 'mape, ''err', 'r2', 'rmse', 'mape_ims', 'err_ims', 'r2_ims',
                                            'rmse_ims'])

    results_summary = run_test(test_data2d, test_data_normalized2d, fwbw_net, scalers, results_summary)

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

    idx = np.random.random_integers(Config.FWBW_CONV_LSTM_STEP, test_data2d.shape[0] - day_size * 2 - 10)

    test_data_normalize = np.copy(test_data_normalized2d[idx:idx + day_size * 2])
    init_data_normalize = np.copy(test_data_normalized2d[idx - Config.FWBW_CONV_LSTM_STEP: idx])
    test_data = test_data2d[idx:idx + day_size * 2]

    return test_data_normalize, init_data_normalize, test_data


def run_test(test_data2d, test_data_normalized2d, fwbw_net, scalers, results_summary):
    mape, err, r2_score, rmse = [], [], [], []
    mape_ims, err_ims, r2_score_ims, rmse_ims = [], [], [], []

    for i in range(Config.FWBW_LSTM_TESTING_TIME):
        print('|--- Run time {}'.format(i))
        test_data_normalize, init_data_normalize, test_data = prepare_test_set(test_data2d, test_data_normalized2d)
        ims_test_data = ims_tm_test_data(test_data=test_data)
        measured_matrix_ims = np.zeros(shape=ims_test_data.shape)

        pred_tm2d, measured_matrix2d, ims_tm2d, pred_tm2d_wo = predict_fwbw_lstm(initial_data=init_data_normalize,
                                                                                 test_data=test_data_normalize,
                                                                                 model=fwbw_net.model)

        pred_tm_invert2d = scalers.inverse_transform(pred_tm2d)
        pred_tm_wo_invert2d = scalers.inverse_transform(pred_tm2d_wo)
        if np.any(np.isinf(pred_tm_invert2d)):
            raise ValueError('Value is infinity!')
        elif np.any(np.isnan(pred_tm_invert2d)):
            raise ValueError('Value is NaN!')

        mape.append(calculate_mape(y_true=test_data, y_pred=pred_tm_invert2d))
        err.append(error_ratio(y_true=test_data, y_pred=pred_tm_invert2d, measured_matrix=measured_matrix2d))
        r2_score.append(calculate_r2_score(y_true=test_data, y_pred=pred_tm_invert2d))
        rmse.append(calculate_rmse(y_true=test_data / 1000000, y_pred=pred_tm_invert2d / 1000000))

        mape_wo = calculate_mape(y_true=test_data, y_pred=pred_tm_wo_invert2d)
        err_wo = error_ratio(y_true=test_data, y_pred=pred_tm_wo_invert2d, measured_matrix=measured_matrix2d)
        r2_score_wo = calculate_r2_score(y_true=test_data, y_pred=pred_tm_wo_invert2d)
        rmse_wo = calculate_rmse(y_true=test_data / 1000000, y_pred=pred_tm_wo_invert2d / 1000000)

        if Config.FWBW_IMS:
            # Calculate error for multistep-ahead-prediction
            ims_tm_invert2d = scalers.inverse_transform(ims_tm2d)

            mape_ims.append(calculate_mape(y_true=ims_test_data, y_pred=ims_tm_invert2d))
            err_ims.append(error_ratio(y_pred=ims_tm_invert2d,
                                       y_true=ims_test_data,
                                       measured_matrix=measured_matrix_ims))

            r2_score_ims.append(calculate_r2_score(y_true=ims_test_data, y_pred=ims_tm_invert2d))
            rmse_ims.append(calculate_rmse(y_true=ims_test_data / 1000000, y_pred=ims_tm_invert2d / 1000000))
        else:
            err_ims.append(0)
            r2_score_ims.append(0)
            rmse_ims.append(0)
            mape_ims.append(0)

        print('Result: mape\terr\trmse\tr2 \t\t mape_ims\terr_ims\trmse_ims\tr2_ims')
        print('        {}\t{}\t{}\t{} \t\t {}\t{}\t{}\t{}'.format(mape[i], err[i], rmse[i], r2_score[i],
                                                                  mape_ims[i], err_ims[i], rmse_ims[i],
                                                                  r2_score_ims[i]))
        print('Result without data correction: mape \t err\trmse\tr2')
        print('        {}\t{}\t{}\t{}'.format(mape_wo, err_wo, rmse_wo, r2_score_wo))

    results_summary['No.'] = range(Config.FWBW_LSTM_TESTING_TIME)
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
