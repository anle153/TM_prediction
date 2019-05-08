import os

from Models.XGB_model import XGB
from common import Config
from common.DataPreprocessing import data_scalling, prepare_train_valid_test_2d, create_offline_lstm_nn_data


def train_xgboost(data, args):
    print('|-- Run xgboost model training.')

    alg_name = args.alg
    tag = args.tag
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
    saving_path = Config.MODEL_SAVE + '{}-{}-{}-{}/'.format(data_name, alg_name, tag, Config.SCALER)
    xgb_model = XGB(data_name, saving_path, alg_name, tag)

    input_shape = (Config.LSTM_STEP, Config.LSTM_FEATURES)

    if not os.path.isfile(xgb_model.saving_path + 'trainX.npy'):
        print('|--- Create offline train set for lstm-nn!')
        trainX, trainY = create_offline_lstm_nn_data(train_data_normalized2d, input_shape, Config.LSTM_MON_RAIO, 0.5)
        np.save(lstm_net.saving_path + 'trainX.npy', trainX)
        np.save(lstm_net.saving_path + 'trainY.npy', trainY)
    else:
        trainX = np.load(lstm_net.saving_path + 'trainX.npy')
        trainY = np.load(lstm_net.saving_path + 'trainY.npy')

    if not os.path.isfile(lstm_net.saving_path + 'validX.npy'):
        print('|--- Create offline valid set for lstm-nn!')
        validX, validY = create_offline_lstm_nn_data(valid_data_normalized2d, input_shape, Config.LSTM_MON_RAIO, 0.5)
        np.save(lstm_net.saving_path + 'validX.npy', validX)
        np.save(lstm_net.saving_path + 'validY.npy', validY)
    else:
        validX = np.load(lstm_net.saving_path + 'validX.npy')
        validY = np.load(lstm_net.saving_path + 'validY.npy')
