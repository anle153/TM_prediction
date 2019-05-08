import numpy as np
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from common import Config
from common.DataPreprocessing import data_scalling, prepare_train_valid_test_2d, create_offline_xgb_data


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
    xgb_model = XGBRegressor(n_jobs=Config.XGB_NJOBS)

    trainX, trainY = create_offline_xgb_data(train_data_normalized2d, Config.XGB_STEP, Config.XGB_MON_RATIO, 0.5)

    validX, validY = create_offline_xgb_data(valid_data_normalized2d, Config.XGB_STEP, Config.XGB_MON_RATIO, 0.5)

    validX2d, validY2d = create_offline_xgb_data(valid_data2d, Config.XGB_STEP, Config.XGB_MON_RATIO, 0.5)
    print('|--- Training model.')

    xgb_model.fit(trainX, trainY)
    print('|--- Testing model.')

    preds = xgb_model.predict(validX)

    preds = np.reshape(preds, newshape=(valid_data2d.shape[0] - Config.XGB_STEP, valid_data2d.shape[1]))

    preds_inv = scalers.inverse_transform(preds)
    # preds_inv = preds_inv.flatten()

    _r2 = r2_score(y_pred=preds_inv.flatten(), y_true=valid_data2d[Config.XGB_STEP:].flatten())
    print("R2_score: %f" % _r2)
