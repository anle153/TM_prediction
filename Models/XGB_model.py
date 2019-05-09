import os
import pickle

from xgboost import XGBRegressor


class XGB(object):
    def __init__(self, data_name, saving_path, alg_name, tag, objective='reg:squarederror'):

        self.data_name = data_name
        self.alg_name = alg_name
        self.tag = tag
        self.saving_path = saving_path
        self.objective = objective

        self.model = XGBRegressor(objective=self.objective)

    def save_model(self):
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        saved_model = open(self.saving_path + 'xgb.models', 'wb')
        pickle.dump(self.model, saved_model, 2)

    def load_model(self):
        if not os.path.isfile(self.saving_path + 'xgb.models'):
            raise Exception('Saved model not found!')
