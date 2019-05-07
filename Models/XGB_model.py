from xgboost import XGBRegressor

from Models.AbstractModel import AbstractModel


class xgb(AbstractModel):
    def __init__(self, saving_path, input_shape, hidden, drop_out,
                 alg_name=None, tag=None, early_stopping=False, check_point=False):
        super().__init__(alg_name=alg_name, tag=tag, early_stopping=early_stopping, check_point=check_point,
                         saving_path=saving_path)

        self.model = XGBRegressor()
