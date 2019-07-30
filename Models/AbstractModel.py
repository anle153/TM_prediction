import os
import time

import keras.callbacks as keras_callbacks
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json


def plot_training_history(alg_name, tag, saving_path, model_history):
    plt.plot(model_history.history['loss'], label='mse')
    plt.plot(model_history.history['val_loss'], label='val_mse')
    plt.savefig(saving_path + '[MSE]{}-{}.png'.format(alg_name, tag))
    plt.legend()
    plt.close()

    plt.plot(model_history.history['val_loss'], label='val_mae')
    plt.legend()
    plt.savefig(saving_path + '[val_loss]{}-{}.png'.format(alg_name, tag))
    plt.close()


class AbstractModel(object):

    def __init__(self, saving_path, early_stopping=False, check_point=False, **kwargs):
        self._kwargs = kwargs
        self.saving_path = os.path.expanduser(saving_path)
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        self.callbacks_list = []

        self.checkpoints_path = self.saving_path + 'checkpoints/'

        if check_point:
            if not os.path.isdir(self.checkpoints_path):
                os.makedirs(self.checkpoints_path)
            self.checkpoints = ModelCheckpoint(
                self.checkpoints_path + "best_model.hdf5",
                monitor='val_loss', verbose=1,
                save_best_only=True,
                mode='auto', period=1)
            self.callbacks_list = [self.checkpoints]
        if early_stopping:
            self.earlystop = EarlyStopping(monitor='val_loss', patience=50,
                                           verbose=1, mode='auto')
            self.callbacks_list.append(self.earlystop)

        self.time_callback = TimeHistory()
        self.callbacks_list.append(self.time_callback)

    def load(self, model_json_file='trained_model.json', model_weight_file='trained_model.h5'):

        assert os.path.isfile(self.saving_path + model_json_file) & os.path.isfile(
            self.saving_path + model_weight_file)

        json_file = open(self.saving_path + model_json_file, 'r')
        model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(model_json)
        self.model.load_weights(self.saving_path + model_weight_file)

        return True

    def plot_training_history(self, model_history):
        plot_training_history(alg_name=self.alg_name,
                              tag=self.tag,
                           saving_path=self.saving_path,
                           model_history=model_history)

    def save_model_history(self, model_history):

        import numpy as np
        import pandas as pd

        loss = np.array(model_history.history['loss'])
        val_loss = np.array(model_history.history['val_loss'])
        dump_model_history = pd.DataFrame(index=range(loss.size),
                                          columns=['epoch', 'loss', 'val_loss', 'train_time'])

        dump_model_history['epoch'] = range(loss.size)
        dump_model_history['loss'] = loss
        dump_model_history['val_loss'] = val_loss

        if self.time_callback.times is not None:
            dump_model_history['train_time'] = self.time_callback.times

        dump_model_history.to_csv(self.saving_path + 'training_history.csv', index=False)


class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
