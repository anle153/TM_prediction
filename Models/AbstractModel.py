import os
import time

import keras.callbacks as keras_callbacks
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json



class AbstractModel(object):

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')
        self._model_kwargs = kwargs.get('model')

        self.saving_path = self._train_kwargs.get('log_dir')

        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        self.callbacks_list = []

        if not os.path.isdir(self.saving_path):
            os.makedirs(self.saving_path)
        self.checkpoints = ModelCheckpoint(
            self.saving_path + "best_model.hdf5",
            monitor='val_loss', verbose=1,
            save_best_only=True,
            mode='auto', period=1)
        self.callbacks_list = [self.checkpoints]

        self.earlystop = EarlyStopping(monitor='val_loss', patience=50,
                                       verbose=1, mode='auto')
        self.callbacks_list.append(self.earlystop)

        self.time_callback = TimeHistory()
        self.callbacks_list.append(self.time_callback)


    def plot_training_history(self, model_history):
        pass

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
