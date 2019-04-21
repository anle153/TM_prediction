import fnmatch
import os

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json


def plot_training_history(alg_name, tag, saving_path, model_history):
    plt.plot(model_history.history['mean_absolute_error'], label='mae')
    plt.plot(model_history.history['val_mean_absolute_error'], label='val_mae')
    plt.legend()
    plt.savefig(saving_path + '[MAE]{}-{}.png'.format(alg_name, tag))
    plt.close()


class AbstractModel(object):

    def __init__(self, saving_path, alg_name=None, tag=None, early_stopping=False, check_point=False, **kwargs):
        self.alg_name = alg_name
        self.tag = tag
        self.saving_path = os.path.expanduser(saving_path)
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        self.callbacks_list = []

        self.checkpoints_path = self.saving_path + '/checkpoints/{}-{}/'.format(self.alg_name, self.tag)

        if check_point:
            if not os.path.isdir(self.checkpoints_path):
                os.makedirs(self.checkpoints_path)
            self.checkpoints = ModelCheckpoint(
                self.checkpoints_path + "weights-{epoch:02d}-{val_acc:.2f}.hdf5",
                monitor='val_loss', verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode='auto', period=1)
            self.callbacks_list = [self.checkpoints]
        if early_stopping:
            self.earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50,
                                           verbose=1, mode='auto')
            self.callbacks_list.append(self.earlystop)

    def save(self, model_json_filename='trained_model.json', model_weight_filename='trained_model.h5'):
        # Save model to dir + record_model/model_train_[%training_set].json

        model_json = self.model.to_json()
        with open(self.saving_path + model_json_filename, "w") as json_file:
            json_file.write(model_json)
            json_file.close()

        # Serialize weights to HDF5
        self.model.save_weights(self.saving_path + model_weight_filename)

    def load(self, model_json_file='trained_model.json', model_weight_file='trained_model.h5'):

        assert os.path.isfile(self.saving_path + model_json_file) & os.path.isfile(
            self.saving_path + model_weight_file)

        json_file = open(self.saving_path + model_json_file, 'r')
        model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(model_json)
        self.model.load_weights(self.saving_path + model_weight_file)

        return True

    def load_trained_model(self, path, weight_file):

        if not os.path.isfile(path + weight_file):
            print('----> [RNN-load_weights_model] --- File %s not found ---' % (path + weight_file))
            return False
        else:
            print('----> [RNN-load_weights_model] --- Load weights from ' + path + weight_file)
            self.model.load_weights(path + weight_file)
            return True

    def load_model_from_check_point(self, _from_epoch=0, weights_file_type='h5'):

        if weights_file_type == 'h5':
            if os.path.exists(self.checkpoints_path):

                list_weights_files = fnmatch.filter(os.listdir(self.checkpoints_path), '*.h5')

                if len(list_weights_files) == 0:
                    print('|--- Found no weights file at %s---' % self.checkpoints_path)
                    return -1

                list_weights_files = sorted(list_weights_files, key=lambda x: int(x.split('-')[1]))
                weights_file_name = ''
                model_file_name = ''
                epoch = -1
                if _from_epoch:
                    for _weights_file_name in list_weights_files:
                        epoch = int(_weights_file_name.split('-')[1])
                        if _from_epoch == epoch:
                            weights_file_name = _weights_file_name
                            model_file_name = 'model-' + str(epoch) + '-.json'
                            break
                else:
                    # Get the last check point
                    weights_file_name = list_weights_files[-1]
                    epoch = int(weights_file_name.split('-')[1])
                    model_file_name = 'model-' + str(epoch) + '-.json'

                if self.load(model_weight_file=weights_file_name, model_json_file=model_file_name):
                    return epoch
                else:
                    return -1
            else:
                print('----> [RNN-load_model_from_check_point] --- Models saving path dose not exist')
                return -1
        else:
            if os.path.exists(self.checkpoints_path):
                list_files = fnmatch.filter(os.listdir(self.checkpoints_path), '*.hdf5')

                if len(list_files) == 0:
                    print(
                        '|--- Found no weights file at %s---' % self.checkpoints_path)
                    return -1

                list_files = sorted(list_files, key=lambda x: int(x.split('-')[1]))

                weights_file_name = ''
                epoch = -1
                if _from_epoch:
                    for _weights_file_name in list_files:
                        epoch = int(_weights_file_name.split('-')[1])
                        if _from_epoch == epoch:
                            weights_file_name = _weights_file_name
                            break
                else:
                    # Get the last check point
                    weights_file_name = list_files[-1]
                    epoch = int(weights_file_name.split('-')[1])

                if self.load_trained_model(path=self.checkpoints_path, weight_file=weights_file_name):
                    return epoch
                else:
                    return -1
            else:
                print('----> [RNN-load_model_from_check_point] --- Models saving path dose not exist')
                return -1

    def plot_training_history(self, model_history):
        plot_training_history(alg_name=self.alg_name,
                              tag=self.tag,
                           saving_path=self.saving_path,
                           model_history=model_history)
