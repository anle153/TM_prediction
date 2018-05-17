from keras.layers import LSTM, Dense, Dropout, Activation, Bidirectional, TimeDistributed, RepeatVector
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os, fnmatch
import numpy as np
from AttentionsLayer.custom_recurrents import AttentionDecoder


class RNN(object):

    def __init__(self, saving_path, early_stopping=False, check_point=False, *args, **kwargs):

        assert 'hidden_dim' in kwargs

        # Parse **kwargs
        for key in ('raw_data', 'look_back', 'n_epoch', 'batch_size', 'hidden_dim', 'scaler'):
            if key in kwargs:
                setattr(self, key, kwargs[key])

        self.trainSet = []
        self.testSet = []
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        # self.scaler = StandardScaler()
        self.scaled_data = []
        self.model = []
        self.saving_path = saving_path
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        self.callbacks_list = []

        if check_point:
            self.checkpoints = ModelCheckpoint(self.saving_path + "weights-{epoch:02d}-{val_acc:.2f}.hdf5",
                                               monitor='val_acc', verbose=1,
                                               save_best_only=False,
                                               save_weights_only=True,
                                               mode='auto', period=10)
            self.callbacks_list = [self.checkpoints]
        if early_stopping:
            self.earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50,
                                           verbose=1, mode='auto')
            self.callbacks_list.append(self.earlystop)

    def modelContruction(self, input_shape, output_dim):
        """
        Construct RNN model from the beginning
        :param input_shape:
        :param output_dim:
        :return:
        """
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_dim, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(output_dim))

    def seq2seq_model_construction(self, n_timesteps, n_features, drop_out=0.2):
        """

        :param n_timesteps:
        :param n_features:
        :return:
        """
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_dim, input_shape=(n_timesteps, n_features), return_sequences=True))
        self.model.add(Dropout(drop_out))
        self.model.add(TimeDistributed(Dense(1)))

    def seq2seq_modelContruction_with_Attention(self, n_timesteps, n_features):
        """
        Construct RNN model from the beginning
        :param input_shape:
        :param output_dim:
        :return:
        """
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_dim, input_shape=(n_timesteps, n_features), return_sequences=True))
        self.model.add(AttentionDecoder(self.hidden_dim, n_features))

    def deep_rnn_io_model_construction(self, input_shape, n_layers=3, drop_out=0.2, output_dim=1, model_type='IO'):
        self.model = Sequential()
        for layer in range(n_layers):

            if layer != (n_layers - 1):
                self.model.add(LSTM(self.hidden_dim, input_shape=input_shape, return_sequences=True))
            else:
                self.model.add(LSTM(self.hidden_dim, input_shape=input_shape, return_sequences=False))
                self.model.add(Dense(output_dim))

            if layer != 0:
                self.model.add(Dropout(drop_out))

    def bidirectional_model_construction(self, input_shape, drop_out=0.3):
        self.model = Sequential()
        self.model.add(
            Bidirectional(LSTM(self.hidden_dim, return_sequences=True), merge_mode='ave', input_shape=input_shape))
        self.model.add(Dropout(drop_out))
        self.model.add(TimeDistributed(Dense(1)))

    def save_model_to_disk(self,
                           model_json_filename='model.json',
                           model_weight_filename='model.h5'):
        """
        Save RNN model to disk
        :param model_json_filename: file name
        :param model_weight_filename: file name
        :param saving_path: path where to save the model
        :return:
        """

        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        # Save model to dir + record_model/model_train_[%training_set].json

        model_json = self.model.to_json()
        with open(self.saving_path + model_json_filename, "w") as json_file:
            json_file.write(model_json)
            json_file.close()

        # Serialize weights to HDF5
        self.model.save_weights(self.saving_path + model_weight_filename)
        self.saving_path = self.saving_path
        print('----> [RNN-save_model_to_disk]--- RNN model was saved at %s ---' % (self.saving_path + model_json_filename))

    def load_model_from_disk(self, model_json_file='model.json', model_weight_file='model.h5'):
        """
        Load RNN model from disk
        :param path_to_file:
        :param model_json_file: model is stored in json format
        :param model_weight_file: model weight
        :return:
        """
        assert os.path.isfile(self.saving_path + model_json_file) & os.path.isfile(self.saving_path + model_weight_file)

        json_file = open(self.saving_path + model_json_file, 'r')
        model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(model_json)
        self.model.load_weights(self.saving_path + model_weight_file)
        self.saving_path = self.saving_path

        print('----> [RNN-load_model_from_disk]--- Model has been loaded from %s' % (self.saving_path + model_json_file))
        return True

    def plot_model_history(self, model_history, show=False):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # summarize history for accuracy
        axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
        axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
        axs[0].legend(['train', 'val'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
        axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
        axs[1].legend(['train', 'val'], loc='best')
        plt.savefig(self.saving_path + 'model_history.png')
        plt.close()

    def load_weights_model(self, path, weight_file):

        if not os.path.isfile(path + weight_file):
            print('----> [RNN-load_weights_model] --- File %s not found ---' % (path + weight_file))
            return False
        else:
            print('----> [RNN-load_weights_model] --- Load weights from ' + path + weight_file)
            self.model.load_weights(path + weight_file)
            return True

    def load_model_from_check_point(self, _from_epoch=0, weights_file_type='h5'):

        if weights_file_type == 'h5':
            if os.path.exists(self.saving_path):

                list_weights_files = fnmatch.filter(os.listdir(self.saving_path), '*.h5')

                if len(list_weights_files) == 0:
                    print('----> [RNN-load_model_from_check_point] --- Found no weights file at %s---' % self.saving_path)
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

                if self.load_model_from_disk(model_weight_file=weights_file_name, model_json_file=model_file_name):
                    return epoch
                else:
                    return -1
            else:
                print('----> [RNN-load_model_from_check_point] --- Model saving path dose not exist')
                return -1
        else:
            if os.path.exists(self.saving_path):
                list_files = fnmatch.filter(os.listdir(self.saving_path), '*.hdf5')

                if len(list_files) == 0:
                    print('----> [RNN-load_model_from_check_point] --- Found no weights file at %s---' % self.saving_path)
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

                if self.load_weights_model(path=self.saving_path, weight_file=weights_file_name):
                    return epoch
                else:
                    return -1
            else:
                print('----> [RNN-load_model_from_check_point] --- Model saving path dose not exist')
                return -1

