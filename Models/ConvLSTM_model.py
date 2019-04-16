import fnmatch
import os

from keras.layers import *
from keras.layers import Activation
from keras.models import Sequential
from keras.models import model_from_json

from Models.AbstractModel import AbstractModel


class ConvLSTM(AbstractModel):

    def __init__(self, saving_path, input_shape,
                 cnn_layers, a_filters, a_strides, dropouts, kernel_sizes, rnn_dropouts,
                 alg_name=None, tag=None, early_stopping=False, check_point=False):

        super().__init__(alg_name=alg_name, tag=tag, early_stopping=early_stopping, check_point=check_point,
                         saving_path=saving_path)

        if cnn_layers != len(a_filters) or cnn_layers != len(a_strides) or cnn_layers != len(
                rnn_dropouts) or cnn_layers != len(dropouts):
            print('|--- [ConvLSTM] Error: size of filters and/or strides need to be equal to the cnn_layers!')
            exit(-1)

        self.cnn_layers = cnn_layers
        self.a_filters = a_filters
        self.a_strides = a_strides
        self.dropout = dropouts
        self.rnn_dropout = rnn_dropouts
        self.kernel_sizes = kernel_sizes

        self.n_timsteps = input_shape[0]
        self.height = input_shape[1]
        self.weight = input_shape[2]
        self.depth = input_shape[3]
        self.saving_path = saving_path
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        self.model = Sequential()

        for cnn_layer in range(self.cnn_layers):
            self.model.add(ConvLSTM2D(filters=self.a_filters[cnn_layer],
                                      kernel_size=self.kernel_sizes[cnn_layer],
                                      strides=[1, 1],
                                      padding='same',
                                      dropout=self.dropout[cnn_layer],
                                      input_shape=(None, self.weight, self.height, self.depth),
                                      return_sequences=True,
                                      recurrent_dropout=self.rnn_dropout[cnn_layer]))

            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))

        self.model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                              activation='relu',
                              padding='same', data_format='channels_last'))

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
        print('----> [RNN-save_model_to_disk]--- RNN model was saved at %s ---' % (
                self.saving_path + model_json_filename))

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

        print('----> [CNN_LSTM-load_model_from_disk]--- Models has been loaded from %s' % (
                self.saving_path + model_json_file))
        return True

    def load_weights_model(self, path, weight_file):

        if not os.path.isfile(path + weight_file):
            print('----> [RNN-load_weights_model] --- File %s not found ---' % (path + weight_file))
            return False
        else:
            print('----> [RNN-load_weights_model] --- Load weights from ' + path + weight_file)
            self.model.load_weights(path + weight_file)
            return True

    def load_model_from_check_point(self, _from_epoch=0, weights_file_type='hdf5'):

        if weights_file_type == 'h5':
            if os.path.exists(self.saving_path):

                list_weights_files = fnmatch.filter(os.listdir(self.saving_path), '*.h5')

                if len(list_weights_files) == 0:
                    print(
                        '----> [CNN_LSTM-load_model_from_check_point] --- Found no weights file at %s---' % self.saving_path)
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
                print('----> [CNN_LSTM-load_model_from_check_point] --- Models saving path dose not exist')
                return -1
        else:
            if os.path.exists(self.saving_path):
                list_files = fnmatch.filter(os.listdir(self.saving_path), '*.hdf5')

                if len(list_files) == 0:
                    print('----> [CNN_LSTM-load_model_from_check_point] --- Found no weights file at %s---'
                          % self.saving_path)
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
                print('----> [CNN_LSTM-load_model_from_check_point] --- Models saving path dose not exist')
                return -1