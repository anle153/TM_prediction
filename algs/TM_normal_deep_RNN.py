import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from Models.RNN_LSTM import *
from common.DataHelper import *
from Utils.DataPreprocessing import *
from multiprocessing import cpu_count

# PATH CONFIGURATION
FIGURE_DIR = './figures/'
MODEL_RECORDED = './Model_Recorded/'

# DATASET CONFIGURATION
DATASET = ['Geant', 'Geant_noise_removed', 'Abilene', 'Abilene_noise_removed']
GEANT = 0
GEANT_NOISE_REMOVED = 1
ABILENE = 2
ABILENE_NOISE_REMOVE = 3

HOME = os.path.expanduser('~')

# Abilene dataset path.
""" The Abilene dataset contains 
    
    X           a 2016x2016 matrix of flow volumes
    A           a 30x121 matrix of routing of the 121 flows over 30 edge between adjacent nodes in Abilene network.
    odnames     a 121x1 character vector of OD pair names
    edgenames   a 30x1 character vector of node pairs sharing an edge 
"""
ABILENE_DATASET_PATH = './Dataset/SAND_TM_Estimation_Data.mat'

# Geant dataset path.
""" The Geant dataset contains 

    X: a (10772 x 529) matrix of flow volumes
"""
DATAPATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices'
GEANT_DATASET_PATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices/Geant_dataset.csv'
GEANT_DATASET_NOISE_REMOVED_PATH = DATAPATH + '/Gean_noise_removed.csv'
# RNN CONFIGURATION
INPUT_DIM = 100
HIDDEN_DIM = 200
LOOK_BACK = 26
N_EPOCH = 50
BATCH_SIZE = 512

# TRAINING MODES
SPLIT_TRAINING_SET_MODE = 0
HIDDEN_DIM_MODE = 1
BATCH_SIZE_MODE = 2
LOOK_BACK_MODE = 3

TRAINING_MODES = ['Split_dataset', 'hidden_layers', 'batch_size', 'lookback']

# Scaler mode
STANDARDSCALER = 0
MINMAX = 1
NORMALIZER = 2
DATANORMALIZER = 3
SCALER = ['StandardScaler', 'MinMax', 'SKNormalizer', 'MeanStdScale']


def prepare_input_online_prediction(data, n_timesteps, labels):
    labels = labels.astype(int)
    dataX = np.empty(shape=(0, n_timesteps, 2))
    for flow_id in range(data.shape[1]):
        x = data[-n_timesteps:, flow_id]
        label = labels[-n_timesteps:, flow_id]

        sample = np.array([x, label]).T
        sample = np.expand_dims(sample, axis=0)
        dataX = np.concatenate([dataX, sample], axis=0)

    return dataX


def predict_normal_deep_rnn(test_set, n_timesteps, model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)
    n_days = int(test_set.shape[0] / day_size) if (test_set.shape[0] % day_size) == 0 \
        else int(test_set.shape[0] / day_size) + 1

    # Initialize the first input for RNN to predict the TM at time slot look_back
    ret_tm = test_set[0:n_timesteps, :]
    # Results TM
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.ones(shape=(ret_tm.shape[0], ret_tm.shape[1]))

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - n_timesteps, 1):
        # Create 3D input for rnn
        rnn_input = prepare_input_online_prediction(data=ret_tm, n_timesteps=n_timesteps, labels=measured_matrix)

        # Get the TM prediction of next time slot
        predictX = model.predict(rnn_input)

        pred = np.expand_dims(predictX[:, -1, 0], axis=1)

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf,
                                                   size=(test_set.shape[1]),
                                                   p=[sampling_ratio, 1 - sampling_ratio]), axis=0)
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = pred.T * inv_sampling
        measured_input = np.expand_dims(test_set[tslot + n_timesteps, :], axis=0) * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=0)

    return ret_tm, measured_matrix


def traffic_prediction_consecutive_loss(test_set, look_back, rnn_model, sampling_ratio):
    """

    :param test_set: the testing set
    :param look_back: No. of history information data point using as input for RNN
    :param rnn_model: the RNN model
    :return: ret_tm: the prediction TM
    """

    day_size = 24 * (60 / 5)
    n_days = int(test_set.shape[0] / day_size) if (test_set.shape[0] % day_size) == 0 \
        else int(test_set.shape[0] / day_size) + 1

    # Initialize the first input for RNN to predict the TM at time slot look_back
    rnn_input = test_set[0:look_back, :]
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - look_back, 1):
        date = int(tslot / day_size)

        # Create 3D input for rnn
        rnn_input = prepare_input_online_predict(pred_tm=ret_tm, look_back=look_back)
        rnn_input = np.reshape(rnn_input, (rnn_input.shape[0], rnn_input.shape[1], 1))

        # Get the TM prediction of next time slot
        predictX = rnn_model.predict(rnn_input)

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)
        # boolean array(1 x n_flows):for choosing value from predicted data

        # For Consecutive loss
        if (tslot > (date * day_size + 120)) and (tslot <= (date * day_size + 120 + 12 * 4)):
            sampling = np.zeros(shape=(1, measured_matrix.shape[1]), dtype=bool)
            print('Consecutive loss')
        else:
            sampling = np.expand_dims(np.random.choice(tf,
                                                       size=(test_set.shape[1]),
                                                       p=[sampling_ratio, 1 - sampling_ratio]), axis=0)

        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = predictX.T * inv_sampling
        measured_input = test_set[tslot + look_back, :] * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input + measured_input
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=0)

    return ret_tm, measured_matrix


def normal_deep_rnn(raw_data, dataset_name='Abilene24s', n_timesteps=26):
    test_name = 'normal_deep_rnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'
    nprocs = cpu_count()
    random_eps = 1
    sampling_ratioes = [0.3]
    n_layers = 2

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/test_hidden_%i_timesteps_%i/' % (HIDDEN_DIM, n_timesteps)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    for sampling_ratio in sampling_ratioes:
        if not os.path.isfile(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX.npy'):
            if not os.path.exists(HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps):
                os.makedirs(HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps)

            print('|--- Splitting train-test set')
            train_set, test_set = prepare_train_test_set(data=raw_data,
                                                         sampling_itvl=5,
                                                         splitting_ratio=splitting_ratio)

            print("|--- Create XY set.")
            mean_train = np.mean(train_set)
            std_train = np.std(train_set)

            training_set = (train_set - mean_train) / std_train

            trainX, trainY = parallel_create_xy_set_by_random(training_set, n_timesteps, sampling_ratio, random_eps,
                                                              nprocs - 1)

            np.save(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX.npy',
                trainX)
            np.save(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY.npy',
                trainY)
        else:

            print("|---  Load xy set from " + HOME + '/TM_estimation_dataset/' + dataset_name + '/' + dataset_name)

            trainX = np.load(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainX.npy')
            trainY = np.load(
                HOME + '/TM_estimation_dataset/' + dataset_name + '/timesteps_%i/' % n_timesteps + dataset_name + '_trainY.npy')

        print("|--- Creating Deep RNN model")

        rnn = RNN(
            saving_path=model_recorded_path + 'layers_%i_hidden_%i_timesteps_%i_sampling_ratio_%.2f/' % (
                n_layers, HIDDEN_DIM, n_timesteps, sampling_ratio),
            raw_data=raw_data,
            look_back=n_timesteps,
            n_epoch=N_EPOCH,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            check_point=True)

        rnn.seq2seq_deep_model_construction(n_layers=n_layers, n_timesteps=n_timesteps, n_features=2)

        if os.path.isfile(path=rnn.saving_path + 'model.json'):
            rnn.load_model_from_disk(model_json_file='model.json',
                                     model_weight_file='model.h5')

        else:
            print('[%s]---Compile model. Saving path %s --- ' % (test_name, rnn.saving_path))
            from_epoch = rnn.load_model_from_check_point(weights_file_type='hdf5')

            rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

            if from_epoch > 0:
                print('[%s]--- Continue training deep model from epoch %i --- ' % (test_name, from_epoch))

                training_history = rnn.model.fit(trainX,
                                                 trainY,
                                                 epochs=rnn.n_epoch,
                                                 batch_size=rnn.batch_size,
                                                 initial_epoch=from_epoch,
                                                 validation_split=0.25,
                                                 callbacks=rnn.callbacks_list)
                rnn.plot_model_metrics(training_history,
                                       plot_prefix_name='Metrics')
            else:
                print('[%s]--- Training new deep model ' % test_name)

                training_history = rnn.model.fit(trainX,
                                                 trainY,
                                                 epochs=rnn.n_epoch,
                                                 batch_size=rnn.batch_size,
                                                 validation_split=0.25,
                                                 callbacks=rnn.callbacks_list)

                rnn.plot_model_metrics(training_history,
                                       plot_prefix_name='Metrics')

        print(rnn.model.summary())

    return


def normal_deep_rnn_test(raw_data, dataset_name, n_timesteps=26):
    test_name = 'normal_deep_rnn'
    splitting_ratio = [0.8, 0.2]
    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'
    errors = np.empty((0, 3))
    sampling_ratio = 0.3
    n_layers = 3

    figures_saving_path = HOME + '/TM_estimation_figures/' + dataset_name \
                          + '/' + test_name + '/test_hidden_%i_timesteps_%i/' % (HIDDEN_DIM, n_timesteps)

    if not os.path.exists(figures_saving_path):
        os.makedirs(figures_saving_path)

    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    copy_test_set = np.copy(test_set)

    mean_train = np.mean(train_set)
    std_train = np.std(train_set)

    testing_set = np.copy(test_set)
    testing_set = (testing_set - mean_train) / std_train

    copy_testing_set = np.copy(testing_set)

    rnn = RNN(
        saving_path=model_recorded_path + 'layers_%i_hidden_%i_timesteps_%i_sampling_ratio_%.2f/' % (
            n_layers, HIDDEN_DIM, n_timesteps, sampling_ratio),
        raw_data=raw_data,
        look_back=n_timesteps,
        n_epoch=N_EPOCH,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        check_point=True)
    rnn.seq2seq_deep_model_construction(n_layers=n_layers, n_timesteps=n_timesteps, n_features=2)
    list_weights_files_rnn = fnmatch.filter(os.listdir(rnn.saving_path), '*.hdf5')

    if len(list_weights_files_rnn) == 0:
        print('----> [RNN-load_model_from_check_point] --- Found no weights file at %s---' %
              rnn.saving_path)
        return -1

    list_weights_files_rnn = sorted(list_weights_files_rnn, key=lambda x: int(x.split('-')[1]))

    _max_epoch_rnn = int(list_weights_files_rnn[-1].split('-')[1])

    for epoch in range(1, _max_epoch_rnn + 1, 1):

        err_ratio_temp = []

        for running_time in range(10):
            testing_set = np.copy(copy_testing_set)
            test_set = np.copy(copy_test_set)

            rnn.load_model_from_check_point(_from_epoch=epoch, weights_file_type='hdf5')
            rnn.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])

            # print(rnn.model.summary())
            print('|--- Epoch %i - Run time: %i' % (epoch, running_time))

            pred_tm, measured_matrix = predict_normal_deep_rnn(test_set=testing_set,
                                                               n_timesteps=n_timesteps,
                                                               model=rnn.model,
                                                               sampling_ratio=sampling_ratio)

            pred_tm = pred_tm * std_train + mean_train
            measured_matrix = measured_matrix.astype(bool)

            err_ratio_temp.append(error_ratio(y_true=test_set, y_pred=pred_tm, measured_matrix=measured_matrix))
            print('|--- error: %.3f' % error_ratio(y_true=test_set, y_pred=pred_tm, measured_matrix=measured_matrix))

        err_ratio_temp = np.array(err_ratio_temp)
        err_ratio = np.mean(err_ratio_temp)
        err_ratio_std = np.std(err_ratio_temp)

        error = np.expand_dims(np.array([epoch, err_ratio, err_ratio_std]), axis=0)
        errors = np.concatenate([errors, error], axis=0)
        print('|--- Errors by epoches ---')
        print(errors)
        print('|-------------------------------------------------------')

    np.savetxt('./Errors/[%s][layers_%i_hidden_%i_lookback_%i_sampling_ratio_%.2f]Errors_by_epoch.csv' % (
        test_name, n_layers, HIDDEN_DIM, n_timesteps, sampling_ratio), errors, delimiter=',')


if __name__ == "__main__":
    Abilene24 = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24.csv')

    with tf.device('/device:GPU:1'):
        normal_deep_rnn(raw_data=Abilene24, dataset_name='Abilene24')
        # normal_deep_rnn_test(raw_data=Abilene24, dataset_name='Abilene24', n_timesteps=26)
