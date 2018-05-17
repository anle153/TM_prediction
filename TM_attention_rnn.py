from RNN import *
from Utils.DataHelper import *
from Utils.DataPreprocessing import *
import datetime

# PATH CONFIGURATION
FIGURE_DIR = './figures/'
MODEL_RECORDED = './Model_Recorded/'
HOME = os.path.expanduser('~')

# DATASET CONFIGURATION
DATASET = ['Geant', 'Geant_noise_removed', 'Abilene', 'Abilene_noise_removed']
GEANT = 0
GEANT_NOISE_REMOVED = 1
ABILENE = 2
ABILENE_NOISE_REMOVE = 3

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
HIDDEN_DIM = 300
LOOK_BACK = 26
N_EPOCH = 50
BATCH_SIZE = 128

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


def predict_with_loss_test_inputshpae(test_set, look_back, rnn_model, sampling_ratio):
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
    rnn_input = test_set[0:look_back, :].T
    # Results TM
    ret_tm = rnn_input
    # The TF array for random choosing the measured flows
    tf = np.array([True, False])
    measured_matrix = np.array([True] * look_back * test_set.shape[1])
    measured_matrix = np.reshape(measured_matrix, (look_back, test_set.shape[1]))

    # Predict the TM from time slot look_back
    for tslot in range(0, test_set.shape[0] - look_back, 1):
        # print ('--- Predict at timeslot %i ---' % tslot)

        # Create 3D input for rnn
        rnn_input = ret_tm[:, tslot:]
        rnn_input = np.reshape(rnn_input, (rnn_input.shape[0], rnn_input.shape[1], 1))
        # Get the TM prediction of next time slot
        predictX = rnn_model.predict(rnn_input)
        predictX = np.reshape(predictX, (predictX.shape[0], predictX.shape[1]))

        # Using part of current prediction as input to the next estimation
        # Randomly choose the flows which is measured (using the correct data from test_set)

        # boolean array(1 x n_flows):for choosing value from predicted data
        sampling = np.expand_dims(np.random.choice(tf,
                                                   size=(test_set.shape[1]),
                                                   p=[sampling_ratio, 1 - sampling_ratio]), axis=0)
        measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
        # invert of sampling: for choosing value from the original data
        inv_sampling = np.invert(sampling)

        pred_input = predictX.T * inv_sampling
        measured_input = test_set[tslot + look_back, :] * sampling

        # Merge value from pred_input and measured_input
        new_input = pred_input.T + measured_input.T
        # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

        # Concatenating new_input into current rnn_input
        ret_tm = np.concatenate([ret_tm, new_input], axis=1)

    return ret_tm.T, measured_matrix


def training_rnn(raw_data, dataset_name):
    test_name = 'attention_lstm'
    splitting_ratio = [0.7, 0.3]
    look_back = 26
    errors = np.empty((0, 3))
    n_unique = 10000

    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)
    train_set = remove_zero_flow(train_set)

    ################################################################################################################
    #                                         For testing Flows Clustering                                         #

    seperated_train_set, centers_train_set = mean_std_flows_clustering(train_set)
    # training_set, train_scalers, train_cluster_lens = different_flows_scaling(seperated_train_set[1:],
    #                                                                           centers_train_set[1:])

    seperated_test_set, centers_test_set = mean_std_flows_clustering(test_set)

    rnn = RNN(
        saving_path=model_recorded_path + 'hidden_%i_lookback_%i_epoch_%i/' % (HIDDEN_DIM, look_back, N_EPOCH),
        raw_data=raw_data,
        look_back=look_back,
        n_epoch=N_EPOCH,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM)
    rnn.seq2seq_modelContruction(n_timesteps=look_back, n_features=n_unique)
    rnn.model.compile(loss='categorical_crossentropy', optimizer='adam')

    train_path = HOME + '/TM_estimation_dataset/' + dataset_name + '/'
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    for cluster in range(1, len(seperated_train_set)):
        print('--- Training Model ---')
        processing_data = seperated_train_set[cluster]
        print('--- Processing data shape' + str(processing_data.shape))

        time_period = 288 / 4
        n_time_period = int(processing_data.shape[0] / time_period)

        processing_data = remove_zero_flow(processing_data)
        _min = processing_data.min()
        _max = processing_data.max()
        unique_step = (_max - _min) / (n_unique - 1)

        for t in range(0, n_time_period):
            for flowID in range(0, processing_data.shape[1], 4):

                _to = flowID + 4 if (flowID + 4) < processing_data.shape[1] else processing_data.shape[1]

                trainX_filename = 'TrainX_' + test_name + '_' + dataset_name + '_' + str(
                    splitting_ratio[0]) + '_look_back_%i_cluster_%i_period_%i_flow_%i.npy' % (
                                      look_back, cluster, t, flowID)

                trainY_filename = 'TrainY_' + test_name + '_' + dataset_name + '_' + str(
                    splitting_ratio[0]) + '_look_back_%i_cluster_%i_period_%i_flow_%i.npy' % (
                                      look_back, cluster, t, flowID)

                trainX = []
                trainY = []
                if not os.path.exists(train_path + trainX_filename):
                    print('--- Can not find dataset at: ' + train_path + trainX_filename)
                    print('--- Create TrainX, TrainY ---')
                    print('--- ')

                    training_set = parallel_one_hot_encoder(
                        data=processing_data[t * time_period: (t + 1) * time_period, flowID:_to],
                        min_v=_min,
                        max_v=_max,
                        unique_step=unique_step,
                        n_unique=n_unique,
                        nproc=8)

                    trainX, trainY = parallel_create_xy_set_encoded(training_set, look_back, nproc=8)

                    print('--- Save TrainX, TrainY to %s ---' % train_path + trainX_filename)
                    np.save(train_path + trainX_filename, trainX)
                    np.save(train_path + trainY_filename, trainY)
                else:
                    print('--- Load TrainX, TrainY from file %s --- ' % train_path + trainX_filename)
                    trainX = np.load(train_path + trainX_filename)
                    trainY = np.load(train_path + trainY_filename)

                print(trainX.shape)

                print('--- Compile model for test_inputshape %s --- ' % (rnn.saving_path))

                rnn.model.fit(trainX,
                              trainY,
                              epochs=rnn.n_epoch,
                              batch_size=rnn.batch_size,
                              verbose=1)

                trainX = None
                trainY = None

            rnn.save_model_to_disk()

    print(rnn.model.summary())
    return


def testing_rnn(raw_data, look_back, dataset_name, sampling_ratio=0.2):
    test_name = 'attention_lstm'
    splitting_ratio = [0.7, 0.3]
    errors = np.empty((0, 3))
    n_unique = 10000

    model_recorded_path = HOME + '/TM_estimation_models/Model_Recorded/' + dataset_name + '/' + test_name + '/'

    rnn = RNN(
        saving_path=model_recorded_path + 'hidden_%i_lookback_%i_epoch_%i/' % (HIDDEN_DIM, look_back, N_EPOCH),
        raw_data=raw_data,
        look_back=look_back,
        n_epoch=N_EPOCH,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM)
    rnn.seq2seq_modelContruction(n_timesteps=look_back, n_features=n_unique)
    rnn.model.compile(loss='categorical_crossentropy', optimizer='adam')


    if os.path.isfile(path=rnn.saving_path + 'model.json'):
        rnn.load_model_from_disk(model_json_file='model.json',
                                 model_weight_file='model.h5')
    else:
        print('--- Saved Model not found')
        return
    train_set, test_set = prepare_train_test_set(data=raw_data,
                                                 sampling_itvl=5,
                                                 splitting_ratio=splitting_ratio)

    test_set = remove_zero_flow(test_set)
    print(test_set.shape)

    seperated_test_set, centers_test_set = mean_std_flows_clustering(test_set)

    predicted_traffic_matrix = np.empty((test_set.shape[0], 0))

    for cluster in range(1, len(seperated_test_set)):

        time_period = 288 / 4
        n_time_period = int(raw_data.shape[0] / time_period)
        processing_data = seperated_test_set[cluster]
        processing_data = remove_zero_flow(processing_data)

        _min = processing_data.min()
        _max = processing_data.max()
        _unique_step = (_max - _min) / (n_unique - 1)

        tf = np.array([True, False])
        measured_matrix = np.array([True] * look_back * processing_data.shape[1])
        measured_matrix = np.reshape(measured_matrix, (look_back, processing_data.shape[1]))

        ret_tm = np.empty((0, processing_data.shape[1]))
        for ts in range(processing_data.shape[0] / 3):
            print('------- Time slot: %i -------' % ts)
            pred_ts_tm = np.empty((1, 0))
            for flow_id in range(0, processing_data.shape[1], 8):
                _to = flow_id + 8 if (flow_id + 8) < processing_data.shape[1] else processing_data.shape[1]

                rnn_input = processing_data[ts:(ts + look_back + 1), flow_id:_to]
                encode_rnn_input = parallel_one_hot_encoder(data=rnn_input, max_v=_max, min_v=_min,
                                                            unique_step=_unique_step,
                                                            n_unique=n_unique, nproc=8)
                encoded_inputX, encode_inputY = parallel_create_xy_set_encoded(data=encode_rnn_input,
                                                                                   look_back=look_back, nproc=8)

                # encode_rnn_input = np.reshape(encode_rnn_input,
                #                               (1, encode_rnn_input.shape[0], encode_rnn_input.shape[2]))
                encoded_predX = rnn.model.predict(encoded_inputX)

                decoded_predX = one_hot_decoder(encoded_predX, unique_step=_unique_step)
                print('decoded_predX ' + str(decoded_predX.shape))

                pred_ts_tm = np.concatenate([pred_ts_tm, np.expand_dims(decoded_predX[-1, :], axis=0)], axis=1)

                # pred_ts_tm = np.concatenate([pred_ts_tm, decoded_predX[-1]], axis=1)

            sampling = np.expand_dims(np.random.choice(tf,
                                                       size=(processing_data.shape[1]),
                                                       p=[sampling_ratio, 1 - sampling_ratio]), axis=0)
            measured_matrix = np.concatenate([measured_matrix, sampling], axis=0)
            # invert of sampling: for choosing value from the original data
            inv_sampling = np.invert(sampling)

            pred_input = pred_ts_tm * inv_sampling
            measured_input = processing_data[ts + look_back, :] * sampling

            # Merge value from pred_input and measured_input
            new_input = pred_input + measured_input
            # new_input = np.reshape(new_input, (new_input.shape[0], new_input.shape[1], 1))

            # Concatenating new_input into current rnn_input
            ret_tm = np.concatenate([ret_tm, new_input], axis=0)

            # visualize_results_by_timeslot(y_true=np.expand_dims(processing_data[ts + look_back, :], axis=0),
            #                               y_pred=new_input,
            #                               measured_matrix=measured_matrix,
            #                               description=test_name + '_sampling_%f' % sampling_ratio,
            #                               saving_path=HOME + '/TM_estimation_figures/' + dataset_name + '/')

            arg_sampling = np.argwhere(sampling == True).T[1]
            nmse = normalized_mean_squared_error(y_true=np.expand_dims(processing_data[ts + look_back, :], axis=0), y_hat=new_input)
            plt.title('TM prediction at time slot %i' % ts + '\n NMSE: %.3f' % nmse)
            plt.plot(np.expand_dims(processing_data[ts + look_back, :], axis=0).squeeze(), label='Original Data')
            plt.plot(new_input.squeeze(), label='Prediction Data')
            plt.legend()
            plt.xlabel('FlowID')
            plt.ylabel('Mbps')
            # Mark the measured data in the predicted data as red start
            plt.plot(arg_sampling, new_input.squeeze()[arg_sampling], 'r*')
            plt.savefig(HOME + '/TM_estimation_figures/' + dataset_name + '/' + test_name + '_sampling_%f' % sampling_ratio+'Timeslot_%i.png' % ts)
            plt.show()
            plt.close()

        predicted_traffic_matrix = np.concatenate([predicted_traffic_matrix, ret_tm], axis=1)


if __name__ == "__main__":
    np.random.seed(10)

    # Abilene24s_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene24s.csv')
    # Abilene_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene.csv')
    Abilene1_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene1.csv')
    # Abilene3_data = load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene3.csv')

    # run_test_deep_rnn(raw_data=Abilene24s_data, hidden_dim=50)
    # test_rnn_inputshape(Abilene24s_data)
    # running_test_lookback(Abilene24s_data, train_mode=LOOK_BACK_MODE)
    # testing_multiple_rnn(Abilene24s_data)
    # run_test_hidden_layer(raw_data=Abilene24s_data)
    # run_test_bidirect_rnn(raw_data=Abilene24s_data, hidden_dim=300, dataset_name='Abilene3')
    training_rnn(raw_data=Abilene1_data, dataset_name='Abilene1')
    testing_rnn(raw_data=Abilene1_data, dataset_name='Abilene1', look_back=26)
