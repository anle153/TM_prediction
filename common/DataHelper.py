import datetime
import xml.etree.ElementTree as et
from math import sqrt, log

import scipy.io as sio
from scipy.signal import argrelextrema
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from tensorflow.python.client import device_lib

from FlowClassification.SpatialClustering import *
from common import Config



########################################################################################################################
#                             Loading ABILENE Traffic trace into Traffic Matrix                                        #
#                                             Number of node: 12                                                       #
########################################################################################################################


ABILENE24_DIM = 144


def convert_abilene_24(path_dir='/home/anle/Documents/sokendai/research/TM_estimation_RNN/Dataset'
                                '/Abilene_24/Abilene/2004/Measured'):
    if os.path.exists(path_dir):
        list_files = os.listdir(path_dir)
        list_files = sorted(list_files, key=lambda x: x[:-4])

        TM = np.empty((12, 12, 0))
        for raw_file in list_files:
            if raw_file.endswith('.dat'):
                print(raw_file)
                _tm = np.genfromtxt(path_dir + '/' + raw_file, delimiter=',')
                _tm = np.expand_dims(_tm, axis=2)
                TM = np.concatenate([TM, _tm], axis=2)

    print('--- Finish load original Abilene3d to csv. Saving at ./Dataset/Abilene3d')
    np.save('./Dataset/Abilene3d', TM)


def load_abilene_3d(path_dir='/home/anle/Documents/sokendai/research/TM_estimation_RNN/Dataset'
                             '/Abilene_24/Abilene/2004/Measured'):
    if os.path.exists(path_dir):
        list_files = os.listdir(path_dir)
        list_files = sorted(list_files, key=lambda x: x[:-4])

        TM = np.empty((12, 12, 0))
        for raw_file in list_files:
            if raw_file.endswith('.dat'):
                print(raw_file)
                _tm = np.genfromtxt(path_dir + '/' + raw_file, delimiter=',')
                _tm = np.expand_dims(_tm, axis=0)
                TM = np.concatenate([TM, _tm], axis=0) if TM.size else _tm

    print('--- Finish converting Abilene24 to csv. Saing at ./Dataset/Abilene24_3.csv')
    np.save(HOME + '/TM_estimation_dataset/Abilene24_3d/Abilene24_3d', TM)


def load_Abilene_dataset_from_matlab(path='./Dataset/abilene.mat'):
    """
    Load Abilene from original matlab file
    :param path: dataset path
    :return:
    """
    if path_exist(path):
        # ...
        data = sio.loadmat(path)
        X = data['X']
        A = data['A']
        odnames = data['odnames']
        edgenames = data['edgenames']
        return X
    else:
        return None, None, None, None


def load_Abilene_dataset_from_csv(csv_file_path='./Dataset/Abilene.csv'):
    """
    Load Abilene dataset from csv file. If file is not found, create the one from original matlab file and remove noise
    :param csv_file_path:
    :return: A traffic matrix (m x k)
    """
    if not os.path.exists(csv_file_path):
        print('--- %s not found. Create csv file from original matlab file ---' % csv_file_path)
        abilene_data = np.asarray(load_Abilene_dataset_from_matlab('./Dataset/SAND_TM_Estimation_Data.mat')) / 1000000
        # noise_removed(data=abilene_data, sampling_interval=5, threshold=30)
        np.savetxt(csv_file_path, abilene_data, delimiter=',')
        return abilene_data
    else:
        print('--- Load dataset from %s' % csv_file_path)
        return np.genfromtxt(csv_file_path, delimiter=',')


def create_abilene_data_3d(path):
    tm_3d = np.zeros(shape=(2016 * 24, 12, 12))
    for i in range(24):
        print('Read file X{:02d}'.format(i + 1))
        raw_data = np.genfromtxt(path + 'X{:02d}'.format(i + 1), delimiter=' ')
        tm = raw_data[:, range(0, 720, 5)].reshape((2016, 12, 12))
        tm_3d[i * 2016: (i + 1) * 2016, :, :] = tm

    np.save(Config.DATA_PATH + 'Abilene.npy', tm_3d)


def create_abilene_data_2d(path):
    tm_2d = np.zeros(shape=(2016 * 24, 144))
    for i in range(24):
        print('Read file X{:02d}'.format(i + 1))
        raw_data = np.genfromtxt(path + 'X{:02d}'.format(i + 1), delimiter=' ')
        tm = raw_data[:, range(0, 720, 5)]
        tm_2d[i * 2016: (i + 1) * 2016, :] = tm

    np.save(Config.DATA_PATH + 'Abilene2d.npy', tm_2d)

########################################################################################################################
#                             Loading GEANT Traffic trace into Traffic Matrix from XML files                           #
#                                                 Number of node: 23                                                   #
########################################################################################################################

MATRIX_DIM = 23
GEANT_XML_PATH = './GeantDataset/traffic-matrices-anonymized-v2/traffic-matrices'


def get_row(xmlRow):
    """
    Parse Traffic matrix row from XLM element "src"
    :param xmlRow: XML element "src"
    :return: Traffic row corresponds to the measured traffic of a source node.
    """
    TM_row = [0] * MATRIX_DIM
    for dst in xmlRow.iter('dst'):
        dstId = int(dst.get('id'))
        TM_row[dstId - 1] = float(dst.text)

    return TM_row


def load_Geant_from_xml(datapath=GEANT_XML_PATH):
    TM = np.empty((0, MATRIX_DIM * MATRIX_DIM))

    if path_exist(datapath):
        list_files = os.listdir(datapath)
        list_files = sorted(list_files, key=lambda x: x[:-4])

        for file in list_files:
            if file.endswith(".xml"):
                print('----- Load file: %s -----' % file)
                data = et.parse(datapath + '/' + file)
                root = data.getroot()

                TM_t = []
                for src in root.iter('src'):
                    TM_row = get_row(xmlRow=src)
                    TM_t.append(TM_row)

                aRow = np.asarray(TM_t).reshape(1, MATRIX_DIM * MATRIX_DIM)
                TM = np.concatenate([TM, aRow]) if TM.size else aRow

    return TM


def load_Geant_from_csv(csv_file_path='./Dataset/Geant_noise_removed.csv'):
    """

    :param csv_file_path:
    :return:
    """
    if os.path.exists(csv_file_path):
        return np.genfromtxt(csv_file_path, delimiter=',')
    else:
        print('--- Find not found. Create Dataset from XML file ---')
        data = load_Geant_from_xml(datapath=GEANT_XML_PATH) / 1000
        noise_removed(data=data, sampling_interval=15, threshold=30)
        np.savetxt(csv_file_path, data, delimiter=",")
        return data


########################################################################################################################
#                                                 Data visualization                                                   #
########################################################################################################################


def visualize_retsult_by_flows(y_true,
                               y_pred,
                               sampling_itvl,
                               measured_matrix=[],
                               saving_path='/home/anle/TM_estimation_figures/',
                               description='',
                               visualized_day=-1,
                               show=False):
    """
    Visualize the original flows and the predicted flows over days
    :param y_true: (numpy.ndarray) the measured TM
    :param y_pred: (numpy.ndarray) the predicted TM
    :param sampling_itvl: (int) sampling interval between each sampling
    :param measured_matrix: (numpy.ndarray) identify which elements in the predicted TM are predicted using RNN
    :param saving_path: (str) path to saved figures directory
    :param description: (str) (optional) the description of this visualization
    :return:
    """

    # Get date-time when visualizing, create dir corresponding to the date-time and the description.
    import datetime
    now = datetime.datetime.now()
    description = description + '_' + str(now)
    if not os.path.exists(saving_path + description + '/'):
        os.makedirs(saving_path + description + '/')

    # Calculate no. time slots within a day and the no. days over the period
    n_ts_day = 24 * (60 / sampling_itvl)
    n_days = int(y_true.shape[0] / n_ts_day)

    # Calculate the nmse and plot both original and predicted data of each day by flow.
    path = saving_path + description + '/'

    for day in range(n_days):
        if (visualized_day != -1 and visualized_day == day) or visualized_day == -1:
            if not os.path.exists(path + 'Day%i/' % day):
                os.makedirs(path + 'Day%i/' % day)
            for flowID in range(y_true.shape[1]):
                print('--- Visualize flow %i in day %i' % (flowID, day))
                upperbound = (day + 1) * n_ts_day if (day + 1) * n_ts_day < y_true.shape[0] else y_true.shape[0]
                y1 = y_true[day * n_ts_day:upperbound, flowID]
                y2 = y_pred[day * n_ts_day:upperbound, flowID]
                sampling = measured_matrix[day * n_ts_day:upperbound, flowID]
                arg_sampling = np.argwhere(sampling == True).squeeze(axis=1)

                rmse_by_day = rmse_tm_prediction(y_true=np.expand_dims(y1, axis=1), y_pred=np.expand_dims(y2, axis=1))

                plt.title('Flow %i prediction result - Day %i \n RMSE: %f' % (flowID, day, rmse_by_day))
                plt.plot(y1, label='Original Data')
                plt.plot(y2, label='Prediction Data')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Mbps')
                # Mark the measured data in the predicted data as red start
                plt.plot(arg_sampling, y2[arg_sampling], 'r*')
                plt.savefig(path + 'Day%i/' % day + '%i.png' % flowID)
                if show:
                    plt.show()
                plt.close()


def plot_flow_acf(data):
    path = '/home/anle/TM_estimation_figures/ACF/'
    if not os.path.exists(path):
        os.makedirs(path)

    for flowID in range(data.shape[1]):
        print('--- Plotting acf of flow %i' % flowID)
        acf_plt = plot_acf(x=data[:, flowID], lags=288 * 3)
        plt.show()
        # acf_plt.savefig(path+'acf_flow_%i.png'%flowID)


def remove_zero_flow(data, eps=0.001):
    means = np.mean(data, axis=0)
    non_zero_data = data[:, means > eps]

    return non_zero_data


def get_max_acf(data, interval=5):
    day_size = 24 * (60 / interval)

    flows_acf = []
    for flow_id in range(data.shape[1]):
        flow_acf = acf(data[:, flow_id], nlags=day_size * 3)
        arg_local_max = argrelextrema(flow_acf, np.greater)
        flow_acf_local_max_index = np.argmax(flow_acf[arg_local_max[0]])
        flows_acf.append(arg_local_max[0][flow_acf_local_max_index])
        plt.plot(flow_acf)
        plt.plot(arg_local_max[0][flow_acf_local_max_index], flow_acf[arg_local_max[0][flow_acf_local_max_index]], 'r*')
        plt.show()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
