import xml.etree.ElementTree as et

from FlowClassification.SpatialClustering import *
from common import Config


########################################################################################################################
#                             Loading ABILENE Traffic trace into Traffic Matrix                                        #

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

    if os.path.exists(datapath):
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

