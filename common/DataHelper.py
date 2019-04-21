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
GEANT_XML_PATH = '/home/anle/GEANT_data/traffic-matrices/'


def get_row(xmlRow):
    """
    Parse Traffic matrix row from XLM element "src"
    :param xmlRow: XML element "src"
    :return: Traffic row corresponds to the measured traffic of a source node.
    """
    TM_row = np.zeros(shape=23)
    for dst in xmlRow.iter('dst'):
        dstId = int(dst.get('id'))
        TM_row[dstId - 1] = float(dst.text)

    return TM_row


def create_Geant2d(datapath=GEANT_XML_PATH):
    TM = np.empty((0, MATRIX_DIM * MATRIX_DIM))

    if os.path.exists(datapath):
        list_files = os.listdir(datapath)
        list_files = sorted(list_files, key=lambda x: x[:-4])

        TM_t = np.zeros(shape=(23, 23))

        for file in list_files:
            if file.endswith(".xml"):
                print('----- Load file: %s -----' % file)
                data = et.parse(datapath + '/' + file)
                root = data.getroot()

                for src in root.iter('src'):
                    TM_row = get_row(xmlRow=src)
                    TM_t[int(src.get('id')) - 1] = TM_row

                aRow = TM_t.reshape(1, MATRIX_DIM * MATRIX_DIM)
                TM = np.concatenate([TM, aRow], axis=0)

    np.save(Config.DATA_PATH + 'Geant2d.npy', TM)

    return
