import os

import numpy as np

DATA_PATH = '/home/anle/eclipse-workspace-java/Repetita/data/Abilene/lstm/'
NUM_NODE = 12

NODE_MAP = {"0": 11, "1": 9, "2": 1, "3": 6, "4": 8, "5": 10, "6": 7, "7": 5, "8": 0, "9": 4, "10": 3, "11": 2}

pred_data = np.load(os.path.join(DATA_PATH, 'pred.npy'))
gt_data = np.load(os.path.join(DATA_PATH, 'gt.npy'))

pred_demand_file_list = 'Abilene.pd.dm.list'
gt_demand_file_list = 'Abilene.gt.dm.list'

with open(os.path.join(DATA_PATH, pred_demand_file_list), 'w') as pd_list:
    with open(os.path.join(DATA_PATH, gt_demand_file_list), 'w') as gt_list:

        for i in range(pred_data.shape[0]):
            pred_demand_file_name = 'Abilene.{}.pd.dm'.format(i)
            gt_demand_file_name = 'Abilene.{}.gt.dm'.format(i)

            DEMAND = 'DEMANDS {}\n'.format(pred_data.shape[1] - NUM_NODE)
            label = 'label src dest bw\n'

            with open(os.path.join(DATA_PATH, pred_demand_file_name), 'w') as pred_file:
                with open(os.path.join(DATA_PATH, gt_demand_file_name), 'w') as gt_file:
                    pred_file.write(DEMAND)
                    pred_file.write(label)
                    gt_file.write(DEMAND)
                    gt_file.write(label)

                    d = 0

                    for j in range(pred_data.shape[1]):
                        if j % NUM_NODE == int(j / NUM_NODE):
                            continue
                        else:
                            prefix = 'demand_{}'.format(d)
                            src = NODE_MAP.get(str(int(j / NUM_NODE)))
                            dst = NODE_MAP.get(str(j % NUM_NODE))
                            pred_bw = int(pred_data[i, j])
                            gt_bw = int(gt_data[i, j])

                            if pred_bw < 0:
                                pred_bw = 0
                            if gt_bw < 0:
                                gt_bw = 0

                            pre_demand = prefix + ' {} {} {}\n'.format(src, dst, pred_bw)
                            gt_demand = prefix + ' {} {} {}\n'.format(src, dst, gt_bw)

                            pred_file.write(pre_demand)
                            gt_file.write(gt_demand)

                            d += 1

                    gt_file.close()
                pred_file.close()

            if i != 0:
                pd_list.write(os.path.join(DATA_PATH, pred_demand_file_name) + '\n')
                gt_list.write(os.path.join(DATA_PATH, gt_demand_file_name) + '\n')

        gt_list.close()
    pd_list.close()
