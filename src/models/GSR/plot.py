import os.path as osp
import pickle
import sys
import os
sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import os
import sys
import numpy as np
import pandas as pd
import dgl
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from utils.data_utils import preprocess_data
root_path = '/home/jianan/Desktop/MGSL-new/'
os.chdir(root_path)
result_folder = f'{root_path}temp/GSR/'
graph_folder = f'{root_path}temp/GSR/refined_graphs/'
EVAL_METRIC = 'test_acc'
plt.rc('font', family='Times New Roman')

FONT_SIZE = 18


def homo(graph, labels):
    graph = dgl.remove_self_loop(graph)
    src, dst = graph.edges()
    intra_num = (labels[src]==labels[dst]).long().sum().numpy()
    inter_num = (labels[src] != labels[dst]).long().sum().numpy()

    return int(intra_num), int(inter_num)

dataset = 'arxiv'
train_percentage = 0
g_ori, features, n_feat, n_class, labels, train_x, val_x, test_x = preprocess_data(dataset, train_percentage)

homo_dict = {}
homo_dict['intra'] = np.zeros((11, 11)).astype(int)
homo_dict['inter'] = np.zeros((11, 11)).astype(int)
homo_dict['homo_ratio'] = np.zeros((11, 11)).astype(float)
filebase = 'PR-_lr0.001_bsz1024_pi2_encGCN_dec-l2_hidden48-prt_intra_w-0.5_ncek16382_fanout5_10_prdo0_act_Elu_d256_GR-fsim_norm1_fsim_weight0.0_add0.0_rm0.25.bin'
filebase = filebase.split('fsim_weight')[0]

fsim_weight = 0.0
add_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rm_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


for aid in range(len(add_list)):
    add = add_list[aid]
    for rid in range(len(rm_list)):
        rm = rm_list[rid]
        fileupdate = f'fsim_weight{fsim_weight}_add{add}_rm{rm}.bin'
        filename = filebase + fileupdate
        print(filename)
        if add == 0.0 and rm == 0.0:
            intra, inter = homo(g_ori, labels)
        elif os.path.exists(f'{graph_folder}{dataset}/{filename}'):
            g_new = dgl.load_graphs(f'{graph_folder}{dataset}/{filename}')[0][0]
            intra, inter = homo(g_new, labels)
        else:
            print(filename)
            raise ValueError('Files Not Found')

        homo_ratio = intra / (intra + inter)
        homo_dict['intra'][aid, rid] = intra
        homo_dict['inter'][aid, rid] = inter
        homo_dict['homo_ratio'][aid, rid] = homo_ratio

        print(intra, inter, homo_ratio)

pickle.dump(homo_dict, open(f'{result_folder}{dataset}_homo.pkl', 'wb'))