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


def Intersection(g1, g2):

    bsrc, bdst = g1.edges()
    bsrc = bsrc.numpy()
    bdst = bdst.numpy()
    assert len(bsrc) == len(bdst)
    b_edges = [(bsrc[i], bdst[i]) for i in range(len(bsrc))]

    src, dst = g2.edges()
    src = src.numpy()
    dst = dst.numpy()
    assert len(src) == len(dst)
    edges = [(src[i], dst[i]) for i in range(len(src))]

    intersection = list(set(edges) & set(b_edges))
    inter_ratio = len(intersection)/len(b_edges)

    return inter_ratio


root_path = '/home/jianan/Desktop/MGSL_new/'
os.chdir(root_path)
graph_folder = f'{root_path}temp/GSR/refined_graphs/'

dataset = 'cora'
train_percentage = 0
g_ori, features, n_feat, n_class, labels, train_x, val_x, test_x = preprocess_data(dataset, train_percentage)
print(g_ori)

#### graph configuration
### best graph
b_add_ratio = 0.4
b_rm_ratio = 0.0
b_intra_weight = 0.75
b_fsim_weight = 0.5
b_p_epochs = 20
b_p_batch_size = 256
bse_seed = 0
b_graph_name = f'PR-_se_seed{bse_seed}_lr0.001_bsz{b_p_batch_size}_pi{b_p_epochs}_encGCN_dec-l2_hidden48-prt_intra_w-{b_intra_weight}_ncek16382_fanout20_40_prdo0_act_Relu_d64_GR-fsim_norm1_fsim_weight{b_fsim_weight}_add{b_add_ratio}_rm{b_rm_ratio}.bin'
b_graph = dgl.load_graphs(f'{graph_folder}{dataset}/{b_graph_name}')[0][0]
print(b_graph.ndata['sim'])

# b_graph = dgl.remove_self_loop(b_graph)
#
# ### compare graph
# add_ratio = 0.4
# rm_ratio = 0.0
# intra_weight = 0.75
# fsim_weight = 0.5
# p_epochs = 20
# p_batch_size = 256
# seed_list = [0, 1, 2, 3, 4]
# intra_list = [0.0, 0.25, 0.5, 0.75, 1.0]
# fsim_list = [0.0, 0.25, 0.5, 0.75, 1.0]
# add_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# epoch_list = [20, 25, 30, 35, 40]
# batch_list = [128, 256, 512]


# diff = {}
# diff['intra'] = []
# for val in intra_list:
#     for seed in seed_list:
#         graph_name = f'PR-_se_seed{seed}_lr0.001_bsz{p_batch_size}_pi{p_epochs}_encGCN_dec-l2_hidden48-prt_intra_w-{intra_weight}_ncek16382_fanout20_40_prdo0_act_Relu_d64_GR-fsim_norm1_fsim_weight{fsim_weight}_add{add_ratio}_rm{rm_ratio}.bin'
#         graph = dgl.load_graphs(f'{graph_folder}{dataset}/{graph_name}')[0][0]
#         graph = dgl.remove_self_loop(graph)
#         inter_ratio = Intersection(b_graph, graph)
#         diff.append(inter_ratio)




