import os
import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import torch as th
import torch.nn as nn
import dgl
import numpy as np
import dgl.function as fn
import torch.nn.functional as F
import time
import argparse

from utils.data_utils import preprocess_data, graph_normalization
from emb.fe_config import FEConfig
from emb.se_config import SEConfig
from emb.LINE_dgl.line import LineTrainer
from emb.GraphSage.train import train_graphsage
from utils.data_utils import min_max_scaling, row_norm


def gen_meta_sources(g, cf, features, labels, train_x):
    start = time.time()
    # ! Generate Label Meta Structure
    assert g.num_nodes() == len(labels)
    assert sum(labels < 0) == 0
    # Ground truth, used only on
    # l_tensor = th.sparse_coo_tensor((range(g.num_nodes()), labels), th.ones(g.num_nodes()),device=cf.device)
    # g.ndata['G'] = l_tensor.to_dense()

    row_id, col_id = train_x.cpu().numpy(), labels[train_x].cpu().numpy()

    l_tensor = th.sparse_coo_tensor((row_id, col_id), th.ones(len(train_x)), (g.num_nodes(), cf.n_class))
    g.ndata['L'] = l_tensor.to_dense().to(cf.device)
    # ! Generate Feature Meta Structure
    print(f'Loading meta embedding...')

    if not os.path.exists(cf.embedding_file):
        print('Embedding file not exist, start training')
        cf.load_device = th.device('cpu')
        emb = dict()
        print('Generating the feature embedding...')
        emb['F'] = generate_feature(g.to(cf.load_device), features.to(cf.load_device), cf.feat_order, cf)
        # ! Generate Structure Meta Structure
        print('\nGenerating the structure embedding...')
        emb['S'] = generate_structure(g.to(cf.load_device), cf)

        print('Training embedding used time: %.2fs' % (time.time() - start))
        th.save(emb, cf.embedding_file)
        print(f'Embedding file saved')
    else:
        # emb = th.load(cf.embedding_file)
        emb = th.load(cf.embedding_file, map_location=cf.device)
        print('Load embedding file successfully')
    # ! Min-Max scaling to [0,1]
    g.ndata['F'] = min_max_scaling(emb['F'])
    g.ndata['S'] = min_max_scaling(emb['S'])
    for _ in ['F', 'S']:  # Assert no zero row or column
        assert sum(g.ndata['F'].sum(1) == 0) == 0
        assert sum(g.ndata['F'].sum(0) == 0) == 0

    return g


def generate_feature(g, features, n_layer, exp_cf):
    cf = FEConfig(exp_cf)
    if exp_cf.gpu >= 0:
        cf.device = th.device("cuda:0")
    else:
        cf.device = th.device('cpu')
    cf.fe_layer = n_layer
    emb_total = train_graphsage(g, features, cf).to(exp_cf.device)

    return emb_total


def generate_structure(g, exp_cf):
    cf = SEConfig(exp_cf)
    if exp_cf.gpu >= 0:
        cf.se_gpus = [exp_cf.gpu]
        cf.se_only_gpu = True
    else:
        cf.se_gpus = [-1]
        cf.se_only_cpu = True

    cf.G = g
    trainer = LineTrainer(cf)
    trainer.train()
    emb_total = trainer.emb.to(exp_cf.device)

    return emb_total
