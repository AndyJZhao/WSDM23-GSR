import os
import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import torch as th
import time

from emb.fe_config import FEConfig
from emb.se_config import SEConfig
from emb.GraphSage.train import train_graphsage
from emb.DeepWalk.train import train_deepwalk
from utils.data_utils import min_max_scaling
from utils.util_funcs import time2str


def get_node_property_prior(g, cf, features, labels, train_x):
    '''
    Get structural and feature node property prior embedding
    Parameters
    '''
    start = time.time()

    # ! Generate Feature Meta Structure
    print(f'Loading meta embedding...')

    if not os.path.exists(cf.embedding_file):
        print(f'Embedding file {cf.embedding_file} not exist, start training')
        cf.load_device = th.device('cpu')
        emb = dict()
        print('Generating the feature embedding...')
        emb['F'] = generate_feature(g.to(cf.load_device), features.to(cf.load_device), cf.feat_order, cf)
        # ! Generate Structure Meta Structure
        print('\nGenerating the structure embedding...')
        emb['S'] = generate_structure(g.to(cf.load_device), cf)

        print(f'Training embedding used time: {time2str(time.time() - start)}')
        th.save(emb, cf.embedding_file)
        print(f'Embedding file saved')
    else:
        # emb = th.load(cf.embedding_file)
        emb = th.load(cf.embedding_file, map_location=cf.device)
        print('Load embedding file successfully')

    g.ndata['F'] = emb['F']
    # g.ndata['F'] = features  # Fixme
    # g.ndata['F'][0, 444] = 1
    g.ndata['S'] = emb['S']
    for _ in ['F', 'S']:  # Assert no zero row or column
        assert sum(g.ndata[_].sum(1) == 0) == 0
        assert sum(g.ndata[_].sum(0) == 0) == 0

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
        cf.device = th.device("cuda:0")
    else:
        cf.device = th.device('cpu')

    emb_total = train_deepwalk(cf, g).to(exp_cf.device)

    return emb_total
