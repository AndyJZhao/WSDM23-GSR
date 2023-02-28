import os
import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import torch as th
import dgl
from emb.se_config import SEConfig
from emb.DeepWalk.train import train_deepwalk
import numpy as np
from utils.util_funcs import time_logger
from utils.data_utils import *


@time_logger
def get_structural_feature(g, cf):
    '''
    Get structural node property prior embedding
    '''
    print(f'Loading structural embedding...')

    if not os.path.exists(cf.structural_em_file):
        print(f'Embedding file {cf.structural_em_file} not exist, start training')
        if cf.semb == 'dw':
            cf.load_device = th.device('cpu')
            dw_cf = SEConfig(cf)
            dw_cf.device = th.device("cuda:0") if cf.gpu >= 0 else th.device('cpu')
            emb = train_deepwalk(dw_cf, g).to(cf.device)
        elif cf.semb == 'de':
            emb = DistanceEncoding(g, cf.se_k).to(cf.device)
        else:
            raise ValueError

        th.save(emb, cf.structural_em_file)
        print(f'Embedding file saved')
    else:
        emb = th.load(cf.structural_em_file, map_location=th.device('cpu'))
        # print(emb)
        # print(emb.shape)
        # print(ssss)
        print(f'Load embedding file {cf.structural_em_file} successfully')
    return emb


def get_pretrain_loader(g, cf):
    g = g.remove_self_loop()  # Self loops shan't be sampled
    src, dst = g.edges()
    n_edges = g.num_edges()
    train_seeds = np.arange(n_edges)
    g = dgl.graph((th.cat([src, dst]), th.cat([dst, src])))
    reverse_eids = th.cat([th.arange(n_edges, 2 * n_edges), th.arange(0, n_edges)])
    # Create sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in cf.fan_out.split('_')])
    return dgl.dataloading.EdgeDataLoader(
        g.cpu(), train_seeds, sampler, exclude='reverse_id',
        reverse_eids=reverse_eids,
        batch_size=cf.p_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=cf.num_workers)


def get_stochastic_loader(g, train_nids, batch_size, num_workers):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    return dgl.dataloading.NodeDataLoader(
        g.cpu(), train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)
