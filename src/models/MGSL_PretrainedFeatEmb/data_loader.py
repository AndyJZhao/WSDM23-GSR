import pickle
import dgl
import dgl.data
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sparse
import torch as th
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import utils.util_funcs as uf
from utils.conf_utils import Dict2Config
from utils.proj_settings import *
from utils.data_utils import *
from bidict import bidict
import random
from itertools import combinations_with_replacement, product, chain
from tqdm import tqdm
from utils.data_utils import row_norm
from utils.util_funcs import *


def batcher(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def adj_batch_loader(shape, batch_size, is_symmetric):
    'Load NxN combinations'

    if is_symmetric:  # Since the types of symmetric relations are the same, the order doesn't matters
        adj_combinations = list(combinations_with_replacement(list(range(shape[0])), r=2))
    else:  # Since the types of symmetric relations are the same, the order matters
        adj_combinations = list(product(list(range(shape[0])), list(range(shape[1]))))

    return batcher(adj_combinations, batch_size)

def mlp_pretrain_collate_fn():  # finetune (with label)
    '''
    Convert batch input to OpenKE https://github.com/thunlp/OpenKE format.
    '''

    def batcher_dev(batch):
        ssl_nodes, supervised_nodes = lot_to_tol(batch)
        # length = batch_size * n_ssl_relations * pos_ent_per_rel * (1+ neg_rate)
        return ssl_nodes, supervised_nodes

    return batcher_dev


def mlp_pretrain_loader(g, train_nodes, batch_size, p_epochs):
    # Load batch_size labeled and random nodes for semi-supervised and self-supervised respectively
    ssl_q_list = np.random.choice(g.nodes().cpu().numpy(), p_epochs * batch_size, replace=True)
    ssl_k_list = dgl.sampling.random_walk(g.cpu(), ssl_q_list, length=1)[0][:, 1].numpy()
    supervised_nodes = np.random.choice(train_nodes.cpu(), p_epochs * batch_size, replace=True)
    return batcher(list(zip(ssl_q_list, ssl_k_list, supervised_nodes)), batch_size)
