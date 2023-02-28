import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
import utils.util_funcs as uf

import torch.nn.functional as F
import torch
import time
import warnings
from emb.DeepWalk.model import gen_dw_emb
import torch as th
import argparse

from sklearn.linear_model import LogisticRegression


def train_deepwalk(cf, g):
    g = g.to(cf.device)
    adj = g.adj(scipy_fmt='coo')# .todense()

    emb_mat = gen_dw_emb(adj, number_walks=cf.se_num_walks, walk_length=cf.se_walk_length, window=cf.se_window_size,
                         size=cf.se_n_hidden,
                         workers=cf.se_num_workers)

    emb_mat = torch.FloatTensor(emb_mat)
    return emb_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    dataset = 'cora'
    # dataset = 'pubmed'
    # dataset = 'citeseer'
    # ! Settings
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("-b", "--block_log", action="store_true", help="block log or not")
    parser.add_argument("-t", "--train_percentage", default=10, type=int)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()
    # ! Train
    cf = train_deepwalk(args)
