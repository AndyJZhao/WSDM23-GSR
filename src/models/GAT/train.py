import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

from utils.early_stopper import EarlyStopping
from utils.evaluation import *
from utils.data_utils import preprocess_data, graph_normalization

import torch.nn.functional as F
from time import time
import warnings
from models.GAT.model import GAT
from models.GAT.config import GATConfig
import torch as th
from numpy import mean
import argparse
from utils.util_funcs import exp_init, time_logger, print_log
import dgl
from models.GSR.trainer import FullBatchTrainer
from utils.conf_utils import *


@time_logger
def train_gcn(args):
    exp_init(args.seed, gpu_id=args.gpu)
    # ! config
    cf = GATConfig(args)
    cf.device = th.device("cuda:0" if args.gpu >= 0 else "cpu")

    # ! Load Graph
    g, features, n_feat, cf.n_class, labels, train_x, val_x, test_x = preprocess_data(cf.dataset, cf.train_percentage)
    features = features.to(cf.device)
    g = dgl.add_self_loop(g).to(cf.device)
    supervision = SimpleObject({'train_x': train_x, 'val_x': val_x, 'test_x': test_x, 'labels': labels})

    # ! Train Init
    print(f'{cf}\nStart training..')
    heads = ([cf.n_head] * cf.n_layer) + [cf.n_out_head]
    model = GAT(g, cf.n_layer, n_feat, cf.n_hidden, cf.n_class, heads, F.elu, cf.feat_drop, cf.attn_drop,
                cf.negative_slope)
    model.to(cf.device)
    print(model)
    optimizer = th.optim.Adam(
        model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    if cf.early_stop > 0:
        stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file)
    else:
        stopper = None

    # ! Train
    trainer = FullBatchTrainer(model=model, g=g, cf=cf, features=features,
                               sup=supervision, stopper=stopper, optimizer=optimizer,
                               loss_func=th.nn.CrossEntropyLoss())
    trainer.run()
    trainer.eval_and_save()

    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    # dataset = 'pubmed'
    # dataset = 'citeseer'
    # dataset = 'cora'
    dataset = 'arxiv'
    # ! Settings
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("-b", "--block_log", action="store_true", help="block log or not")
    parser.add_argument("-t", "--train_percentage", default=10, type=int)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()
    # ! Train
    cf = train_gcn(args)
