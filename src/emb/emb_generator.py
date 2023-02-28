import os.path as osp
import sys
sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import torch
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

from sklearn.linear_model import LogisticRegression


def accuracy(pred, target):
    return (pred == target).sum().item() / target.numel()


def generate_featature(g, features, n_layer, exp_cf):

    cf = FEConfig(exp_cf)
    if exp_cf.gpu>=0:
        cf.device = torch.device("cuda:0")
    else:
        cf.device = torch.device('cpu')
    cf.fe_layer = n_layer
    emb_total = train_graphsage(g, features, cf)

    return emb_total

def generate_structure(g, exp_cf):

    cf = SEConfig(exp_cf)
    if exp_cf.gpu>=0:
        cf.se_gpus = [exp_cf.gpu]
        cf.se_only_gpu = True
    else:
        cf.se_gpus = [-1]
        cf.se_only_cpu = True

    cf.G = g
    trainer = LineTrainer(cf)
    trainer.train()
    emb_total = trainer.emb

    return emb_total

def emb_generator(graph, features, K, args):

    start = time.time()

    print('generating the feature embedding...')
    fe_emb = generate_featature(graph, features, K, args)
    graph.ndata['fe'] = fe_emb

    # print('\ngenerating the structure embedding...')
    # se_emb = generate_structure(graph, args)
    # graph.ndata['se'] = se_emb

    print('training embedding used time: %.2fs' % (time.time()-start))
    return graph


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

    train_percentage = 10
    load_device = torch.device('cpu')
    graph, features, n_feat, n_class, labels, train_x, val_x, test_x = preprocess_data(dataset, train_percentage,
                                                                                   load_device)
    graph = graph_normalization(graph, False)
    train_y, val_y, test_y = labels[train_x], labels[val_x], labels[test_x]

    K = 2
    new_graph = emb_generator(graph, features, K, args)

    # emb = new_graph.ndata['se'].numpy()
    # LR = LogisticRegression()
    # LR.fit(emb[train_x.numpy()], train_y.numpy())
    # val_pred = torch.tensor(LR.predict(emb[val_x.numpy()]))
    # test_pred = torch.tensor(LR.predict(emb[test_x.numpy()]))
    #
    # val_acc = accuracy(val_pred, val_y)
    # test_acc = accuracy(test_pred, test_y)
    #
    # print(f'val_acc: {val_acc}')
    # print(f'test_acc: {test_acc}')

    emb = new_graph.ndata['fe'].numpy()[:, 128:]
    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)
    LR = LogisticRegression(multi_class='multinomial', max_iter=10000)
    LR.fit(emb[train_x.numpy()], train_y.numpy())
    val_pred = torch.tensor(LR.predict(emb[val_x.numpy()]))
    test_pred = torch.tensor(LR.predict(emb[test_x.numpy()]))

    val_acc = accuracy(val_pred, val_y)
    test_acc = accuracy(test_pred, test_y)

    print(f'val_acc: {val_acc}')
    print(f'test_acc: {test_acc}')