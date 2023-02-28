import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
import utils.util_funcs as uf

from utils.early_stopper import EarlyStopping
from utils.evaluation import *
from utils.data_utils import preprocess_data, graph_normalization
from utils.util_funcs import exp_init, time_logger, print_log

import torch.nn as nn
import torch.nn.functional as F
import time
import dgl
import dgl.function as fn
import warnings
from emb.GraphSage.model import SAGE
from emb.fe_config import FEConfig
from emb.GraphSage.Sampler import NegativeSampler
import torch as th
import argparse
import sklearn.linear_model as lm
import sklearn.metrics as skm
import numpy as np


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss


def compute_acc(emb, train_nids, val_nids, train_labels, val_labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    train_nids = train_nids.numpy()
    train_labels = train_labels.numpy()
    val_nids = val_nids.numpy()
    val_labels = val_labels.numpy()
    # test_nids = test_nids.numpy()
    # test_labels = test_labels.numpy()

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)
    pred = lr.predict(emb)

    acc_train = skm.accuracy_score(train_labels, pred[train_nids])
    acc_val = skm.accuracy_score(val_labels, pred[val_nids])
    f1_train = skm.f1_score(train_labels, pred[train_nids], average='macro')
    f1_val = skm.f1_score(val_labels, pred[val_nids], average='macro')
    f1_micro_train = skm.f1_score(train_labels, pred[train_nids], average='micro')
    f1_micro_val = skm.f1_score(val_labels, pred[val_nids], average='micro')

    return f1_train, f1_val, f1_micro_train, f1_micro_val, acc_train, acc_val


@time_logger
def train_graphsage(g, features, cf):
    # train_x = tvt_nids[0]
    # val_x = tvt_nids[1]
    # test_x = tvt_nids[2]
    # train_y, val_y, test_y = labels[train_x], labels[val_x], labels[test_x]

    n_feat = features.shape[1]
    # Create PyTorch DataLoader for constructing blocks
    num_edges = g.num_edges()
    train_eids = np.arange(num_edges)
    graph_edges = g.edges()
    reverse_eids = g.edge_ids(graph_edges[1], graph_edges[0])

    # Create sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in cf.fe_fan_out.split(',')])
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eids, sampler, exclude='reverse_id',
        reverse_eids=reverse_eids,
        negative_sampler=NegativeSampler(g, cf.fe_num_negs),
        batch_size=cf.fe_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=cf.num_workers)

    # ! Train Init
    cla_loss = CrossEntropyLoss()
    model = SAGE(n_feat, cf.fe_hidden, cf.fe_hidden, cf.fe_layer, F.relu, cf.fe_dropout, cf.fe_aggregator)
    model = model.to(cf.device)
    print(model)
    loss_fcn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.fe_lr, weight_decay=cf.fe_weight_decay)
    stopper = EarlyStopping(patience=cf.fe_early_stop, path=cf.checkpoint_file)

    # ! Train
    dur = []
    for epoch in range(cf.fe_epochs):
        t0 = time.time()
        model.train()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            batch_inputs = features[input_nodes].to(cf.device)
            pos_graph = pos_graph.to(cf.device)
            neg_graph = neg_graph.to(cf.device)
            blocks = [block.int().to(cf.device) for block in blocks]
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # model.eval()
        # with torch.no_grad():
        #     emb, _ = model.inference(g, features, cf)
        #     train_maf1, val_maf1, train_mif1, val_mif1, train_acc, val_acc = compute_acc(emb, train_x, val_x,
        #         train_y, val_y)
        # dur.append(time.time() - t0)
        # print_log(epoch, Time=np.mean(dur), loss=loss.item(), TrainAcc=train_acc, ValAcc=val_acc)

    model.eval()
    with torch.no_grad():
        emb, emb_list = model.inference(g, features, cf)
        emb_total = torch.cat(emb_list, -1).detach()

    return emb


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
    cf = train_graphsage(args)
