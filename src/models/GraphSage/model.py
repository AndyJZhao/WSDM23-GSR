"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv
import torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, x):
        # h = self.dropout(x)
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h
