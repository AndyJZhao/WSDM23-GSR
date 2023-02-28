"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=None))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=None))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = dropout

    def forward(self,g, x):
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(g, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](g, x)
        return x.log_softmax(dim=-1)
