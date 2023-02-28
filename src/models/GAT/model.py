"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self,
                 g,
                 n_layer,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        self.n_layer = n_layer
        self.gat_layers = nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False))
        self.bns.append(torch.nn.BatchNorm1d(num_hidden*heads[0]))
        # hidden layers
        for l in range(1, n_layer):
            # due to multi-head, the in_dim = num_hidden * n_head
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual))
            self.bns(torch.nn.BatchNorm1d(num_hidden*heads[l]))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.n_layer):
            h = self.gat_layers[l](g, h).flatten(1)
            h = self.bns[l](h)
            h = self.activation(h)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits.log_softmax(dim=-1)
