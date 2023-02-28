import torch as th
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
from utils.data_utils import mp_to_relations
from itertools import combinations
import torch.nn as nn
import os
from openke.model import *
from openke.strategy import NegativeSampling
from openke.loss import *
from tqdm import tqdm
from utils.data_utils import *
from utils.debug_utils import *
from dgl.nn import EdgeWeightNorm


#
class NP_Encoder(nn.Module):
    def __init__(self, g, cf):
        """ Encode graph to embeddings."""
        super(NP_Encoder, self).__init__()
        # ! Load configs
        hge_config = ['device', 'n_feat', 'n_hidden', 'enc_conf', 'decoder_layer', 'decoder_n_hidden']
        self.__dict__.update(cf.get_sub_conf(hge_config))
        self.views = views = cf.views
        # ! Init Encoder
        self.encoder = nn.ModuleDict({
            src_view: self._get_encoder(g, src_view) for src_view in views})
        self.decoder = nn.ModuleDict(
            {f'{src_view}->{tgt_view}':
                 MLP(n_layer=self.decoder_layer,
                     input_dim=self.n_hidden,
                     n_hidden=self.decoder_n_hidden,
                     output_dim=self.n_hidden, dropout=0,
                     activation=nn.ELU(),
                     )
             for tgt_view in views
             for src_view in views if src_view != tgt_view})

    def _get_encoder(self, g, src_view):
        if self.enc_conf.split('_')[0] == 'MLP':
            return MLP(n_layer=int(self.enc_conf.split('_')[1]),
                       input_dim=g.ndata[src_view].shape[1],
                       n_hidden=int(self.enc_conf.split('_')[2]),
                       output_dim=self.n_hidden, dropout=0,
                       activation=nn.ELU(),
                       )
        elif self.enc_conf.split('_')[0] == 'GCN':
            return

    def forward(self, g, nodes):
        # ! Step 1: Encode node properties to embeddings
        Z = {src_view: encoder(g.ndata[src_view][nodes])
             for src_view, encoder in self.encoder.items()}
        # ! Step 2: Decode embeddings if inter-view
        Z.update({dec: decoder(Z[dec[0]])
                  for dec, decoder in self.decoder.items()})
        return Z


class MGSL_PretrainedFeatEmb(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, g, cf):
        # ! Initialize variabless
        super(MGSL_PretrainedFeatEmb, self).__init__()
        self.__dict__.update(cf.model_conf)
        self.device = cf.device
        self.g = g

        # ! GSL Modules
        self.np_encoder = NP_Encoder(g, cf)
        self.ew_calculator = nn.ModuleDict({
            g_name: EdgeWeightGenerator(
                g.ndata[g_name[-1]].shape[1] if cf.ew_mode[:2] == 'F0' else cf.n_hidden, cf.num_head, cf.device)
            for g_name in self.g_names if g_name != 'ori'
        })
        if self.gf_mode[:2] == 'cf':  # Fuse at classification level
            # since there is an output layer, the number of message passing is cf.n_layer - 1
            self.gcn = GCN(cf.n_feat, cf.n_hidden, cf.n_class, cf.n_layer - 1)
        elif self.gf_mode[:2] == 'ef':  # Fuse at embedding level
            self.gcn = GCN_for_emb(cf.n_feat, cf.n_hidden, cf.n_layer)

        # ! Graph Fusion Modules
        if self.gf_mode == 'ef_att':
            self.predictor = SemanticAttentionClf(cf.n_hidden, cf.att_n_hidden, cf.n_class)
        elif self.gf_mode == 'cf_att':
            self.predictor = SemanticAttention(cf.n_class, cf.att_n_hidden)
        elif self.gf_mode == 'ef_chn':
            self.predictor = ChannelAttentionClf(len(cf.g_names), cf.n_hidden, cf.n_class)
        elif self.gf_mode == 'cf_chn':
            self.predictor = ChannelAttention(len(cf.g_names))
        elif self.gf_mode == 'ef_mlp':
            emb_dim = len(self.g_names) * cf.n_hidden
            self.predictor = MLP(n_layer=cf.mlp_n_layer, input_dim=emb_dim, output_dim=cf.n_class,
                                 n_hidden=cf.mlp_n_hidden, dropout=cf.mlp_dropout)

    def att_weight_to_dict(self, weights):
        if weights is None:
            return {}
        else:
            w = weights.flatten().detach().cpu().numpy().tolist()
            return dict(zip(self.g_names, [round(_, 4) for _ in w]))

    def coarse_filter_cand_graphs(self):
        '''
        Find the neighborhood candidates for each candidate graph
        :param g: DGL graph
        '''

        def edge_lists_to_set(_):
            return set(list(map(tuple, _)))

        # ! Get Node Property
        # Freeze parameters except for finetune
        if self.ew_mode != 'NPFine_SimEW':  #
            self.np_encoder.eval()
            for param in self.np_encoder.parameters():
                param.requires_grad = False
        # Get node property by np_encoder for pretrain and finetune
        if self.ew_mode[:2] == 'NP':
            emb = self.np_encoder(self.g, self.g.nodes())
            np_for_coarse_filtering = {_: emb[_].detach().to(self.device) for _ in ['F', 'S']}
            if self.ew_mode.split('_')[0] == 'NPFreeze':  # The node property is freezed
                self.node_prop = np_for_coarse_filtering
        else:  # Use feature and structural prior embedding as node property
            self.node_prop = np_for_coarse_filtering = {_: self.g.ndata[_] for _ in ['F', 'S']}

        # ! Filter Graphs
        self.graphs = {}
        for g_name in self.g_names:
            if g_name == 'ori':
                self.graphs[g_name] = None  # No need to coarse filter for original graph
            else:
                # ? TODO Whether norm should be applied
                property = np_for_coarse_filtering[g_name]
                adj = matrix_rowwise_cosine_sim(property, property)

                if self.filter_mode.split('_')[0] == 'TRS':
                    adj = (adj - adj.min()) / (adj.max() - adj.min())
                    threshold = float(self.filter_mode.split('_')[1])
                    self.graphs[g_name] = dgl.add_self_loop(
                        dgl.graph(th.where(adj > threshold), num_nodes=self.g.num_nodes()))
                elif self.filter_mode.split('_')[0] == 'TopK':
                    K = int(self.filter_mode.split('_')[1])
                    row_id = [i for _ in range(K) for i in range(adj.shape[0])]
                    col_id = th.topk(adj, K).indices.flatten().cpu().numpy().tolist()
                    self.graphs[g_name] = dgl.add_self_loop(
                        dgl.graph((row_id, col_id), num_nodes=self.g.num_nodes())).to(self.device)
                    # print(f'{g_name} Neighbors generated: {set(col_id)}')
                elif self.filter_mode.split('_')[0] == 'Mod':
                    edges = edge_lists_to_set(
                        np.column_stack([_.cpu().numpy() for _ in self.g.edges()]).tolist())
                    low_k, high_k = [int(float(_) * self.g.num_edges())
                                     for _ in self.filter_mode.split('_')[1:]]
                    # ! Remove the lowest K existing edges.
                    # Other edges are guaranteed not to be selected with similairty 99
                    if low_k > 0:
                        low_candidate_mat = th.ones_like(adj) * 99
                        low_candidate_mat[self.g.edges()] = adj[self.g.edges()]
                        low_inds = edge_lists_to_set(global_topk(low_candidate_mat, k=low_k, largest=False))
                        edges -= low_inds
                    # ! Add the highest K from non-existing edges.
                    # Exisiting edges and shouldn't be selected
                    if high_k > 0:
                        adj.masked_fill_(th.eye(adj.shape[0]).to(self.device).bool(), -1)
                        adj[self.g.edges()] = -1
                        high_inds = edge_lists_to_set(global_topk(adj, k=high_k, largest=True))
                        edges |= high_inds
                    row_id, col_id = map(list, zip(*list(edges)))
                    # print(f'Graph-{g_name}:high_inds:{list(high_inds)[:10]},low_inds{list(low_inds)[:10]}')
                    self.graphs[g_name] = dgl.add_self_loop(
                        dgl.graph((row_id, col_id), num_nodes=self.g.num_nodes())).to(self.device)
        # ! Initialize Edge Norm Module
        if self.edge_weight_norm is not None:
            self.ew_norm = EdgeWeightNorm(self.edge_weight_norm)

    def _prediction_layer(self, emb_list):
        if self.emb_norm == 'row1':
            emb_list = [F.normalize(emb, p=1, dim=1) for emb in emb_list]
        elif self.emb_norm == 'row2':
            emb_list = [F.normalize(emb, p=2, dim=1) for emb in emb_list]
        elif self.emb_norm == 'col1':
            emb_list = [F.normalize(emb, p=1, dim=0) for emb in emb_list]
        elif self.emb_norm == 'col2':
            emb_list = [F.normalize(emb, p=2, dim=0) for emb in emb_list]

        if self.gf_mode == 'ef_mean':
            return th.mean(th.stack(emb_list, dim=0), dim=1), None
        elif self.gf_mode == 'ef_mlp':
            return self.predictor(th.cat(emb_list, dim=1)), None
        elif self.gf_mode in ['ef_att', 'cf_att']:
            return self.predictor(th.stack(emb_list, dim=1))
        elif self.gf_mode in ['ef_chn', 'cf_chn']:
            return self.predictor(th.stack(emb_list, dim=0))

    def forward(self, features):
        '''
        Generate graph structure first then message passing
        :param features:
        :return:
        '''
        emb = []
        if self.ew_mode.split('_')[0] == 'NPFine':
            node_property = self.np_encoder(self.g, self.g.nodes())
        for g_name, g_temp in self.graphs.items():
            print(g_name)
            if g_name == 'ori':
                emb.append(self.gcn(self.g, features))
            else:
                if self.ew_mode.split('_')[1] == 'SimEW':  # Similarity based Edge Weight
                    # ! Edge weight calculation
                    row_id, col_id = g_temp.edges()
                    np = node_property[g_name] if self.ew_mode.split('_')[0] == 'NPFine' else self.node_prop[g_name]
                    edge_weight = self.ew_calculator[g_name](np, row_id, col_id)
                    edge_weight[edge_weight <= 0] = 1e-8
                    print(f'NodeProperty{g_name} {np[0, :3]}, edgeweight[:5]={edge_weight[:5]}')
                    if self.edge_weight_norm:
                        edge_weight = self.ew_norm(g_temp, edge_weight)
                    # ! Message Passing
                    emb_ = self.gcn(g_temp, features, edge_weight)
                else:  # Equal Edge Weight
                    emb_ = self.gcn(g_temp, features, edge_weight=None)
                emb.append(emb_)
        logits, weight = self._prediction_layer(emb)
        return logits, self.att_weight_to_dict(weight)


class EdgeWeightGenerator(nn.Module):
    """
    Generate graph using similarity.
    """

    def __init__(self, dim, num_head=2, dev=None):
        super(EdgeWeightGenerator, self).__init__()
        self.metric_layer = nn.ModuleList([MetricCalcLayer(dim) for _ in range(num_head)])
        self.num_head = num_head
        self.dev = dev

    def forward(self, mat, left_id, right_id):
        """

        Args:
            left_h: left_node_num * hidden_dim/feat_dim
            right_h: right_node_num * hidden_dim/feat_dim
        Returns:

        """
        edge_weight = th.zeros_like(left_id, dtype=th.float).to(self.dev)
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](mat[left_id])
            weighted_right_h = self.metric_layer[i](mat[right_id])
            edge_weight += F.cosine_similarity(weighted_left_h, weighted_right_h)

        return edge_weight / self.num_head


class MLP(nn.Module):
    def __init__(self, n_layer, input_dim, output_dim, n_hidden, dropout=0.5, activation=th.nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(input_dim, n_hidden))
        # hidden layers
        for i in range(n_layer - 2):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, output_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        return

    def forward(self, input):
        h = input
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            # h = F.relu(layer(h))
            h = self.activation(layer(h))
        return h


class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        return h * self.weight


class ChannelAttention(nn.Module):

    def __init__(self, num_channel):
        super(ChannelAttention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight

    def forward(self, input):
        weights = F.softmax(self.weight, dim=0)
        return torch.sum(input * weights, dim=0), weights


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta_ = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta_ * z).sum(1), beta  # (N, D * K), (M,1)


class SemanticAttentionClf(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(SemanticAttentionClf, self).__init__()
        self.sem_att_layer = SemanticAttention(in_size, hidden_size)
        self.output_layer = nn.Linear(in_size, out_size)

    def forward(self, z):
        emb, weight = self.sem_att_layer(z)
        return self.output_layer(emb), weight


class ChannelAttentionClf(nn.Module):
    def __init__(self, n_channel, in_size, out_size):
        super(ChannelAttentionClf, self).__init__()
        self.chn_att_layer = ChannelAttention(n_channel)
        self.output_layer = nn.Linear(in_size, out_size)

    def forward(self, z):
        emb, weight = self.chn_att_layer(z)
        return self.output_layer(emb), weight


class GCN_for_emb(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers=2, activation=F.relu, dropout=0.5):
        '''
        # Modified from the DGL GCN Implementation
        n_layer times of message passing
        PLEASE NOTE that the output layer is removed, since we only want to generate the embeddings and fuse them using MLP
        prediction layer outside this GCN function afterwards.
        are shared by

        '''
        super(GCN_for_emb, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features, edge_weight=None):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=edge_weight)
        return h


class GCN(nn.Module):
    # Original implementation of GCN by dgl. number of message passing: n_layer (input + hidden layer) + 1 (output layer)
    def __init__(self, in_feats, n_hidden, n_classes, n_layers=2, activation=F.relu, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features, edge_weight=None):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=edge_weight)
        return h
