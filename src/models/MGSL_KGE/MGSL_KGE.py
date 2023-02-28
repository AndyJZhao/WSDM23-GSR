import torch as th
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
from utils.data_utils import mp_to_relations
from itertools import combinations
import torch.nn as nn
import os
from models.MGSL_KGE.data_loader import *
from openke.model import *
from openke.strategy import NegativeSampling
from openke.loss import *
from tqdm import tqdm
from utils.data_utils import min_max_scaling, row_norm, standarize
from utils.debug_utils import *

F.margin_ranking_loss


def check_kge_score(score_mat):
    return sum([sum(score_mat[i, :] > score_mat[i, i]) for i in range(score_mat.shape[0])]) / score_mat.shape[1]


class GraphStructureGenerator(nn.Module):
    # Meta graph structure
    # Modified from TransE implementation by OpenKE (https://github.com/thunlp/OpenKE)

    def __init__(self, g, cf):
        super(GraphStructureGenerator, self).__init__()
        gsg_conf = ['kge_model', 'filter_mode', 'meta_types']
        self.vir_rels, self.ssl_rels, self.mlid2eid, self.rel2rid = init_graph_meta_info(g, cf)
        self.__dict__.update(cf.get_sub_conf(gsg_conf))
        meta_sources_cnt = sum(g.ndata[t].shape[1] for t in self.meta_types)
        meta_relations_cnt = len(self.ssl_rels)
        kge_batch_size = cf.batch_size * cf.pos_ent_per_rel * len(self.ssl_rels)
        if self.kge_model == 'TransE':
            self.KGE = NegativeSampling(
                model=TransE(
                    ent_tot=meta_sources_cnt,
                    rel_tot=meta_relations_cnt,
                    dim=cf.meta_emb_dim,
                    p_norm=1,
                    norm_flag=True),
                loss=MarginLoss(margin=5.0),
                batch_size=kge_batch_size
            )
        elif self.kge_model == 'TransR':
            self.KGE = NegativeSampling(
                model=TransR(
                    ent_tot=meta_sources_cnt,
                    rel_tot=meta_relations_cnt,
                    dim_e=cf.meta_emb_dim,
                    dim_r=cf.meta_emb_dim,
                    p_norm=1,
                    norm_flag=True,
                    rand_init=False),
                loss=MarginLoss(margin=5.0),
                batch_size=kge_batch_size
            )

    def forward(self, batch):
        loss = self.KGE(batch)
        return loss


class MGSL_KGE(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, g, cf):
        # ! Initialize variabless
        super(MGSL_KGE, self).__init__()
        self.__dict__.update(cf.model_conf)
        self.device = cf.device
        self.g = g

        # ! Initialize Graph Meta Structures
        self.vir_rels, self.ssl_rels, self.mlid2eid, self.rel2rid = init_graph_meta_info(g, cf)
        self.gs_generator = GraphStructureGenerator(g, cf)

        if self.gf_mode[:2] == 'cf':  # Fuse at classification level
            # since there is an output layer, the number of message passing is cf.n_layer - 1
            self.gcn = GCN(cf.n_feat, cf.n_hidden, cf.n_class, cf.n_layer - 1)
        elif self.gf_mode[:2] == 'ef':  # Fuse at embedding level
            self.gcn = GCN_for_emb(cf.n_feat, cf.n_hidden, cf.n_layer)

        # ! Graph Fusion Modules
        if self.gf_mode == 'ef_att':
            self.predictor = nn.Sequential(
                SemanticAttention(cf.n_hidden),
                nn.Linear(cf.n_hidden, cf.n_class)
            )
        elif self.gf_mode == 'cf_att':
            self.predictor = SemanticAttention(cf.n_class)
        elif self.gf_mode == 'ef_mlp':
            emb_dim = (len(self.gsmp_list) + 1) * cf.n_hidden
            self.predictor = MLP(n_layer=cf.mlp_n_layer, input_dim=emb_dim, output_dim=cf.n_class,
                                 n_hidden=cf.mlp_n_hidden, dropout=cf.mlp_dropout)

    def _neighbor_predict(self, batch_i, batch_j, r):
        # Convert adj index to entity ids
        hid = [self.mlid2eid[r[0]][_] for _ in batch_i]
        rid = [self.rel2rid[r] for _ in range(len(batch_i))]
        tid = [self.mlid2eid[r[1]][_] for _ in batch_j]
        hid, rid, tid = [th.Tensor(_).type(th.long).to(self.device) for _ in [hid, rid, tid]]
        batch_data = {'batch_h': hid, 'batch_r': rid, 'batch_t': tid, 'mode': 'normal'}
        return self.gs_generator.KGE.model(batch_data)

    def _get_gmvr(self):
        def score_to_adj(adj, adj_norm):
            if adj_norm:
                return row_norm(1 - min_max_scaling(adj, type='global'))
            else:
                return 1 - min_max_scaling(adj, type='global')

        gmvr = {}
        # ! Indexing meta relationships
        for t in self.meta_types:
            if self.adj_norm:
                gmvr['N' + t] = row_norm(self.g.ndata[t])
                gmvr[t + 'N'] = row_norm(self.g.ndata[t].t())
            else:
                gmvr['N' + t] = self.g.ndata[t]
                gmvr[t + 'N'] = self.g.ndata[t].t()

        # ! Generate virtual relationships
        for r in self.vir_rels:
            is_symmetric = r[0] == r[-1]
            adj = th.zeros((self.g.ndata[r[0]].shape[1], self.g.ndata[r[1]].shape[1]), device=self.device)
            for batch in adj_batch_loader(adj.shape, self.batch_size, is_symmetric):
                batch_i, batch_j = zip(*batch)
                adj[batch_i, batch_j] = self._neighbor_predict(batch_i, batch_j, r)
            if is_symmetric:  # Generate for r
                adj += adj.t()  # if r is symmetric, the lower triangular values is calculated by upper triangular values.
                gmvr[r] = score_to_adj(adj, self.adj_norm)
            else:  # Generate for r and inverse r
                gmvr[r] = score_to_adj(adj, self.adj_norm)
                gmvr[r[1] + r[0]] = score_to_adj(adj.t(), self.adj_norm)
            # print(f'Top 5 neighbors of {r}: {th.topk(gmvr[r], 5).indices.flatten().unique()}')
        return gmvr

    def process_attention_weight(self, weights):
        if weights is None:
            return {}
        else:
            return dict(zip(self.gsmp_list, weights.flatten().detach().cpu().numpy().tolist()))

    def coarse_filter_cand_graphs(self):
        '''
        Find the neighborhood candidates for each candidate graph
        Find the neighborhood candidates for each candidate graph
        :param g: DGL graph
        '''
        self.eval()
        # ! Create graph meta virtual relations
        gmvr = self._get_gmvr()

        # ! Construct the commuting matrix
        self.graphs = {}
        for g_name in self.gsmp_list:
            if g_name == 'ori':
                self.graphs[g_name] = None  # No need to coarse filter for original graph
            else:
                gsmp = mp_to_relations(g_name)
                for rel_id, r in enumerate(gsmp):
                    if r[0] == 'N' and rel_id == 0:  # [Start] Graph Meta Structure
                        adj = gmvr[r].clone()
                    else:  # [Middle or End] Graph Meta Structure or Virtual Relations
                        adj = th.matmul(adj, gmvr[r])
                    print(f'{len(topk_uniques(gmvr[r].t()))} Top 5 uniques of GMVR {r}: {topk_uniques(gmvr[r].t())}')
                    print(f'{len(topk_uniques(adj))} current TopK neigbors for {g_name[:rel_id + 2]}:'
                          f'{topk_uniques(adj)}')

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
            return self.predictor(torch.stack(emb_list, dim=1))

    def forward(self, features):
        '''
        Generate graph structure first then message passing
        :param features:
        :return:
        '''
        gmvr = self._get_gmvr()
        emb = []
        for g_name, g_temp in self.graphs.items():
            if g_name == 'ori':
                emb.append(self.gcn(self.g, features))
            else:
                # ! Graph Generation
                row_id, col_id = g_temp.edges()
                gsmp = mp_to_relations(g_name)

                ori_implementation = False
                ori_implementation = True
                if not ori_implementation:
                    for rel_id, r in enumerate(gsmp):
                        if rel_id == 0:  # [Start] Graph Meta Structure
                            edge_weight = gmvr[r][row_id].clone()
                        elif rel_id == len(gsmp) - 1:  # [End] [Pairwise Dot Product] Graph Meta Structure
                            edge_weight = th.sum(edge_weight * gmvr[r].t()[col_id], dim=1)
                        else:  # [Middle] Graph Meta Structure or Virtual Relations
                            edge_weight = th.matmul(edge_weight, gmvr[r])
                else:
                    for rel_id, r in enumerate(gsmp):
                        if r[0] == 'N' and rel_id == 0:  # [Start] Graph Meta Structure
                            edge_weight = self.g.ndata[r[1]][row_id].clone()
                        elif rel_id == len(gsmp) - 1:  # [End] [Hadamard Product] Graph Meta Structure
                            edge_weight = th.sum(edge_weight * self.g.ndata[r[0]][col_id], dim=1)
                        elif 'N' in r:  # [Middle] Graph Meta Structure
                            graph_meta_rel = self.g.ndata[r[1]] if r[0] == 'N' else self.g.ndata[r[0]].t()
                            edge_weight = th.matmul(edge_weight, graph_meta_rel)
                        elif 'N' not in r:  # [Middle] Graph Virtual Relations
                            edge_weight = th.matmul(edge_weight, gmvr[r])
                if self.edge_weight_norm:
                    for id in g_temp.nodes():
                        edge_weight[row_id == id] /= sum(edge_weight[row_id == id])
                # ! Message Passing
                emb_ = self.gcn(g_temp, features, edge_weight)
                emb.append(emb_)
        logits, weight = self._prediction_layer(emb)
        return logits, self.process_attention_weight(weight)


class MLP(nn.Module):
    def __init__(self, n_layer, input_dim, output_dim, n_hidden, dropout=0.5):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(input_dim, n_hidden))
        # hidden layers
        for i in range(n_layer - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.output_layer = nn.Linear(n_hidden, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        return

    def forward(self, input):
        h = input
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = F.relu(layer(h))
        return h


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
