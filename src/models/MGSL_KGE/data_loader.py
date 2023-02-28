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


def convert_to_openke_batch(batch, device):
    h, r, t = [th.Tensor(_).type(th.long).to(device=device) for _ in batch]
    return {'batch_h': h, 'batch_r': r, 'batch_t': t, 'mode': 'normal'}


def ke_triple_collate_fn():  # finetune (with label)
    '''
    Convert batch input to OpenKE https://github.com/thunlp/OpenKE format.
    '''

    def batcher_dev(batch):
        triples = pd.concat(batch)
        triples = triples.sort_values(by='ind').reset_index()
        # length = batch_size * n_ssl_relations * pos_ent_per_rel * (1+ neg_rate)
        return [triples[_].to_numpy() for _ in ['h', 'r', 't']]

    return batcher_dev


def worker_init_fn(worker_id):
    "Different workers generate different seeds"
    np.random.seed(th.utils.data.get_worker_info().seed % (2 ** 32))


def init_graph_meta_info(g, cf):
    # ! Generate meta relations
    # Virtual Relations: Used in the graph structure generation process.
    meta_relations = set()
    for gsmp in cf.gsmp_list:
        if gsmp != 'ori':
            assert gsmp[0] == 'N' and gsmp[-1] == 'N', f'gsmp {gsmp} invalid, the start and end types must be N!'
            meta_relations.update(mp_to_relations(gsmp))
    vir_rels = [mr for mr in meta_relations if 'N' not in mr]
    # SSL Relations: used in the KGE pretrain process
    ssl_rels = []
    for t1 in cf.meta_types:
        for t2 in cf.meta_types:
            if t1 == t2 or (f'{t2}{t1}' not in ssl_rels):
                ssl_rels.append(f'{t1}{t2}')

    # ! Meta Node to Entity ID
    t_cnt = {t: g.ndata[t].shape[1] for t in cf.meta_types}
    cur_ind, t_info = 0, {}
    for t in cf.meta_types:
        t_info[t] = {'ind': range(cur_ind, cur_ind + t_cnt[t]), 'cnt': t_cnt[t]}
        cur_ind += t_cnt[t]
    mlid2eid = {t: bidict(dict(zip(list(range(t_cnt[t])), list(t_info[t]['ind']))))
                for t in cf.meta_types}
    # ! Meta Relation to Relation ID
    rel2rid = bidict({r: rid for rid, r in enumerate(ssl_rels)})

    return vir_rels, ssl_rels, mlid2eid, rel2rid


class GraphMetaTripleDataset(th.utils.data.IterableDataset):
    def __init__(self, g, cf):
        # ! Load configs
        super(GraphMetaTripleDataset).__init__()
        gmtl_config = ['neg_rate', 'meta_types', 'batch_size', 'p_iters', 'pos_ent_per_rel']
        self.__dict__.update(cf.get_sub_conf(gmtl_config))
        self.vir_rels, self.ssl_rels, self.mid2eid, self.rel2rid = init_graph_meta_info(g, cf)
        # ! Construct Meta Relationships
        kge_rels = {}
        for r in self.ssl_rels:  # virtual relationship e.g. FF; FS
            vir_rel_temp_list = []
            # * Construct 2-hop triples e.g. FNF; FNS
            if r[0] == r[1] and th.count_nonzero(g.ndata[r[1]]) <= g.ndata[r[1]].shape[0]:
                # Remove self-loop relation generators. For example, if a node with only one label, LNL relationships
                # generate triples with same h and t, therefore h+r-t will be r and head and tail entities
                # cannot be trained.
                print(f'Relation {r[0]}N{r[1]} skipped to avoid generation of invalid triples.')
            else:
                if cf.adj_norm:
                    temp_adj = th.matmul(row_norm(g.ndata[r[0]].t()), row_norm(g.ndata[r[1]]))
                else:
                    temp_adj = th.matmul(g.ndata[r[0]].t(), g.ndata[r[1]])
                temp_adj[temp_adj < 0] = 0
                vir_rel_temp_list.append(F.normalize(temp_adj, p=1, dim=1))
            # * Construct 4-hop triples e.g. FNLNF, FNSNF; FNLNS, FNSNS
            for mid_type in cf.meta_types:  # Take FNLNS as an example, mid_type = L
                if cf.adj_norm:
                    temp_adj = th.matmul(row_norm(g.ndata[r[0]].t()), row_norm(g.ndata[mid_type]))
                    temp_adj = th.matmul(temp_adj, row_norm(g.ndata[mid_type].t()))
                    temp_adj = th.matmul(temp_adj, row_norm(g.ndata[r[1]]))
                else:
                    temp_adj = th.matmul(g.ndata[r[0]].t(), g.ndata[mid_type])  # To mid type FNL = FN * NL
                    temp_adj = th.matmul(temp_adj, g.ndata[mid_type].t())  # To meta nodes FNLN = FNL * LN
                    temp_adj = th.matmul(temp_adj, g.ndata[r[1]])  # To target type FNLNS = FNLN * NS
                temp_adj[temp_adj < 0] = 0
                vir_rel_temp_list.append(F.normalize(temp_adj, p=1, dim=1))
            kge_rels[r] = th.mean(th.stack(vir_rel_temp_list), dim=0)

        # ! Calculate sample probabilities
        # Convert relations to sampling probabilities -> proportional to degree of meta relations
        self.head_prob = {r: F.normalize(th.sum(kge_rels[r], dim=1), p=1, dim=0).cpu().numpy()
                          for r in self.ssl_rels}
        self.tail_prob = {r: F.normalize(th.sum(kge_rels[r], dim=0), p=1, dim=0).cpu().numpy()
                          for r in self.ssl_rels}
        self.tail_prob_given_head_prob = {r: F.normalize(kge_rels[r], p=1, dim=1).cpu().numpy()
                                          for r in self.ssl_rels}

        self.kge_rels = kge_rels

    def _to_eid(self, t, id_list):
        return list(map(self.mid2eid[t].get, id_list))

    def __iter__(self):
        '''
        Sample batch of nodes for pretraining using the degree summation of all relations
        :return: triples
        '''
        for _ in range(self.p_iters * self.batch_size):
            yield self.__getitem__()

    def __getitem__(self):
        '''
        Generate Meta Triples given head type and head entities.
        :return: pandas DataFrame that stores the triples
        '''

        # Notes:
        # The 'cor_type' key is used to generate corrupted negative examples, h for replacing head entity,
        # t for replacing tail entity.
        # The 'ind' key is used to sort the triples so that the neg triples generated from same pos triples are paired.
        def tail_sample_by_head(head_id, r):
            sample_prob = self.tail_prob_given_head_prob[r][head_id, :]
            return np.random.choice(len(sample_prob), size=1, replace=True, p=sample_prob)[0]

        def head_sample(r, num_of_nodes=1):
            return np.random.choice(len(self.head_prob[r]), size=num_of_nodes, replace=True, p=self.head_prob[r])

        def neg_samples(pt):
            # From entity id to meta node id
            cor_type, r = pt['cor_type'], self.rel2rid.inv[pt['r']]
            h, t = self.mid2eid[r[0]].inv[pt['h']], self.mid2eid[r[1]].inv[pt['t']]
            threshold = self.kge_rels[r][h, t]
            # Results Init: Initialized as the original entities.
            sampled_mids = np.zeros(len(cor_type))

            # Prob Init: e.g. head_prob: num of tail entities
            head_prob, tail_prob = [np.ones(_) for _ in self.kge_rels[r].shape]

            # According to pairwise ranking objective, score of negative samples are lower than the score of positive sample
            # Head prob: the triples with same tail node whose scores are greater than threshold should not be sampled.
            head_prob[self.kge_rels[r].t()[t, :] > threshold] = 0
            tail_prob[self.kge_rels[r][h, :] > threshold] = 0

            #
            if head_prob.sum() > 0 and tail_prob.sum() > 0:
                # Sample meta nodes
                sampled_mids[cor_type == 'h'] = np.random.choice(
                    len(head_prob), size=sum(cor_type == 'h'), replace=True, p=head_prob / head_prob.sum())
                sampled_mids[cor_type == 't'] = np.random.choice(
                    len(tail_prob), size=sum(cor_type == 't'), replace=True, p=tail_prob / tail_prob.sum())
                # Convert to entity ids
                ret = [self.mid2eid[r[0]][sampled_mids[_]] if ct == 'h' else self.mid2eid[r[1]][sampled_mids[_]]
                       for _, ct in enumerate(cor_type)]
                return ret
            elif head_prob.sum() > 0:  # minimum score in tail entites, convert to head corrupt types
                pt['cor_type'] = ['h' for _ in len(cor_type)]
                return neg_samples(pt)
            elif tail_prob.sum() > 0:  # minimum score in head entites, convert to tail corrupt types
                pt['cor_type'] = ['h' for _ in len(cor_type)]
                return neg_samples(pt)
            else:  # minimum score both row and column wise, return original entities
                return [pt['h'] if ct == 'h' else pt['t'] for ct in cor_type]

        def get_pos_triples(r, h_nodes):
            return [{
                'h': self.mid2eid[r[0]][head_id],  # Head entity ID
                'r': self.rel2rid[r],  # Relation ID
                't': self.mid2eid[r[1]][tail_sample_by_head(head_id, r)],  # Tail entity ID
                'l': 1,
                'cor_type': np.random.choice(['h', 't'], self.neg_rate),
                'ind': f'-Pos-{r}-{head_id}',
            } for head_id in h_nodes]

        pos_triples = [get_pos_triples(r, head_sample(r, self.pos_ent_per_rel))
                       for r in self.ssl_rels]
        pos_triples = list(chain(*pos_triples))

        neg_triples = [{
            'h': corrupted_entity if pt['cor_type'][_] == 'h' else pt['h'],
            'r': pt['r'],
            't': corrupted_entity if pt['cor_type'][_] == 't' else pt['t'],
            'l': -1,
            'cor_type': pt['cor_type'][_],
            'ind': f'{_}Neg-{self.rel2rid.inv[pt["r"]]}-{pt["h"]}',
        } for pt in pos_triples  # Iter triples
            for _, corrupted_entity in enumerate(neg_samples(pt))  # Iter neg entities
        ]
        return pd.DataFrame.from_records(pos_triples + neg_triples)


def adj_batch_loader(shape, batch_size, is_symmetric, desc=''):
    'Load NxN combinations'

    if is_symmetric:  # Since the types of symmetric relations are the same, the order doesn't matters
        adj_combinations = list(combinations_with_replacement(list(range(shape[0])), r=2))
    else:  # Since the types of symmetric relations are the same, the order matters
        adj_combinations = list(product(list(range(shape[0])), list(range(shape[1]))))

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    batch_cnt = int(len(adj_combinations) / batch_size) + 1
    return batch(adj_combinations, batch_size)
