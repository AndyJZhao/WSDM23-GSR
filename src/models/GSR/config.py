from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class GSRConfig(ModelConfig):

    def __init__(self, args):
        super(GSRConfig, self).__init__('GSR')
        # ! Model settings
        self.dataset = args.dataset
        self.gnn_model = 'SGC'
        self.gnn_model = 'GCN'
        # self.activation = 'Elu'
        self.activation = 'Relu'

        # ! Pretrain settings
        # self.prt_lr = 0.01
        # self.se_seed = 0
        self.semb = 'dw'
        self.prt_lr = 0.001 if self.dataset != 'arxiv' else 0.001

        self.p_epochs = 50
        self.p_batch_size = 512  # number of samples per batch
        self.p_schedule_step = 500
        self.poly_decay_lr = True
        self.fan_out = '20_40'
        self.num_workers = 0
        self.decoder_layer = 2
        self.decoder_n_hidden = 48
        self.intra_weight = 0.75
        # self.intra_weight = 0.25

        # Moco settings
        self.momentum_factor = 0.999
        self.nce_k = 16382
        self.nce_t = 0.07
        self.pre_dropout = 0

        # ! Graph Refinement Settings
        self.cos_batch_size = 2000 if self.dataset == 'arxiv' else 10000

        # self.fsim_norm = False  #
        self.fsim_norm = True
        self.fsim_weight = 0.0
        self.rm_ratio = 0.0
        self.add_ratio = 0.25
        # ! GCN settings
        self.cla_batch_size = 5000
        self.lr = 0.01
        self.dropout = 0.5
        self.n_hidden = 64 if self.dataset != 'arxiv' else 256

        self.weight_decay = 5e-4
        self.early_stop = 100
        self.epochs = 1000

        # ! Experiment settings
        self.train_percentage = 0
        self.update__model_conf_list()  # * Save the model config list keys
        self._file_conf_list += ['structural_em_file', 'pretrain_model_ckpt', 'refined_graph_file']
        self.update_modified_conf(args.__dict__)

    @property
    def pretrain_conf(self):

        if self.gnn_model == 'GAT':
            self.in_head = 8
            self.prt_out_head = 8
            self.gat_hidden = 8

        if self.gnn_model == 'GraphSage':
            self.aggregator = 'gcn'

        if self.gnn_model == 'SGC':
            self.k = 2

        if self.gnn_model == 'GCNII':
            self.alpha = 0.2
            self.lda = 1.0

        if self.semb == 'dw':
            return f"_lr{self.prt_lr}_bsz{self.p_batch_size}_pi{self.p_epochs}_enc{self.gnn_model}_dec-l{self.decoder_layer}_hidden{self.decoder_n_hidden}-prt_intra_w-{self.intra_weight}_ncek{self.nce_k}_fanout{self.fan_out}_prdo{self.pre_dropout}_act_{self.activation}_d{self.n_hidden}_pss{self.p_schedule_step}"
        elif self.semb == 'de':
            return f"_lr{self.prt_lr}_bsz{self.p_batch_size}_pi{self.p_epochs}_enc{self.gnn_model}_dec-l{self.decoder_layer}_hidden{self.decoder_n_hidden}-prt_intra_w-{self.intra_weight}_ncek{self.nce_k}_fanout{self.fan_out}_prdo{self.pre_dropout}_act_{self.activation}_d{self.n_hidden}_pdl{self.poly_decay_lr}_sek{self.se_k}"
        else:
            raise ValueError
        # return f"_se_seed{self.se_seed}_lr{self.prt_lr}_bsz{self.p_batch_size}_pi{self.p_epochs}_enc{self.gnn_model}_dec-l{self.decoder_layer}_hidden{self.decoder_n_hidden}-prt_intra_w-{self.intra_weight}_ncek{self.nce_k}_fanout{self.fan_out}_prdo{self.pre_dropout}_act_{self.activation}_d{self.n_hidden}"

    @property
    def graph_refine_conf(self):
        return f'fsim_norm{int(self.fsim_norm)}_fsim_weight{self.fsim_weight}_add{self.add_ratio}_rm{self.rm_ratio}'

    @property
    def finetune_conf(self):
        return f"lr{self.lr}_GCN-do{self.dropout}"

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_PR-{self.pretrain_conf}-_GR-{self.graph_refine_conf}_{self.finetune_conf}"

    @property
    def pretrain_model_ckpt(self):
        return f"{TEMP_PATH}{self.model}/p_model_ckpts/{self.dataset}/{self.pretrain_conf}.ckpt"

    @property
    def refined_graph_file(self):
        return f"{TEMP_PATH}{self.model}/refined_graphs/{self.dataset}/PR-{self.pretrain_conf}_GR-{self.graph_refine_conf}.bin"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/f_model_ckpts/{self.dataset}/{self.f_prefix}.ckpt"

    @property
    def structural_em_file(self):
        DWConfig_dict = {
            'cora': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'citeseer': {
                'num_walks': 10,
                'window_size': 13,
                'walk_length': 100,
            },
            'blogcatalog': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'flickr': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'arxiv': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'airport': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
        }

        DEConfig_dict = {
            'cora': {
                'k': 'full'
            },
            'citeseer': {
                'k': 6
            },
        }

        if self.semb == 'dw':
            self.se_num_walks = DWConfig_dict[self.dataset]['num_walks']
            self.se_window_size = DWConfig_dict[self.dataset]['window_size']
            self.se_walk_length = DWConfig_dict[self.dataset]['walk_length']
        if self.semb == 'de':
            self.se_k = DEConfig_dict[self.dataset]['k']

        # return f"{TEMP_PATH}{self.model}/structural_embs/{self.dataset}/{self.dataset}_seed{self.se_seed}_nw{self.se_num_walks}_ws{self.se_window_size}_wl{self.se_walk_length}.dw_emb"
        if self.semb == 'dw':
            return f"{TEMP_PATH}{self.model}/structural_embs/{self.dataset}/{self.dataset}_nw{self.se_num_walks}_ws{self.se_window_size}_wl{self.se_walk_length}.{self.semb}_emb"
        elif self.semb == 'de':
            return f"{TEMP_PATH}{self.model}/structural_embs/{self.dataset}/{self.dataset}_k{self.se_k}.{self.semb}_emb"
        else:
            raise ValueError