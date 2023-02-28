from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class MGSL_KGEConfig(ModelConfig):

    def __init__(self, args):
        super(MGSL_KGEConfig, self).__init__('MGSL_KGE')
        # ! Model settings
        self.dataset = args.dataset
        # ! Pretrain settings
        self.meta_types = 'FSL'
        self.feat_order = 2
        self.kge_model = 'TransR'
        self.kge_model = 'TransE'
        self.p_iters = 100  # Iters per relation
        self.batch_size = 16  # number of samples per batch
        self.pos_ent_per_rel = 8  # number of pos entities sampled per batch per relation
        self.neg_rate = 3
        self.meta_emb_dim = 64
        self.num_workers = 0
        # ! Coarse filtering settings
        self.filter_mode = 'TRS_0.8'
        self.filter_mode = 'TopK_5'
        # ! Predictor Settings
        self.mlp_n_layer = 1  # Tuned Apr 16, 1 layer MLP is enough for most cases
        self.mlp_n_hidden = 32
        self.mlp_dropout = 0.5

        # ! GSL settings
        self.gf_mode = 'gf_att'
        self.gf_mode = 'gf_att'
        self.gf_mode = 'ef_mean'
        self.gf_mode = 'ef_mlp'
        self.gf_mode = 'ef_att'
        self.gf_mode = 'cf_att'

        self.gsmp_list = ['NLLN']
        self.gsmp_list = ['NFFN']
        self.gsmp_list = ['NLLN', 'NFFN']
        self.gsmp_list = ['NFFN', 'NLLN', 'NSSN', 'NFLN', 'NSLN', 'NFSN']
        self.gsmp_list = []
        self.gsmp_list = ['NFSN', 'NLLN']
        self.gsmp_list = ['ori']
        self.gsmp_list = ['NLLN']
        self.gsmp_list = ['ori', 'NSSN']
        self.gsmp_list = ['ori']
        self.gsmp_list = ['NFSN']
        self.gsmp_list = ['ori', 'NSSN', 'NFSN', 'NFFN', 'NLLN']
        self.gsmp_list = ['NSSN']

        # ! Normalization
        self.adj_norm = False  # graph virtual adjacency normalization
        self.adj_norm = True  # graph virtual adjacency normalization
        self.edge_weight_norm = True
        self.edge_weight_norm = False

        self.emb_norm = None
        self.emb_norm = 'row1'
        # ! GCN settings
        self.lr = 0.01
        self.dropout = 0.5
        self.n_hidden = 64
        self.n_layer = 2
        self.weight_decay = 5e-4
        self.early_stop = 100
        self.epochs = 1000

        self.update__model_conf_list()  # * Save the model config list keys
        self.update_modified_conf(args.__dict__)

    @property
    def pretrain_prefix(self):
        return f"l{self.train_percentage}_lr{self.lr}_{self.kge_model}_{self.meta_types}_bsz{self.batch_size}_pi{self.p_iters}_neg{self.neg_rate}_fo{self.feat_order}_dm{self.meta_emb_dim}_norm{self.adj_norm}"

    @property
    def f_prefix(self):
        return f"PR-{self.pretrain_prefix}_GSMPs-{'_'.join(self.gsmp_list)}_FM-{self.filter_mode}_GM-{self.gf_mode}_L{self.n_layer}-Norm{int(self.adj_norm)}_GCN-do{self.dropout}_L{self.mlp_n_layer}MLP-do{self.mlp_dropout}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"

    @property
    def embedding_file(self):
        return f"{TEMP_PATH}{self.model}/prt_embs/{self.dataset}/fo_{self.feat_order}.pth"

    @property
    def pretrain_model_ckpt(self):
        return f"{TEMP_PATH}{self.model}/prt_model_ckpts/{self.dataset}/{self.pretrain_prefix}.ckpt"
