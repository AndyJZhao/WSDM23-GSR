from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class MGSL_PretrainedFeatConfig(ModelConfig):

    def __init__(self, args):
        super(MGSL_PretrainedFeatConfig, self).__init__('MGSL')
        # ! Model settings
        self.dataset = args.dataset

        # ! Pretrain settings
        self.feat_order = 2
        self.batch_size = 128  # number of samples per batch
        self.num_workers = 0
        self.enc_conf = 'GCN_2_48'
        self.enc_conf = 'MLP_2_48'
        self.decoder_layer = 2
        self.decoder_n_hidden = 48
        self.pretrain_weights = '0.6_0.4'
        self.pretrain_weights = '1.0_0'
        self.pretrain_weights = '0.2_0.8'
        self.pretrain_weights = '0_1'
        self.pretrain_weights = '0.5_0.5'

        self.p_epochs = 2000
        self.p_epochs = 1000
        self.momentum_factor = 0.999
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.nce_k = 4096
        self.nce_t = 0.07

        # ! Coarse filtering settings
        self.ew_mode = 'F0_EqEW'
        self.ew_mode = 'NPFreeze_EqEW'
        self.ew_mode = 'NPFreeze_SimEW'
        self.ew_mode = 'NPFine_SimEW'
        self.ew_mode = 'F0_SimEW'

        self.filter_mode = 'TRS_0.8'
        self.filter_mode = 'TopK_50'
        self.filter_mode = 'Mod_0_1.0'
        self.filter_mode = 'Mod_0_1.0'
        self.filter_mode = 'Mod_0_0'
        self.filter_mode = 'Mod_0.02_2.0'
        # ! Predictor Settings
        self.mlp_n_layer = 1  # Tuned Apr 16, 1 layer MLP is enough for most cases
        self.mlp_n_hidden = 32
        self.mlp_dropout = 0.5
        self.att_n_hidden = 128
        self.att_n_hidden = 32

        # ! GSL settings
        self.views = ['F', 'S']
        self.gf_mode = 'gf_att'
        self.gf_mode = 'gf_att'
        self.gf_mode = 'ef_mean'
        self.gf_mode = 'cf_att'
        self.gf_mode = 'ef_mlp'  # Bad Performance
        self.gf_mode = 'ef_att'
        self.gf_mode = 'ef_chn'
        self.gf_mode = 'cf_att'
        self.gf_mode = 'cf_chn'

        self.g_names = ['S', 'ori']
        self.g_names = ['F']
        self.g_names = ['S']
        self.g_names = ['S', 'F', 'ori']
        self.g_names = ['ori']
        self.g_names = ['S', 'F']
        self.num_head = 2

        # ! Normalization
        self.adj_norm = False  # graph virtual adjacency normalization
        self.adj_norm = True  # graph virtual adjacency normalization
        self.edge_weight_norm = None
        self.emb_norm = None

        # ! GCN settings
        self.lr = 0.01
        self.dropout = 0.5
        self.n_hidden = 64
        self.n_layer = 2
        self.weight_decay = 5e-4
        self.early_stop = 100
        self.epochs = 1000
        # ! Experiment settings
        self.train_percentage = 15
        self.update__model_conf_list()  # * Save the model config list keys
        self._file_conf_list += ['embedding_file', 'pretrain_model_ckpt']
        self.update_modified_conf(args.__dict__)

    @property
    def pretrain_prefix(self):
        return f"fo{self.feat_order}_lr{self.lr}_bsz{self.batch_size}_pi{self.p_epochs}_fo{self.feat_order}_enc{self.enc_conf}_dec-l{self.decoder_layer}_hidden{self.decoder_n_hidden}-prt_w-{self.pretrain_weights}_ncek{self.nce_k}"

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_PR-{self.pretrain_prefix}-_G-{'_'.join(self.g_names)}_FM-{self.filter_mode}_EW{self.ew_mode}_GM-{self.gf_mode}att_hidden{self.att_n_hidden}_L{self.n_layer}-Norm{int(self.adj_norm)}_ewNorm{self.edge_weight_norm}_GCN-do{self.dropout}_L{self.mlp_n_layer}MLP_do{self.mlp_dropout}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"

    @property
    def embedding_file(self):
        return f"{TEMP_PATH}{self.model}/prt_embs/{self.dataset}/fo_{self.feat_order}.pth"

    @property
    def pretrain_model_ckpt(self):
        return f"{TEMP_PATH}{self.model}/prt_model_ckpts/{self.dataset}/{self.pretrain_prefix}.ckpt"
