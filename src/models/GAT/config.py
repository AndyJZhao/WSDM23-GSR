from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class GATConfig(ModelConfig):

    def __init__(self, args):
        super(GATConfig, self).__init__('GAT')
        # ! Model settings
        self.dataset = args.dataset
        self.lr = 0.005
        self.n_head = 8
        self.n_out_head = 1
        self.n_layer = 1
        self.n_hidden = 8  # dgl default =8
        self.feat_drop = 0.5
        self.attn_drop = 0.5
        self.early_stop = 100
        self.epochs = 500
        self.train_percentage = 10
        # ! Other settings
        self.negative_slope = 0.2
        self.weight_decay = 5e-4
        self.update__model_conf_list()  # * Save the model config list keys
        self.update_modified_conf(args.__dict__)

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_layer{self.n_layer}_nhidden{self.n_hidden}_nhead{self.n_head}_{self.model}_lr{self.lr}_featdrop{self.feat_drop}_attndrop{self.attn_drop}_es{self.early_stop if self.early_stop > 0 else 0}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"
