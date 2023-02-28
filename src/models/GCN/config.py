from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class GCNConfig(ModelConfig):

    def __init__(self, args):
        super(GCNConfig, self).__init__('GCN')
        # ! Model settings
        self.dataset = args.dataset
        self.lr = 0.01
        self.dropout = 0.5
        self.n_hidden = 64
        self.n_layer = 2
        self.n_hidden = 256
        self.weight_decay = 5e-4
        self.early_stop = -1
        self.early_stop = 100
        self.epochs = 500
        # ! Other settings
        self.update__model_conf_list()  # * Save the model config list keys
        self.update_modified_conf(args.__dict__)

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_layer{self.n_layer}_nhidden{self.n_hidden}_{self.model}_lr{self.lr}_dropout{self.dropout}_es{self.early_stop if self.early_stop > 0 else 0}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"
