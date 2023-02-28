from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class GraphSageConfig(ModelConfig):

    def __init__(self, args):
        super(GraphSageConfig, self).__init__('GraphSage')
        # ! Model settings
        self.dataset = args.dataset
        self.lr = 0.01
        self.epochs = 500
        self.n_layer = 1
        self.n_hidden = 256
        self.dropout = 0.5
        self.aggregator = 'gcn'
        self.early_stop = 100
        self.train_percentage = 10
        # ! Other settings
        self.weight_decay = 5e-4
        self.update__model_conf_list()  # * Save the model config list keys
        self.update_modified_conf(args.__dict__)

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_layer{self.n_layer}_nhidden{self.n_hidden}_{self.model}_lr{self.lr}_dropout{self.dropout}_agg<{self.aggregator}>_es{self.early_stop if self.early_stop > 0 else 0}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"
