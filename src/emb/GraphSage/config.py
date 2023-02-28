from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class GaphSageConfig(ModelConfig):

    def __init__(self, args):
        super(GaphSageConfig, self).__init__('GaphSage')
        # ! Model settings
        self.dataset = args.dataset
        self.lr = 0.001
        self.epochs = 20
        self.n_layer = 2
        self.n_hidden = 16
        self.dropout = 0.5
        self.num_negs = 1
        self.aggregator = 'gcn'
        self.batch_size = 10000
        self.early_stop = 100
        self.train_percentage = 10
        # ! Other settings
        self.seed = args.seed
        self.neg_share = False
        self.fan_out = '25,50'
        self.num_workers = 2
        self.weight_decay = 5e-4
        self.update__model_conf_list()  # * Save the model config list keys
        self.update_modified_conf(args.__dict__)

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_{self.model}_lr{self.lr}_dropout{self.dropout}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"
