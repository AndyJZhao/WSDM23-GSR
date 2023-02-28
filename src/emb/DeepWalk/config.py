from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class DeepWalkConfig(ModelConfig):

    def __init__(self, args):
        super(DeepWalkConfig, self).__init__('DeepWalk')
        # ! Model settings
        self.dataset = args.dataset
        self.num_walks = 10
        self.walk_length = 100
        self.window_size = 11
        self.train_rate = 10
        self.n_hidden = 64
        self.num_workers = 32
        self.train_percentage = 10
        # ! Other settings
        self.seed = args.seed
        self.update_model_conf_list()  # * Save the model config list keys
        self.update_modified_conf(args.__dict__)

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_{self.model}_nw{self.num_walks}_wl{self.walk_length}_ws{self.window_size}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"
