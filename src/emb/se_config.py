from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class SEConfig(ModelConfig):

    def __init__(self, args):
        super(SEConfig, self).__init__('StructureEmbedding')

        # # ! Model setting
        SEConfig_dict = {
            'cora': {
                'num_walks': 100,
                'window_size': 11,
                'walk_length': 100,
            },
            'citeseer': {
                'num_walks': 10,
                'window_size': 13,
                'walk_length': 100,
            },
            'pubmed': {
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

        self.dataset = args.dataset
        # dataset_config = SEConfig_dict[self.dataset]
        # self.se_num_walks = dataset_config['num_walks']
        # self.se_walk_length = dataset_config['walk_length']
        # self.se_window_size = dataset_config['window_size']
        self.se_num_walks = args.se_num_walks
        self.se_walk_length = args.se_walk_length
        self.se_window_size = args.se_window_size
        self.se_n_hidden = 64
        self.se_num_workers = 32
        self.train_percentage = args.train_percentage

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_{self.model}_nw{self.se_num_walks}_wl{self.se_walk_length}_ws{self.se_window_size}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"
