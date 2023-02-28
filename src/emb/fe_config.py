from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH


class FEConfig(ModelConfig):

    def __init__(self, args):
        super(FEConfig, self).__init__('FeatureEmbedding')
        # ! Model settings

        FEConfig_dict = {
            'cora': {
                'lr': 0.001,
                'epochs': 100,
                'dropout': 0.8,
                'aggregator': 'gcn',
            },
            'citeseer': {
                'lr': 0.001,
                'epochs': 100,
                'dropout': 0.5,
                'aggregator': 'gcn',
            },
            'pubmed': {
                'lr': 0.005,
                'epochs': 3,
                'dropout': 0.5,
                'aggregator': 'gcn',
            },
        }

        self.dataset = args.dataset
        dataset_config = FEConfig_dict[self.dataset]
        self.fe_lr = dataset_config['lr']
        self.fe_epochs = dataset_config['epochs']
        self.fe_dropout = dataset_config['dropout']
        self.fe_aggregator = dataset_config['aggregator']
        self.fe_layer = 2
        self.fe_hidden = 64
        self.fe_num_negs = 1
        self.fe_batch_size = 10000
        self.fe_early_stop = 100
        self.fe_train_percentage = args.train_percentage
        # ! Other settings
        self.seed = args.seed
        self.fe_neg_share = False
        # self.fe_fan_out = '25,50'
        self.fe_fan_out = '25'
        self.num_workers = 2
        self.fe_weight_decay = 5e-4


    @property
    def f_prefix(self):
        return f"l{self.fe_train_percentage}_{self.model}_lr{self.fe_lr}_do{self.fe_dropout}_e{self.fe_epochs}_agg-{self.fe_aggregator}"

    @property
    def checkpoint_file(self):
        return f"src/emb/GraphSage/{self.dataset}_{self.f_prefix}.ckpt"
