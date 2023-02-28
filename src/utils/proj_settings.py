# ! Project Settings
RES_PATH = 'temp_results/'
SUM_PATH = 'results/'
LOG_PATH = 'log/'
TEMP_PATH = 'temp/'
DATA_PATH = 'data/'

EVAL_METRIC = 'test_acc'
P_EPOCHS_SAVE_LIST = [1, 2, 3, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300]

PARA_DICT = {
    'intra_weight': {
        'cora': 0.5,
        'citeseer': 0.75,
        'airport': 0.75,
        'blogcatalog': 0.75,
        'flickr': 0.25,
        'arxiv': 0
    },
    'fsim_weight': {
        'cora': 0.25,
        'citeseer': 0.75,
        'airport': 1.0,
        'blogcatalog': 0.1,
        'flickr': 1,
        'arxiv': 0.250
    },

    'fan_out': {
        'cora': '20_40',
        'citeseer': '1_2',
        'airport': '5_10',
        'blogcatalog': '15_30',
        'flickr': '15_30',
        'arxiv': '5_10',
    },

    'add_ratio': {
        'cora': 0.5,
        'citeseer': 0.5,
        'airport': 0.7,
        'blogcatalog': 0,
        'flickr': 0,
        'arxiv': 0.1,
    },

    'rm_ratio': {
        'cora': 0,
        'citeseer': 0,
        'airport': 0.45,
        'blogcatalog': 0.05,
        'flickr': 0.05,
        'arxiv': 0.05,
    },

    'p_epochs': {
        'cora': 50,
        'citeseer': 100,
        'airport': 5,
        'blogcatalog': 3,
        'flickr': 1,
        'arxiv': 2,
    },

    'p_batch_size': {
        'cora': 128,
        'citeseer': 128,
        'airport': 128,
        'blogcatalog': 128,
        'flickr': 128,
        'arxiv': 1024,
    },
    'prt_lr': {
        'cora': 0.001,
        'citeseer': 0.005,
        'airport': 0.001,
        'blogcatalog': 0.01,
        'flickr': 0.01,
        'arxiv': 0.001,
    },
    'activation': {
        'cora': 'Elu',
        'citeseer': 'Elu',
        'airport': 'Elu',
        'blogcatalog': 'Elu',
        'flickr': 'Elu',
        'arxiv': 'Elu',
    },
}
