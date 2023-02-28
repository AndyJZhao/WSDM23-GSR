import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from utils import *

import argparse
import numpy as np

rm_ratio_list = [0, 0.05, 0.1, 0.2, 0.4, 0.6]
rm_ratio_list = [0, 0.2, 0.4, 0.6]
add_ratio_list = [0, 1.0, 2.0]
add_ratio_list = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
zero_to_one_rough_list = [0, 0.25, 0.5, 0.75, 1.0]
zero_to_half_rough_list = [0, 0.25, 0.5]
small_list = [0, 0.25, 0.5]
fsim_weight_list = [0, 0.25, 0.5, 0.75, 1.0]
p_epoch_list = [100, 0, 5, 10, 20, 50, 100]
zero_to_one_fine_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# rm_cand_ratio = [0, 0.025, 0.05]
# add_cand_ratio = [0, 0.05, 0.1]
# filter_mode_list = [f'Mod_{i}_{j}' for i in rm_cand_ratio for j in add_cand_ratio]
c_conf_dict = {  # 25 x Trials
    'semb': 'dw',
    'fsim_weight': zero_to_one_rough_list,  # 5
    'intra_weight': zero_to_one_rough_list,  # 5
    'fsim_norm': True,
    'stochastic_trainer': False,
    'activation': ['Elu'],
    'p_batch_size': [256],
    'p_schedule_step': [500]
}
fan_out_list = ['1_2', '3_5', '5_10', '10_20', '15_30', '20_40', '30_50']
EXP_DICT = {
    'RoughTune': {
        **c_conf_dict
        ,
        'data_spec_configs': {
            'fan_out': {
                'cora': '20_40',  # 5
                'citeseer': '10_20',  # 5
                'airport': '5_10',  # 2
                'blogcatalog': '15_30',
                'flickr': '15_30',
                'arxiv': '5_10',
            },
            'add_ratio': {
                'cora': zero_to_one_rough_list,  # 5
                'citeseer': zero_to_one_rough_list,  # 5
                'airport': zero_to_one_rough_list,  # 4
                'blogcatalog': [0, 0.5],
                'flickr': [0, 0.5],
                'arxiv': [0.0, 0.1, 0.2],

            },
            'rm_ratio': {
                'cora': [0.0],  # 2
                'citeseer': [0.0],
                'airport': [0.0, 0.25],
                'blogcatalog': [0.0, 0.1, 0.2],
                'flickr': [0.0, 0.1, 0.2],
                'arxiv': [0.0, 0.1],
            },
            'p_epochs': {
                'cora': [100, 10, 50],
                'citeseer': [200, 50, 100],
                'airport': [100, 10, 50],
                'blogcatalog': [3, 1],
                'flickr': [3, 1],
                'arxiv': [2, 1]
            },
        },
    },
}


@time_logger
def tune_mgsl():
    # * =============== Init Args =================
    exp_name = 'RoughTune'
    dataset = 'cora'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_times', type=int, default=3)
    parser.add_argument('-s', '--start_ind', type=int, default=0)
    parser.add_argument('-v', '--reverse_iter', action='store_true', help='reverse iter or not')
    parser.add_argument('-b', '--log_on', action='store_true', help='show log or not')
    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('-e', '--exp_name', type=str, default=exp_name)
    parser.add_argument('-g', '--gpu', type=int, default=1)
    parser.add_argument('-t', '--train_percentage', type=int, default=0)
    parser.add_argument('-o', '--gnn_type', type=str, default='GraphSage')
    args = parser.parse_args()
    if is_runing_on_local():
        args.gpu = -1

    # * =============== Model Specific Settings =================
    exp_init(seed=0, gpu_id=args.gpu)
    from models.GSR import train_GSR, GSRConfig
    model_settings = {'model': 'GSR', 'model_config': GSRConfig, 'train_func': train_GSR}
    args.__dict__.update(model_settings)

    # * =============== Fine Tune (grid search) =================
    tuner = Tuner(args, search_dict=EXP_DICT[args.exp_name])
    tuner.grid_search()
    tuner.summarize()


if __name__ == '__main__':
    tune_mgsl()

# tu -dcora -t1 -eFineTune1 -g0; tu -dcora -t3 -eFineTune1 -g0; tu -dcora -t5 -eFineTune1 -g0; tu -dcora -t10 -eFineTune1 -g0;

# python /home/zja/PyProject/MGSL/src/models/MGSL/sum_results.py


