import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
import utils.util_funcs as uf
from utils.tune_utils import *
from models.MGSL_PretrainedFeatEmb.trainMGSL import train_mgsl_pretrained_feat
from models.MGSL_PretrainedFeatEmb.config import MGSLConfig
import argparse
import numpy as np

g_name_list = [
    ['S', 'F', 'ori'],
    ['F', 'S'],
    # ['ori'],
    # ['ori', 'S'],
    # ['ori', 'F'],
    # ['F'],
    # ['S'],
]
rm_cand_ratio = [0, 0.2, 0.4, 0.6]
rm_cand_ratio = [0, 0.05, 0.1, 0.2, 0.4, 0.6]
add_cand_ratio = [0, 1.0, 2.0]
add_cand_ratio = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
filter_mode_list = [f'Mod_{i}_{j}' for i in rm_cand_ratio for j in add_cand_ratio]
ew_mode_list = ['F0_SimEW', 'F0_EqEW', 'NPFreeze_EqEW', 'NPFreeze_SimEW', 'NPFine_SimEW']
# rm_cand_ratio = [0, 0.025, 0.05]
# add_cand_ratio = [0, 0.05, 0.1]
# filter_mode_list = [f'Mod_{i}_{j}' for i in rm_cand_ratio for j in add_cand_ratio]
model = 'MGSL_PretrainedFeatEmb'
TUNE_DICT = {
    'RoughTune':
        {
            'edge_weight_norm': None,
            'filter_mode': ['Mod_0_1.0'],
            'g_names': g_name_list,
            'gf_mode': 'cf_chn',  # Tuning
            'ew_mode': ['NPFreeze_EqEW','F0_EqEW'],
            'p_epochs': 2000,
            'batch_size': 256,
            'pretrain_weights': ['0_1', '1_0', '0.8_0.2', '0.2_0.8', '0.5_0.5'],
        },
    'PretrainTune':
        {
            'edge_weight_norm': None,
            'filter_mode': filter_mode_list,
            'att_n_hidden': 32,  # Tuning
            'nce_k': [4096, 8192, 16384],  # Tuning
            'g_names': [['F', 'S']],
            'gf_mode': 'cf_chn',  # Tuning
            'ew_mode': 'NPFreeze_EqEW',
            'p_epochs': [1000, 2000, 5000],
            'batch_size': 256,
            'pretrain_weights': ['0_1', '1_0', '0.8_0.2', '0.2_0.8', '0.5_0.5'],
        },
}


@uf.time_logger
def tune_mgsl_pretrained_feat():
    # * =============== Init Args =================
    exp_name = 'RoughTune'
    exp_name = 'PretrainTune'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_times', type=int, default=3)
    parser.add_argument('-s', '--start_ind', type=int, default=0)
    parser.add_argument('-v', '--reverse_iter', action='store_true', help='reverse iter or not')
    parser.add_argument('-b', '--block_log', action='store_false', help='block log or not')
    parser.add_argument('-d', '--dataset', type=str, default='cora')
    parser.add_argument('-e', '--exp_name', type=str, default=exp_name)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-t', '--train_percentage', type=int, default=15)
    args = parser.parse_args()
    args.__dict__.update({'model': model, 'config': MGSLConfig})
    if '192.168.0' in get_ip():
        args.gpu = -1
    # * =============== Fine Tune (grid search) =================
    print(args.exp_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    tune_df = gen_tune_df(TUNE_DICT[args.exp_name])
    print(f'{tune_df}')
    print(f'{args}')
    exp_settings = ['run_times', 'start_ind', 'reverse_iter']
    ad = args.__dict__
    trial_settings = {k: ad[k] for k in ad if k not in exp_settings}
    grid_search(args, train_mgsl, tune_df, trial_settings)
    # * =============== Result Summary  =================

    summarize_by_tune_df(tune_df, args.__dict__)


if __name__ == '__main__':
    tune_mgsl_pretrained_feat()

# tu -dcora -t1 -eRoughTune -g0
# tu -dcora -t5 -eRoughTune -g0
# tu -dcora -t10 -eRoughTune -g0

# tu -dcora -t1 -ePretrainTune -g1 -v
# tu -dcora -t5 -ePretrainTune -g1 -v
# tu -dcora -t10 -ePretrainTune -g1 -v

# tu -dpubmed -eRoughTune -g0 -t1
# tu -dpubmed -ePretrainTune -g0 -t5
# tu -dpubmed -eRoughTune -g0 -t10

# python /home/zja/PyProject/MGSL/src/models/MGSL/sum_results.py
# tu
# tu -dciteseer -t1 -eRoughTune -g0
# tu -dciteseer -t5 -eRoughTune -g0
# tu -dciteseer -t10 -eRoughTune -g1
# tu -dciteseer -t15 -eRoughTune
# tu
# tu -dpubmed -t5 -eRoughTune
# tu -dpubmed -t10 -eRoughTune
# tu -dpubmed -t15 -eRoughTune -v
# tu -dciteseer -t15 -eCheckEdgeWeightNorm
