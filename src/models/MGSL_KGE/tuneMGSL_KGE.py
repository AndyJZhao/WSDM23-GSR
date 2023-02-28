import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import utils.util_funcs as uf
from utils.tune_utils import *
from models.MGSL_KGE.trainMGSL_KGE import train_MGSL_KGE
from models.MGSL_KGE.config import MGSL_KGEConfig
import argparse
import numpy as np

gsmp_large_list = [
    ['NFFN', 'NLLN', 'NSSN', 'NFLN', 'NSLN', 'NFSN'],
    ['NFLN', 'NSLN', 'NFSN'],
    ['NFFN', 'NLLN', 'NSSN'],
    ['NSSN', 'NSLN'],
    ['NSSN', 'NFLN'],
    ['NFFN', 'NFLN'],
    ['NSSN', 'NFFN'],
    ['NFFN'],
    ['NLLN'],
    ['NSSN'],
    ['NFLN'],
    ['NFSN'],
    ['NSLN'],
    ['NFFN', 'ori'],
    ['NLLN', 'ori'],
    ['NSSN', 'ori'],
    ['NFLN', 'ori'],
    ['NFSN', 'ori'],
    ['NSLN', 'ori'],
    ['ori']
]
gsmp_rough_list = [
    ['NFFN'],
    ['NLLN'],
    ['NSSN'],
    ['NFLN'],
    ['NFSN'],
    ['NSLN'],
    ['ori']
]

dropout_list = np.linspace(0, 1, 11).tolist()  # Start from zero will cost error
TRS_full_list = [f'TRS_{i:.3f}' for i in np.linspace(0.65, 0.975, 14).tolist()]
TRS_rough_list = [f'TRS_{i:.1f}' for i in np.linspace(0.7, 0.9, 3).tolist()]
K_full_list = [f'TopK_{i}' for i in [100, 50, 1, 3, 5, 7, 10, 20]]
K_rough_list = [f'TopK_{i}' for i in [1, 3, 5, 10]]
K_rough_list = [f'TopK_{i}' for i in [1, 3, 5, 10]]
filter_mode_list = K_full_list + TRS_full_list
filter_mode_rough_list = K_rough_list + TRS_rough_list
model = 'MGSL_KGE'
TUNE_DICT = {
    'GSMP': {
        'gsmp_list': gsmp_large_list,
        'filter_mode': filter_mode_list,
    },

    'RoughTune':
        {
            'gsmp_list': gsmp_large_list,
            'adj_norm': [True, False],
            'gf_mode': 'cf_att',
            'filter_mode': 'TopK_5',
            'p_iters': [100]
        },
    'CheckSingle':
        {
            'gsmp_list': gsmp_large_list,
            'adj_norm': True,
            'gf_mode': 'cf_att',
            'filter_mode': 'TopK_5',
            'p_iters': 100,
            'att_weight': None,
        },
}


@uf.time_logger
def tune_MGSL_KGE():
    # * =============== Init Args =================
    exp_name = 'RoughTune'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_times', type=int, default=3)
    parser.add_argument('-s', '--start_ind', type=int, default=0)
    parser.add_argument('-v', '--reverse_iter', action='store_true', help='reverse iter or not')
    parser.add_argument('-b', '--block_log', action='store_false', help='block log or not')
    parser.add_argument('-d', '--dataset', type=str, default='cora')
    parser.add_argument('-e', '--exp_name', type=str, default=exp_name)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-t', '--train_percentage', type=int, default=5)
    args = parser.parse_args()
    args.__dict__.update({'model': model, 'config': MGSL_KGEConfig})

    # * =============== Fine Tune (grid search) =================
    print(args.exp_name)

    tune_df = gen_tune_df(TUNE_DICT[args.exp_name])
    print(f'{tune_df}')
    print(f'{args}')
    exp_settings = ['run_times', 'start_ind', 'reverse_iter']
    ad = args.__dict__
    trial_settings = {k: ad[k] for k in ad if k not in exp_settings}
    grid_search(args, train_MGSL_KGE, tune_df, trial_settings)
    # * =============== Result Summary  =================

    summarize_by_tune_df(tune_df, args.__dict__)


if __name__ == '__main__':
    tune_MGSL_KGE()

# python /home/zja/PyProject/MGSL_KGE/src/models/MGSL_KGE/tuneMGSL_KGE.py -dcora -t-1 -eRoughTune
# python /home/zja/PyProject/MGSL_KGE/src/models/MGSL_KGE/tuneMGSL_KGE.py -dcora -t5 -eRoughTune
# python /home/zja/PyProject/MGSL_KGE/src/models/MGSL_KGE/tuneMGSL_KGE.py -dcora -t10 -eRoughTune
# python /home/zja/PyProject/MGSL_KGE/src/models/MGSL_KGE/tuneMGSL_KGE.py -dcora -t15 -eRoughTune
# python /home/zja/PyProject/MGSL_KGE/src/models/MGSL_KGE/tuneMGSL_KGE.py -dcora -t15 -eCheckSingle
# python /home/zja/PyProject/MGSL_KGE/src/models/MGSL_KGE/tuneMGSL_KGE.py -dciteseer -t5
# python /home/zja/PyProject/MGSL_KGE/src/models/MGSL_KGE/tuneMGSL_KGE.py -dciteseer -t10
# python /home/zja/PyProject/MGSL_KGE/src/models/MGSL_KGE/tuneMGSL_KGE.py -dciteseer -t15
