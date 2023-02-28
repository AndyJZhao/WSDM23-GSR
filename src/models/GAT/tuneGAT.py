import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import utils.util_funcs as uf
from utils.tune_utils import *
from models.GCN.train import train_gcn
from models.GCN.config import GCNConfig
import argparse

model = 'GCN'
TUNE_DICT = {
    'RoughTune':
        {
            'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'lr': [0.005, 0.01, 0.05],
            'n_layer': 2,
        },
    'Test':
        {
            'dropout': [0],
            'lr': [0.05, 0.01],
            'n_layer': 2,
        }
}


@uf.time_logger
def tune_gcn():
    # * =============== Init Args =================
    exp_name = 'RoughTune'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_times', type=int, default=5)
    parser.add_argument('-s', '--start_ind', type=int, default=0)
    parser.add_argument('-v', '--reverse_iter', action='store_true', help='reverse iter or not')
    parser.add_argument('-b', '--block_log', action='store_false', help='block log or not')
    parser.add_argument('-d', '--dataset', type=str, default='cora')
    parser.add_argument('-e', '--exp_name', type=str, default=exp_name)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-t', '--train_percentage', type=int, default=3)
    args = parser.parse_args()
    args.__dict__.update({'model': model, 'config': GCNConfig})

    # * =============== Fine Tune (grid search) =================
    print(args.exp_name)

    tune_df = gen_tune_df(TUNE_DICT[args.exp_name])
    print(f'{tune_df}')
    print(f'{args}')
    exp_settings = ['run_times', 'start_ind', 'reverse_iter']
    ad = args.__dict__
    trial_settings = {k: ad[k] for k in ad if k not in exp_settings}
    grid_search(args, train_gcn, tune_df, trial_settings)
    # * =============== Result Summary  =================

    summarize_by_tune_df(tune_df, args.__dict__)


if __name__ == '__main__':
    tune_gcn()
