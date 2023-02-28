import os
import sys

sys.path.append((os.path.abspath(os.path.dirname(__file__)).split('src')[0] + 'src'))

from utils.proj_settings import RES_PATH
from utils.tune_utils import summarize_by_folder, summarize_by_tune_df, gen_tune_df
from models.GSR.tuneGSR import EXP_DICT
from models.GSR.config import GSRConfig

default_config_dict = {}


def summarize_results(datasets=['cora', 'citeseer', 'arxiv', 'blogcatalog', 'flickr','arxiv'],
                      exp_list=['RoughTune'],
                      train_percentage_list=[0, 1, 3, 5, 10],
                      models=[('GSR', GSRConfig)]):
    for dataset in datasets:
        for model, _ in models:
            try:
                summarize_by_folder(dataset, model)
            except:
                pass

    for dataset in datasets:
        for model, config in models:
            for exp_name in exp_list:
                for train_percentage in train_percentage_list:
                    trial_dict = {'dataset': dataset, 'exp_name': exp_name,
                                  'model': model, 'config': config, 'train_percentage': train_percentage}
                    tune_df = gen_tune_df(EXP_DICT[exp_name])
                    summarize_by_tune_df(tune_df, {**default_config_dict, **trial_dict})


if __name__ == "__main__":
    summarize_results()
# python /home/zja/PyProject/MGSL/src/models/GSR/sum_results.py

import os
import sys

sys.path.append((os.path.abspath(os.path.dirname(__file__)).split('src')[0] + 'src'))

from utils.proj_settings import RES_PATH
from utils.tune_utils import summarize_by_folder, summarize_by_tune_df, gen_tune_df
from utils.proj_settings import TUNE_DICT
from models.MVSE.config import MVSEConfig
from models.MVSE_freeze.config import MVSE_FREEZEConfig

default_config_dict = {'batch_size': 128, 'intra_weight': '0_0_1'}


def summarize_results(datasets=['acm', 'dblp', 'imdb'],
                      exp_list=['WL', 'WH', 'IntraVSCombined', 'woCombined'],
                      train_percentage_list=[1, 3, 5],
                      models=[('MVSE', MVSEConfig), ('MVSE_FREEZE', MVSE_FREEZEConfig)]):
    for dataset in datasets:
        for model, _ in models:
            summarize_by_folder(dataset, model)

    for dataset in datasets:
        for model, config in models:
            for exp_name in exp_list:
                for train_percentage in train_percentage_list:
                    trial_dict = {'dataset': dataset, 'exp_name': exp_name,
                                  'model': model, 'config': config, 'train_percentage': train_percentage}
                    tune_df = gen_tune_df(TUNE_DICT[exp_name])
                    summarize_by_tune_df(tune_df, {**default_config_dict, **trial_dict})


if __name__ == "__main__":
    summarize_results()
# pyt src/utils/sum_results.py
