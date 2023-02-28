import os
import sys

sys.path.append((os.path.abspath(os.path.dirname(__file__)).split('src')[0] + 'src'))

from utils.proj_settings import RES_PATH
from utils.tune_utils import summarize_by_folder, summarize_by_tune_df, gen_tune_df
from models.MGSL.tuneMGSL import TUNE_DICT
from models.MGSL.config import MGSLConfig

default_config_dict = {}


def summarize_results(datasets=['pubmed', 'cora', 'citeseer'],
                      exp_list=['RoughTune'],
                      train_percentage_list=[1, 5, 10],
                      models=[('MGSL', MGSLConfig)]):
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
                    tune_df = gen_tune_df(TUNE_DICT[exp_name])
                    summarize_by_tune_df(tune_df, {**default_config_dict, **trial_dict})


if __name__ == "__main__":
    summarize_results()
# python /home/zja/PyProject/MGSL/src/models/MGSL/sum_results.py
