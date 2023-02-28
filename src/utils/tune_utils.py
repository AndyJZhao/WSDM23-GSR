import multiprocessing
from utils.util_funcs import *
from utils.proj_settings import *
import pandas as pd
import time
from copy import deepcopy
import os
import ast
from itertools import product
from pprint import pformat
import traceback


class Tuner():
    # Major functions
    # ✅ Maintains dataset specific tune dict
    # ✅ Tune dict to tune dataframe (para combinations)
    # ✅ Beautiful printer
    # ✅ Build-in grid search function
    # ✅ Result summarization
    # ✅ Try-catch function to deal with bugs
    # ✅ Tune report to txt.
    def __init__(self, exp_args, search_dict, default_dict=None):
        self.birth_time = get_cur_time(t_format='%m_%d-%H_%M_%S')
        self.__dict__.update(exp_args.__dict__)
        self._d = deepcopy(default_dict) if default_dict is not None else {}
        if 'data_spec_configs' in search_dict:
            self.update_data_specific_cand_dict(search_dict['data_spec_configs'])
        search_dict.pop('data_spec_configs', None)

        self._d.update(search_dict)

    def update_data_specific_cand_dict(self, cand_dict):
        for k, v in cand_dict.items():
            self._d.update({k: v[self.dataset]})

    # * ============================= Properties =============================

    def __str__(self):
        return f'\nExperimental config: {pformat(self.cf)}\n' \
               f'\nGrid searched parameters:{pformat(self._d)}\n' \
               f'\nTune_df:{self.tune_df}\n'

    @property
    def cf(self):
        # All configs = tune specific configs + trial configs
        return {k: v for k, v in self.__dict__.items() if k[0] != '_'}

    @property
    def trial_cf(self):
        # Trial configs: configs for each trial.
        tune_global_cf = ['run_times', 'start_ind', 'reverse_iter',
                          'model', 'model_config', 'train_func', 'log_on', 'birth_time']
        return {k: self.cf[k] for k in self.cf if k not in tune_global_cf}

    @property
    def tune_df(self):
        # Tune dataframe: each row stands for a trial (hyper-parameter combination).
        # convert the values of parameters to list
        for para in self._d:
            if not isinstance(self._d[para], list):
                self._d[para] = [self._d[para]]
        return pd.DataFrame.from_records(dict_product(self._d))

    # * ============================= Tuning =============================

    @time_logger
    def grid_search(self):
        print(self)
        failed_trials, skipped_trials = 0, 0

        total_trials = len(self.tune_df) - self.start_ind
        finished_trials = 0
        outer_start_time = time.time()
        tune_dict = self.tune_df.to_dict('records')
        for i in range(self.start_ind, len(self.tune_df)):
            ind = len(self.tune_df) - i - 1 if self.reverse_iter else i
            para_dict = deepcopy(self.trial_cf)
            para_dict.update(tune_dict[ind])
            inner_start_time = time.time()
            print(f'\n{i}/{len(self.tune_df)} <{self.exp_name}> Start tuning: {para_dict}, {get_cur_time()}')
            res_file = self.model_config(Dict2Config(para_dict)).res_file
            if skip_results(res_file):
                print(f'Found previous results, skipped running current trial.')
                total_trials -= 1
                skipped_trials += 1
            else:
                try:
                    for seed in range(self.run_times):
                        para_dict['seed'] = seed
                        if not self.log_on: block_log()
                        cf = self.train_func(Dict2Config(para_dict))
                        if not self.log_on: enable_logs()
                        iter_time_estimate(f'\tSeed {seed}/{self.run_times}', '',
                                           inner_start_time, seed + 1, self.run_times)
                    finished_trials += 1
                    iter_time_estimate(f'Trial finished, ', '',
                                       outer_start_time, finished_trials, total_trials)
                except Exception as e:
                    log_file = f'log/{self.model}-{self.dataset}-{self.exp_name}-{self.birth_time}.log'
                    mkdir_list([log_file])
                    error_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                    with open(log_file, 'a+') as f:
                        f.write(
                            f'\nTrain failed at {get_cur_time()} while running {para_dict} at seed {seed}, error message:{error_msg}\n')
                        f.write(f'{"-" * 100}')
                    if not self.log_on: enable_logs()
                    print(f'Train failed, error message: {error_msg}')
                    failed_trials += 1
                    continue
                calc_mean_std(cf.res_file)
        print(f'\n\n{"="*24+" Grid Search Finished "+"="*24}\n'
              f'Successfully run {finished_trials} trials, skipped {skipped_trials} previous trials,'
              f'failed {failed_trials} trials.')
        if failed_trials>0: print(f'Check {log_file} for bug reports.\n{"="*70}\n')

    # * ============================= Results Processing =============================

    def summarize(self, metric=EVAL_METRIC):
        exp_name, model, dataset, train_percentage = self.exp_name, self.model, self.dataset, self.train_percentage
        res_f_list = self.tune_df_to_flist()
        print(f'\n\nSummarizing expriment {self.exp_name}...')
        out_prefix = f'{SUM_PATH}{model}/{dataset}/{model}_{dataset}<l{train_percentage:02d}><{exp_name}>'

        try:
            res_file = res_to_excel(res_f_list, out_prefix, f'avg_{metric}')
            print(f'Summary of {self.exp_name} finished. Results saved to {res_file}')
        except:
            print(f'!!!!!!Cannot summarize {self.exp_name} \tres_f_list:{res_f_list}\n '
                  f'was not summarized and skipped!!!!')

    def tune_df_to_flist(self):
        res_f_list = []
        tune_df = self.tune_df
        tune_dict = tune_df.to_dict('records')

        for i in range(len(tune_df)):
            para_dict = deepcopy(self.trial_cf)
            para_dict.update(tune_dict[i])
            res_file = self.model_config(Dict2Config(para_dict)).res_file
            if os.path.exists(res_file):
                res_f_list.append(res_file)
        return res_f_list


def iter_time_estimate(prefix, postfix, start_time, iters_finished, total_iters):
    """
    Generates progress bar AFTER the ith epoch.
    Args:
        prefix: the prefix of printed string
        postfix: the postfix of printed string
        start_time: start time of the iteration
        iters_finished: finished iterations
        max_i: max iteration index
        total_iters: total iteration to run, not necessarily
            equals to max_i since some trials are skiped.

    Returns: prints the generated progress bar
    """
    cur_run_time = time.time() - start_time
    total_estimated_time = cur_run_time * total_iters / iters_finished
    print(
        f'{prefix} [{time2str(cur_run_time)}/{time2str(total_estimated_time)}, {time2str(total_estimated_time - cur_run_time)} left] {postfix} [{get_cur_time()}]')


def dict_product(d):
    keys = d.keys()
    return [dict(zip(keys, element)) for element in product(*d.values())]


def add_tune_df_common_paras(tune_df, para_dict):
    for para in para_dict:
        tune_df[para] = [para_dict[para] for _ in range(len(tune_df))]
    return tune_df


@time_logger
def run_multiple_process(func, func_arg_list):
    '''
    Args:
        func: Function to run
        func_arg_list: An iterable object that contains several dict. Each dict has the input (**kwargs) of the tune_func

    Returns:

    '''
    process_list = []
    for func_arg in func_arg_list:
        _ = multiprocessing.Process(target=func, kwargs=func_arg)
        process_list.append(_)
        _.start()
    for _ in process_list:
        _.join()
    return


def summarize_by_folder(dataset, model, metric=EVAL_METRIC):
    '''
    Summarize model results
    '''
    model_res_path = f'{RES_PATH}{model}/{dataset}/'
    print(f'Summarizing--------{model_res_path}')
    f_list = os.listdir(model_res_path)
    print(f_list)
    for train_percentage in f_list:
        if ('.' not in train_percentage) and len(train_percentage) > 0 and train_percentage != 'debug':
            print(f'Summarizing expriment{train_percentage}')
            res_path = f'{model_res_path}{train_percentage}/'
            out_prefix = f'{model_res_path.replace(RES_PATH, SUM_PATH)}{model}_{dataset}<{train_percentage}>AllRes_'
            mkdir_list([out_prefix])
            res_f_list = [f'{res_path}{f}' for f in os.listdir(res_path)]
            try:
                res_to_excel(res_f_list, out_prefix, f'avg_{metric}')
                print(f'Summary of {res_path} finished.')
            except:
                print(f'!!!!!!Cannot summarize {res_path}\tf_list:{os.listdir(res_path)}\n{res_path}'
                      f'was not summarized and skipped!!!!')


def res_to_excel(res_f_list, out_prefix, metric):
    """
    res_path: folder name of the result files.
    """

    sum_res_list = []
    for res_file in res_f_list:
        # print(f'ResFile:{res_file}')
        if os.path.isfile(res_file) and res_file[-3:] == 'txt':
            # Load records
            sum_dict, conf_dict = {}, {}
            with open(res_file, 'r') as f:
                res_lines = f.readlines()
                # print(f'Summarizing {f}, Lines:{res_lines}')
                for line in res_lines:
                    if line[0] == '{':
                        d = ast.literal_eval(line.strip('\n'))
                        if 'dataset' in d.keys():  # parameters
                            conf_dict = d.copy()
                        elif 'avg_' in list(d.keys())[0]:  # mean results
                            avg_res_dict = dict(zip(d.keys(), [float(v) for v in d.values()]))
                        elif 'std_' in list(d.keys())[0]:
                            std_res_dict = dict(zip(d.keys(), [float(v) for v in d.values()]))
                try:
                    sum_dict.update(subset_dict(conf_dict, conf_dict['_interested_conf_list']))
                    sum_dict.update(avg_res_dict)
                    sum_dict.update(std_res_dict)
                except:
                    print(f'!!!!File {f.name} is not summarized, skipped!!!!')
                    continue
                sum_dict.pop('_interested_conf_list', None)
                conf_dict.pop('_interested_conf_list', None)

                sum_dict['config2str'] = conf_dict
                sum_res_list.append(sum_dict)
    sum_df = pd.DataFrame.from_dict(sum_res_list).sort_values(metric, ascending=False)
    max_res = sum_df.max()[metric]
    # ! Format mean and std
    metric_names = [cname[4:] for cname in sum_df.columns if 'avg' in cname]
    for m in metric_names:
        sum_df['avg_' + m] = sum_df['avg_' + m].apply(lambda x: f'{x:.2f}')
        sum_df['std_' + m] = sum_df['std_' + m].apply(lambda x: f'{x:.2f}')
        sum_df[m] = sum_df['avg_' + m] + '±' + sum_df['std_' + m]
        sum_df = sum_df.drop(columns=['avg_' + m, 'std_' + m])
    # ! Deal with NA columns
    for col in sum_df.columns[sum_df.isnull().any()]:
        for index, row in sum_df.iterrows():
            sum_df.loc[index, col] = row.config2str[col]
    # Reorder column order list : move config2str to the end
    col_names = list(sum_df.columns) + ['config2str']
    col_names.remove('config2str')
    sum_df = sum_df[col_names]
    # Save to excel
    mkdir_list([out_prefix])
    res_file = f'{out_prefix}{max_res:.2f}.xlsx'
    sum_df.to_excel(res_file)
    return res_file


def calc_mean_std(f_name):
    """
    Load results from f_name and calculate mean and std value
    """
    if os.path.exists(f_name):
        out_df, metric_set = load_dict_results(f_name)
    else:
        'Result file missing, skipped!!'
        return
    mean_res = out_df[metric_set].mean()
    std_res = out_df[metric_set].std()
    for k in metric_set:
        for m in ['acc', 'Acc', 'AUC', 'ROC', 'f1', 'F1']:
            if m in k:  # percentage of metric value
                mean_res[k] = mean_res[k] * 100
                std_res[k] = std_res[k] * 100
    mean_dict = dict(zip([f'avg_{m}' for m in metric_set], [f'{mean_res[m]:.2f}' for m in metric_set]))
    std_dict = dict(zip([f'std_{m}' for m in metric_set], [f'{std_res[m]:.2f}' for m in metric_set]))
    with open(f_name, 'a+') as f:
        f.write('\n\n' + '#' * 10 + 'AVG RESULTS' + '#' * 10 + '\n')
        for m in metric_set:
            f.write(f'{m}: {mean_res[m]:.4f} ({std_res[m]:.4f})\n')
        f.write('#' * 10 + '###########' + '#' * 10)
    write_nested_dict({'avg': mean_dict, 'std': std_dict}, f_name)


def load_dict_results(f_name):
    # Init records
    parameters = {}
    metric_set = None
    eid = 0
    with open(f_name, 'r') as f:
        res_lines = f.readlines()
        for line in res_lines:
            if line[0] == '{':
                d = ast.literal_eval(line.strip('\n'))
                if 'model' in d.keys():  # parameters
                    eid += 1
                    parameters[eid] = line.strip('\n')
                elif 'avg_' in list(d.keys())[0] or 'std_' in list(d.keys())[0]:
                    pass
                else:  # results
                    if metric_set == None:
                        metric_set = list(d.keys())
                        for m in metric_set:  # init metric dict
                            exec(f'{m.replace("-", "")}=dict()')
                    for m in metric_set:
                        exec(f'{m.replace("-", "")}[eid]=float(d[\'{m}\'])')
    metric_set_str = str(metric_set).replace('\'', '').strip('[').strip(']').replace("-", "")
    exec(f'out_list_ = [parameters,{metric_set_str}]', globals(), locals())
    out_list = locals()["out_list_"]
    out_df = pd.DataFrame.from_records(out_list).T
    out_df.columns = ['parameters', *metric_set]
    return out_df, metric_set


def skip_results(res_file):
    """
    Case 1: Previous results exists and summarized => skip => Return True
    Case 2: Previous results exists but unfinished => clear and rerun => Clear and return False
    Case 3: Previous results doesn't exist => run => Return False

    """
    if os.path.isfile(res_file):
        with open(res_file, 'r') as f:
            for line in f.readlines():
                if line[0] == '{':
                    d = ast.literal_eval(line.strip('\n'))
                    if 'avg_' in list(d.keys())[0]:
                        # ! Case 1: Previous results exists and summarized => skip => Return True
                        return True
            # ! Case 2: Previous results exists but unfinished => clear and rerun => Clear and return False
            os.remove(res_file)
            print(f'Resuming from {res_file}')
            return False
    else:
        # ! Case 3: Previous results doesn't exist => run => Return False
        return False
