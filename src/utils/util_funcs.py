import os
import sys
import logging
import pickle
import numpy as np
import time
import datetime
import pytz
from utils.proj_settings import SUM_PATH

import socket


# * ============================= Init =============================

def exp_init(seed, gpu_id):
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    init_random_state(seed)
    # Torch related import should be imported afterward setting


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    import dgl
    dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper


def is_runing_on_local():
    try:
        host_name = socket.gethostname()
        if 'MacBook' in host_name:
            return True
    except:
        print("Unable to get Hostname and IP")
    return False


# * ============================= Print Related =============================
def subset_dict(d, sub_keys):
    return {k: d[k] for k in sub_keys if k in d}


def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')


def block_log():
    sys.stdout = open(os.devnull, 'w')
    logger = logging.getLogger()
    logger.disabled = True


def enable_logs():
    # Restore
    sys.stdout = sys.__stdout__
    logger = logging.getLogger()
    logger.disabled = False


def print_log(log_dict):
    log_ = lambda log: f'{log:.4f}' if isinstance(log, float) else f'{log:04d}'
    print(' | '.join([f'{k} {log_(v)}' for k, v in log_dict.items()]))


def mp_list_str(mp_list):
    return '_'.join(mp_list)


# * ============================= File Operations =============================

def write_nested_dict(d, f_path):
    def _write_dict(d, f):
        for key in d.keys():
            if isinstance(d[key], dict):
                f.write(str(d[key]) + '\n')

    with open(f_path, 'a+') as f:
        f.write('\n')
        _write_dict(d, f)


def save_pickle(var, f_name):
    mkdir_list([f_name])
    pickle.dump(var, open(f_name, 'wb'))
    print(f'File {f_name} successfully saved!')


def load_pickle(f_name):
    return pickle.load(open(f_name, 'rb'))


def clear_results(dataset, model, exp_name):
    res_path = f'{SUM_PATH}{dataset}/{model}/{exp_name}/'
    os.system(f'rm -rf {res_path}')
    print(f'Results in {res_path} are cleared.')


# * ============================= Path Operations =============================

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def get_grand_parent_dir(f_name):
    from pathlib import Path
    if '.' in f_name.split('/')[-1]:  # File
        return get_grand_parent_dir(get_dir_of_file(f_name))
    else:  # Path
        return f'{Path(f_name).parent}/'


def get_abs_path(f_name, style='command_line'):
    # python 中的文件目录对空格的处理为空格，命令行对空格的处理为'\ '所以命令行相关需 replace(' ','\ ')
    if style == 'python':
        cur_path = os.path.abspath(os.path.dirname(__file__))
    elif style == 'command_line':
        cur_path = os.path.abspath(os.path.dirname(__file__)).replace(' ', '\ ')

    root_path = cur_path.split('src')[0]
    return os.path.join(root_path, f_name)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:

        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def mkdir_list(p_list, use_relative_path=True, log=True):
    """Create directories for the specified path lists.
        Parameters
        ----------
        p_list :Path lists

    """
    # ! Note that the paths MUST END WITH '/' !!!
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    for p in p_list:
        p = os.path.join(root_path, p) if use_relative_path else p
        p = os.path.dirname(p)
        mkdir_p(p, log)


# * ============================= Time Related =============================

def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


class Dict2Config():
    """
    Dict2Config: convert dict to a config object for better attribute acccessing
    """

    def __init__(self, conf):
        self.__dict__.update(conf)


# * ============================= Itertool Related =============================

def lot_to_tol(list_of_tuple):
    # list of tuple to tuple lists
    # Note: zip(* zipped_file) is an unzip operation
    return map(list, zip(*list_of_tuple))
# * ============================= Torch Related =============================
