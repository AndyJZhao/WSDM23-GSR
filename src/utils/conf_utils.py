import utils.util_funcs as uf
from abc import abstractmethod, ABCMeta
import os
from utils.proj_settings import RES_PATH, SUM_PATH, TEMP_PATH


class ModelConfig(metaclass=ABCMeta):
    """

    """

    def __init__(self, model):
        self.model = model
        self.exp_name = 'default'
        self.seed = 0
        self.birth_time = uf.get_cur_time(t_format='%m_%d-%H_%M_%S')
        # Other attributes
        self._model_conf_list = None
        self._interested_conf_list = ['model']
        self._file_conf_list = ['checkpoint_file', 'res_file']

    def __str__(self):
        # Print all attributes including data and other path settings added to the config object.
        return str({k: v for k, v in self.model_conf.items() if k != '_interested_conf_list'})

    @property
    @abstractmethod
    def f_prefix(self):
        # Model config to str
        return ValueError('The model config file name must be defined')

    @property
    @abstractmethod
    def checkpoint_file(self):
        # Model config to str
        return ValueError('The checkpoint file name must be defined')

    @property
    def res_file(self):
        return f'{RES_PATH}{self.model}/{self.dataset}/l{self.train_percentage:02d}/{self.f_prefix}.txt'
        # return f'{RES_PATH}{self.model}/{self.dataset}/{self.f_prefix}.txt'

    @property
    def model_conf(self):
        # Print the model settings only.
        return {k: self.__dict__[k] for k in self._model_conf_list}

    def get_sub_conf(self, sub_conf_list):
        # Generate subconfig dict using sub_conf_list
        return {k: self.__dict__[k] for k in sub_conf_list}

    def update__model_conf_list(self, new_conf=[]):
        # Maintain a list of interested configs
        other_configs = ['_model_conf_list', '_file_conf_list']
        if len(new_conf) == 0:  # initialization
            self._model_conf_list = sorted(list(self.__dict__.copy().keys()))
            for uninterested_config in other_configs:
                self._model_conf_list.remove(uninterested_config)
        else:
            self._model_conf_list = sorted(self._model_conf_list + new_conf)

    def update_modified_conf(self, conf_dict):
        self.__dict__.update(conf_dict)
        self._interested_conf_list += list(conf_dict)
        unwanted_items = ['log_on', 'gpu', 'train_phase', 'num_workers']
        for item in unwanted_items:
            if item in self._interested_conf_list:
                self._interested_conf_list.remove(item)
        uf.mkdir_list([getattr(self, _) for _ in self._file_conf_list])


class SimpleObject():
    """
    Dict2Config: convert dict to a config object for better attribute acccessing
    """

    def __init__(self, conf):
        self.__dict__.update(conf)
