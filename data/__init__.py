"""create dataset and dataloader"""
import importlib
import logging
import torch
import torch.utils.data
import numpy as np
import random
import os.path as osp
from utils.util import scandir
logger = logging.getLogger('base')

__all__ = ['create_dataset', 'create_dataloader']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_dataset.py')
]
# import all the dataset modules  (所有以_dataset.py结尾的文件导入)
_dataset_modules = [
    importlib.import_module(f'data.{file_name}')
    for file_name in dataset_filenames
]


def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']

    # dynamic instantiation
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} '
        'is created.')
    return dataset



def create_dataloader(dataset, dataset_opt, sampler=None):
    '''
    :param dataset: Dataset
    :param dataset_opt: train or val in datasets in yml
    :param opt: dataset_opt
    :param sampler:
    :return: Dataloader
    '''
    mode = dataset_opt['mode']
    if mode == 'train':
        num_workers = dataset_opt['num_worker_per_gpu']
        batch_size = dataset_opt['batch_size_per_gpu']
        # shuffle = True
        shuffle = dataset_opt['use_shuffle']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler,
                                           pin_memory=True)
    else:
        batch_size = dataset_opt['batch_size_per_gpu']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                           pin_memory=False)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
