import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import math
import cv2
import random
import logging
from datetime import datetime
from termcolor import colored
import torch.distributed as dist
from torch._six import inf


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(img, img_path, mode='BGR'):
    if mode=='BGR':
        cv2.imwrite(img_path, img)
    elif mode == 'RGB':
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) 

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def Generate_rp(result_dir,ippath):
    return os.path.join(result_dir, os.path.splitext(os.path.basename(ippath))[0] + '.png')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    '''
    lg = logging.getLogger(logger_name)
    fmt = '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s'
    color_fmt = colored('%(asctime)s.%(msecs)03d','green') + '- %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt=color_fmt,
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    lg.propagate = False
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def reduce_mean(tensor, nprocs):
    '''各个node上的值加起来求平均, 均值返回各个node
        [1,2,3]  ->  [2,3,4]
        [3,4,5]  ->  [2,3,4]
    '''
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt.item()/nprocs
    return rt

def get_pretrain_model(model,weights,yaml_path,cuda=True,eval=True):
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_path, mode='r'), Loader=Loader)
    type = x['network_g'].pop('type')
    model_restoration = model(**x['network_g'])
    if cuda:
        model_restoration.cuda()
        model_restoration = nn.DataParallel(model_restoration)
    else:
        print('cpu')
        model_restoration.cpu()
    model_restoration.load_state_dict(torch.load(weights), strict=True)
    if eval:
        model_restoration.eval()
    return model_restoration


def random_get_paired_img(type='rain', filename=None):
    """
    input: rain/blur/noise
    return: {lq:path,gt:path} paired paths for pointed degradation type
    """
    path_dict = {'rain': './Test100-Rain800test',
                 'blur': './test/GoPro',
                 'noise': './val/SIDD'}

    # get input & target dir path
    input_path = os.path.join(path_dict[type], 'input')
    target_path = os.path.join(path_dict[type], 'target')

    # assert 'input' in input_path, 'Warning: dir path dose not cantain input'
    # assert 'target' in target_path, 'Warning: dir path dose not cantain target'

    # random get a img name
    files = os.listdir(input_path)
    if not filename:
        indice = random.sample(range(0, len(files)), k=1)[0]
        filename = files[indice]

    return {'lq_path': os.path.join(input_path, filename), 'gt_path': os.path.join(target_path, filename)}


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

