import importlib
import logging
from utils.util import scandir
import os.path as osp
logger = logging.getLogger('base')


# automatically scan and import models modules
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('_model.py')
]
# import all the models modules (所有以_model.py结尾的文件导入)
_model_modules = [
    importlib.import_module(f'models.{file_name}')
    for file_name in model_filenames
]


def create_model(opt,args):
    """Create models.

    Args:
        opt (dict): Configuration. It constains:  opt
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    # dynamic instantiation
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)      # _model.py文件里检索需要model_type
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(opt, args)
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model

