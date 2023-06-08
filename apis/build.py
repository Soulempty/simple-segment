import sys
import copy
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
sys.path.append("..") 
from models.registry import *

MODULES = [MODELS, BACKBONES, DATASETS, TRANSFORMS, LOSSES]

def is_meta_type(val):
    return isinstance(val, dict) and 'type' in val

def build_module(cfg):
    cfg = copy.deepcopy(cfg)
    if 'type' not in cfg:
        raise RuntimeError("It is not possible to create a component object from {}, as 'type' is not specified.".format(cfg))
    class_type = cfg.pop('type')
    module = None
    for m in MODULES:
        if class_type in m.module_dict:
            module = m.module_dict[class_type]
            break
    params = {}

    for key, val in cfg.items():
        if is_meta_type(val):
            params[key] = build_module(val)
        elif isinstance(val, list):
            params[key] = [build_module(item) if is_meta_type(item) else item for item in val]
        else:
            params[key] = val
    try:
        obj = module(**params)
    except Exception as e:
        raise RuntimeError(f"fail building {class_type},the error is {e}.")
    return obj