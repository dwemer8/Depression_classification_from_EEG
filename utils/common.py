import torch
import numpy as np
import random
import os
from copy import deepcopy

def get_object_name(method):
    return method.__name__.replace("_score", "")

def objectName(object):
    return str(type(object)).split(".")[-1].replace("\'>", "")

def seed_all(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

def printLog(s, logfile=None):
    print(s)
    if logfile is not None:
        print(s, file=logfile)

def upd(config, d):
    config = deepcopy(config)
    for k, v in config.items():
        if k in d:
            if isinstance(v, dict):
                config[k] = upd(v, d[k])
            else:
                config[k] = d[k]
    return config

class Config():
    def __init__(self, config):
        self.config = config

    def upd(self, d):
        config = deepcopy(self.config)
        for k, v in config.items():
            if k in d:
                if isinstance(v, dict):
                    config[k] = upd(v, d[k])
                else:
                    config[k] = d[k]
        return config