import torch
import numpy as np
import random
import os
from copy import deepcopy
import json

def check_tags(s, tags):
    for tag in tags:
        if tag in s:
            return True
    return False

def remove_files(root, tags):
    for directory, _, file_names in os.walk(root):
        n = 0
        for file_name in file_names:
            if check_tags(file_name, tags):
                os.remove(os.path.join(directory, file_name))
                n += 1
        print(f"{n} files was removed from {directory}")

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

def printLog(*args, logfile=None, **kwargs):
    print(*args, **kwargs)
    if logfile is not None:
        print(*args, file=logfile, **kwargs)

def upd(config, d):
    '''
    It updates only existing keys!!!
    '''
    config = deepcopy(config)
    for k, v in config.items():
        if k in d:
            if isinstance(v, dict) and isinstance(d[k], dict):
                config[k] = upd(v, d[k])
            else:
                config[k] = d[k]
    return config

class Config():
    def __init__(self, config):
        self.config = config

    def upd(self, d):
        '''
        It updates only existing keys!!!
        '''
        config = deepcopy(self.config)
        for k, v in config.items():
            if k in d:
                if isinstance(v, dict) and isinstance(d[k], dict):
                    config[k] = upd(v, d[k])
                else:
                    config[k] = d[k]
        return config

def replace(s, replacements):
    for substring in replacements: s = s.replace(substring, replacements[substring])
    return s

def read_json_with_comments(path, replacements=None):
    f = open(path, "r")
    lines = list(map(lambda line: line.split("#")[0].replace("\n", ""), f.readlines()))
    if replacements is not None: lines = list(map(lambda line: replace(line, replacements), lines))
    lines = "\n".join(lines)
    return json.loads(lines)

def replace_placeholder(d, placeholder, value):
    d = deepcopy(d)
    for k, v in d.items():
        if v == placeholder:
            d[k] = value
        elif isinstance(v, dict):
            d[k] = replace_placeholder(v, placeholder, value)
    return d

def wrap_field(d, field):
    try:
        return d[field]
    except:
        return {}
    
def check_instance(object, types):
    for class_type in types:
        if isinstance(object, class_type):
            return True
    return False