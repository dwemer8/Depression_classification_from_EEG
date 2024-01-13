import torch
import numpy as np
import random
import os

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