from copy import deepcopy
import importlib

import sklearn
from sklearn.pipeline import Pipeline

from .models_evaluation import *

def get_pipeline(stages_config):
    stages_config = deepcopy(stages_config)
    stages = []
    for k, v in stages_config.items():
        module = importlib.import_module("sklearn." + ".".join(v.split(".")[:-1]))
        stages.append((
            k, 
            getattr(module, v.split(".")[-1])()
        ))
    return Pipeline(stages)

def get_eval_functions(eval_functions_config):
    eval_functions_config = deepcopy(eval_functions_config)
    funcs = []
    for k in eval_functions_config:
        module = importlib.import_module("utils.models_evaluation", package='..')
        funcs.append(getattr(module, k))
    return funcs

def get_eval_functions_kwargs(kwargs_list):
    kwargs_list = deepcopy(kwargs_list)
    for kwargs in kwargs_list:
        kwargs["cv_scorer"] = getattr(sklearn.metrics, kwargs["cv_scorer"])
        
        for tag in ["metrics", "metrics_for_CI"]:
            if tag in kwargs:
                func_mode = []
                for func, mode in kwargs[tag]:
                    func_mode.append((getattr(sklearn.metrics, func), mode))
                kwargs[tag] = func_mode
    return kwargs_list

def parse_ml_config(config):
    config = deepcopy(config)
    config["ml_model"] = get_pipeline(config["ml_model"])
    config["ml_eval_function"] = get_eval_functions(config["ml_eval_function"])
    config["ml_eval_function_kwargs"] = get_eval_functions_kwargs(config["ml_eval_function_kwargs"])
    return config