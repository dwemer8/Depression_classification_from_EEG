#!/usr/bin/env python
# coding: utf-8

'''
# Environment variables

In this section one defines environment variables. 
Because I used this notebook on number of machines, I implemented class especially for this. 
You may not needed in one and use just simple definitions.
'''

from src.utils.system_variables import SystemVariables

# choose system according your current machine
# SYSTEM_NAME = "Windows"
# SYSTEM_NAME = "Colab"
# SYSTEM_NAME = "Kaggle"
SYSTEM_NAME = "Linux"

sv = SystemVariables(SYSTEM_NAME)
PROJECT_FOLDER = sv.get_project_folder()
SRC_FOLDER = sv.get_src_folder()
OUTPUT_FOLDER = sv.get_output_folder()
TUAB_DIRECTORY, TUAB_TRAIN, TUAB_EVAL = sv.get_TUAB_folders()
DEPR_ANON_DIRECTORY = sv.get_depr_anon_folder()
INHOUSE_DIRECTORY = sv.get_inhouse_folder()

print(SYSTEM_NAME)
print()

print(f"{PROJECT_FOLDER = }")
print(f"{SRC_FOLDER = }")
print(f"{OUTPUT_FOLDER = }")
print()

print(f"{TUAB_DIRECTORY = }")
print(f"{TUAB_TRAIN = }")
print(f"{TUAB_EVAL = }")
print()

print(f"{DEPR_ANON_DIRECTORY = }")
print()

print(f"{INHOUSE_DIRECTORY = }")
print()

'''
Common libraries
'''

import warnings
warnings.simplefilter("ignore")

import os
import sys
import json
import random
from copy import deepcopy

import numpy as np

import wandb
wandb.login(key='1b8e8dc9dcf1a34397a04197c4826d3fe7441dae')

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

'''
Project libraries
'''

sys.path.append(SRC_FOLDER)

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '1')
# get_ipython().run_line_magic('aimport', 'utils')

from src.utils import FIXED_SEEDS, SEED_PLACEHOLDER
from src.utils.common import upd, Config, read_json_with_comments, replace_placeholder
from src.trainer.do_experiment import do_experiment


'''
# Config
'''

train_config = read_json_with_comments("configs/train_config.json")
logger_config = read_json_with_comments("configs/logger_config.json")
model_config = read_json_with_comments("configs/model_config.json")
dataset_config = read_json_with_comments(
    "configs/dataset_config.json",
    {
        "{TUAB_DIRECTORY}": TUAB_DIRECTORY,
        "{INHOUSE_DIRECTORY}": INHOUSE_DIRECTORY,
        "{DEPR_ANON_DIRECTORY}": DEPR_ANON_DIRECTORY,
    }
)
optimizer_config = read_json_with_comments("configs/optimizer_config.json")
scheduler_config = read_json_with_comments("configs/scheduler_config.json")
ml_config = read_json_with_comments(
    "configs/ml_config.json",
    {
        SEED_PLACEHOLDER: '\"' + SEED_PLACEHOLDER + '\"'#it will be replaced further
    }
)
ml_validation_config = read_json_with_comments(
    "configs/ml_validation_config.json",
    {
        SEED_PLACEHOLDER: '\"' + SEED_PLACEHOLDER + '\"'#it will be replaced further
    }
)

default_config = {
    "project_name": 'EEG_depression_classification',
    "method": "direct restoration",
    "save_path" : OUTPUT_FOLDER + 'model_weights/',
    "log_path" : OUTPUT_FOLDER + "logs/",
    "hash": "0", #will be replaced further 
    "run_hash": "0", #will be replaced further 
    "run_name": "test", #will be replaced further 
    "seed": 0, #will be replaced further 
    "n_seeds": 3, #no more than length of FIXED_SEEDS from utils
    "display_mode": "terminal", #ipynb/terminal
    "save_after_test": True,
    
    "dataset": dataset_config,
    "model": model_config,
    "optimizer" : optimizer_config,
    "scheduler": scheduler_config,
    "train": train_config,
    "ml": ml_config,
    "ml_validation": ml_validation_config,
    "logger": logger_config,
}

with open("configs/default_config.json", "w") as f: json.dump(default_config, f, indent=4, ensure_ascii=False)

'''
# Define experiments
'''

# experiments = [default_config]
experiments = []
for pretrain_config in [
    None,
    # {
    #     "source":{
    #         "name": "TUAB", #inhouse_dataset/depression_anonymized/TUAB
    #         "file": f"{TUAB_DIRECTORY}fz_cz_pz/dataset_128_1.0.pkl" #TUAB_DIRECTORY + "dataset_128_1.0.pkl",
    #     },
    #     "size": None,
    #     "n_samples": None, #will be updated in train function,
    #     "preprocessing":{
    #         "is_squeeze": False, 
    #         "is_unsqueeze": False, 
    #         "t_max": None
    #     },
    #     "steps": {
    #         "start_epoch": 1, # including #!!CHECK
    #         "end_epoch": 6, # excluding, #!!CHECK
    #         "step_max" : None #!!CHECK
    #     }
    # }
]:
    hash = hex(random.getrandbits(32))
    default_config.update({"hash": hash})
    dc = Config(default_config)
    for beta in [0.075, 1.0, 2.0, 0.5, 1.5, 0.1, 0.25, 0.75, 1.25, 1.75]:
        cc = dc.upd({
            "run_name": f"{'' if pretrain_config is None else 'pretrain, '}beta_{beta}, bVAE_honke_higgins_2010_15274",
            "model": {
                "framework": {
                    "beta" : beta
                }
            },
            "dataset" : {
                "train": {
                    "pretrain": pretrain_config,
                    "train" : {
                        "steps": {
                            "start_epoch": 1 if pretrain_config is None else 6, # including #!!CHECK
                            "end_epoch": 101 if pretrain_config is None else 106, # excluding, #!!CHECK
                        }
                    }
                }
            },
        })
        experiments.append(cc)

print("N experiments:", len(experiments))
for exp in experiments:
    print(exp['hash'], exp['run_name'])

'''
# Conducting experiments
'''
all_results = []
rng = np.random.default_rng()
for config in experiments:
    run_hash = rng.bytes(8).hex() #same for different seeds
    config.update({"run_hash": run_hash})
    
    for seed in FIXED_SEEDS[:config["n_seeds"]]:
        config_copy = deepcopy(config)
        config_copy["seed"] = seed
        config_copy = json.loads(json.dumps(config_copy).replace('\"' + SEED_PLACEHOLDER + '\"', str(seed)))

        exp_results = {
            config_copy["run_name"] : do_experiment(config_copy, device=device, verbose=3)
        }
        
        all_results.append(exp_results)
        with open(os.path.join(config_copy["log_path"], "all_results_" + config_copy["model"]["model_description"].replace(" ", "_").replace("/", ".")), "w") as f:
            json.dump(exp_results, f, indent=4, ensure_ascii=False)
        with open("current_results.json", "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)