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
    "n_seeds": 1, #no more than length of FIXED_SEEDS from utils
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
for model_config in [
    {
        "model": "AE_parametrized",
        "decoder": {
            "in_conv_config": {
                "n_convs": 2,
                "activation": "Sigmoid",
                "in_channels": 24,
                "kernel_size": 3,
                "out_channels": 24
            },
            "up_blocks_config": [
                {
                    "n_convs": 2,
                    "activation": "Sigmoid",
                    "in_channels": 24,
                    "kernel_size": 3,
                    "out_channels": 12
                },
                {
                    "n_convs": 2,
                    "activation": "Sigmoid",
                    "in_channels": 12,
                    "kernel_size": 3,
                    "out_channels": 6
                },
                {
                    "n_convs": 2,
                    "activation": "Sigmoid",
                    "in_channels": 6,
                    "kernel_size": 1,
                    "out_channels": 3,
                    "normalize_last": True
                }
            ]
        },
        "encoder": {
            "out_conv_config": {
                "n_convs": 2,
                "activation": "Sigmoid",
                "in_channels": 24,
                "kernel_size": 3,
                "out_channels": 24,
                "normalize_last": True
            },
            "down_blocks_config": [
                {
                    "n_convs": 2,
                    "activation": "Sigmoid",
                    "in_channels": 3,
                    "kernel_size": 7,
                    "out_channels": 6
                },
                {
                    "n_convs": 2,
                    "activation": "Sigmoid",
                    "in_channels": 6,
                    "kernel_size": 7,
                    "out_channels": 12
                },
                {
                    "n_convs": 2,
                    "activation": "Sigmoid",
                    "in_channels": 12,
                    "kernel_size": 5,
                    "out_channels": 24
                }
            ]
        },
        "framework": {
            "loss_reduction": "mean",
            "first_decoder_conv_depth": 24
        },
        "loss_reduction": "mean",
        "model_description": "duration, finetune, 60 s, AE",
        "artifact" : "dmitriykornilov_team/EEG_depression_classification-PR-AUC/AE_parametrized:v23",
        "file": "85_epoch.pth"
    },
    # {
    #     "model": "AE_parametrized",
    #     "decoder": {
    #         "in_conv_config": {
    #             "n_convs": 2,
    #             "activation": "Sigmoid",
    #             "in_channels": 24,
    #             "kernel_size": 3,
    #             "out_channels": 24
    #         },
    #         "up_blocks_config": [
    #             {
    #                 "n_convs": 2,
    #                 "activation": "Sigmoid",
    #                 "in_channels": 24,
    #                 "kernel_size": 3,
    #                 "out_channels": 12
    #             },
    #             {
    #                 "n_convs": 2,
    #                 "activation": "Sigmoid",
    #                 "in_channels": 12,
    #                 "kernel_size": 3,
    #                 "out_channels": 6
    #             },
    #             {
    #                 "n_convs": 2,
    #                 "activation": "Sigmoid",
    #                 "in_channels": 6,
    #                 "kernel_size": 1,
    #                 "out_channels": 3,
    #                 "normalize_last": True
    #             }
    #         ]
    #     },
    #     "encoder": {
    #         "out_conv_config": {
    #             "n_convs": 2,
    #             "activation": "Sigmoid",
    #             "in_channels": 24,
    #             "kernel_size": 3,
    #             "out_channels": 24,
    #             "normalize_last": True
    #         },
    #         "down_blocks_config": [
    #             {
    #                 "n_convs": 2,
    #                 "activation": "Sigmoid",
    #                 "in_channels": 3,
    #                 "kernel_size": 7,
    #                 "out_channels": 6
    #             },
    #             {
    #                 "n_convs": 2,
    #                 "activation": "Sigmoid",
    #                 "in_channels": 6,
    #                 "kernel_size": 7,
    #                 "out_channels": 12
    #             },
    #             {
    #                 "n_convs": 2,
    #                 "activation": "Sigmoid",
    #                 "in_channels": 12,
    #                 "kernel_size": 5,
    #                 "out_channels": 24
    #             }
    #         ]
    #     },
    #     "framework": {
    #         "loss_reduction": "mean",
    #         "first_decoder_conv_depth": 24
    #     },
    #     "loss_reduction": "mean",
    #     "model_description": "duration, 60 s, AE",
    #     "artifact" : "dmitriykornilov_team/EEG_depression_classification-PR-AUC/AE_parametrized:v31",
    #     "file": "75_epoch.pth"
    # },
    # {
    #     "model": "VAE_deep",
    #     "model_description": "finetune, duration, 60 s, beta-VAE, 3 ch., 4/8/16/32, 7/7/5/3/3/3/3/1, Sigmoid",
    #     "loss_reduction" : "mean",
    #     "latent_dim": 16*32*60,
    #     "beta": 2,
    #     "first_decoder_conv_depth": 32,
    #     "artifact" : "dmitriykornilov_team/EEG_depression_classification-PR-AUC/VAE_deep:v48",
    #     "file": "85_epoch.pth"
    # },
    # {
    #     "model": "VAE_deep",
    #     "model_description": "duration, inhouse_dataset, 60 s, beta-VAE, 3 ch., 4/8/16/32, 7/7/5/3/3/3/3/1, Sigmoid",
    #     "loss_reduction" : "mean",
    #     "latent_dim": 16*32*60,
    #     "beta": 2,
    #     "first_decoder_conv_depth": 32,
    #     "artifact" : "dmitriykornilov_team/EEG_depression_classification-PR-AUC/VAE_deep:v24",
    #     "file": "50_epoch.pth"
    # },
]:
    hash = hex(random.getrandbits(32))
    default_config.update({"hash": hash})
    dc = Config(default_config)
    for channel_name, channel_index in zip([
            # "None", 
            "fz", 
            # "cz", 
            # "pz"
        ], [
            # None, 
            0, 
            # 1, 
            # 2
        ]
    ):
        ml_to_train = True
        if channel_name is not None:   
            model_config.update({
                "artifact": "dmitriykornilov_team/EEG_depression_classification/duration._finetune._60_s._AE:v14",
                "file": "0_epoch_final.pth",
                "ml_artifact" : "dmitriykornilov_team/EEG_depression_classification/duration._finetune._60_s._AE_svm.SVC:v10",
                "ml_file": "0_epoch_svm.SVC_final.pth"
            })
            ml_to_train = False
        cc = dc.upd({
            "run_name": f"zero out {channel_name}, " + model_config["model_description"],
            "model": model_config,
            "dataset": {
                "val": {
                    "source":{
                        "name": "inhouse_dataset",
                        "file": INHOUSE_DIRECTORY + f"fz_cz_pz/dataset_128_60.0.pkl", #TUAB_DIRECTORY + "dataset_128_1.0.pkl",
                    },
                    "preprocessing":{
                        "is_squeeze": False,
                        "is_unsqueeze": False,
                        "t_max": None,
                        "transforms": [
                            "zero_out_channel"
                        ],
                        "transforms_kwargs": [
                            {
                                "channel": channel_index
                            }
                        ]
                    }
                },
                "test": {
                    "source":{
                        "name": "inhouse_dataset",
                        "file": INHOUSE_DIRECTORY + f"fz_cz_pz/dataset_128_60.0.pkl", #TUAB_DIRECTORY + "dataset_128_1.0.pkl",
                    },
                    "preprocessing":{
                        "is_squeeze": False,
                        "is_unsqueeze": False,
                        "t_max": None,
                        "transforms": [
                            "zero_out_channel"
                        ],
                        "transforms_kwargs": [
                            {
                                "channel": channel_index
                            }
                        ]
                    }
                },
            },
            "ml": {
                "ml_to_train": ml_to_train
            }
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
            config_copy['model']["model_description"] : do_experiment(config_copy, device=device, verbose=3)
        }
        
        all_results.append(exp_results)
        with open(os.path.join(config_copy["log_path"], "all_results_" + config_copy["model"]["model_description"].replace(" ", "_").replace("/", ".")), "w") as f:
            json.dump(exp_results, f, indent=4, ensure_ascii=False)
        with open("current_results.json", "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)