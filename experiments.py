#!/usr/bin/env python
# coding: utf-8

# # Libraries installation

# In[1]:


# !pip uninstall torch torchvision torchaudio -y
# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# !pip install mne -q
# !pip install wandb -q
# !pip install tensorboard -q


# # Constants and libraries

# In[2]:


'''
In this section one defines environment variables. 
Because I used this notebook on number of machines, I implemented class especially for this. 
You may not needed in one and use just simple definitions.
'''

from system_variables import SystemVariables

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


# In[3]:


import warnings
warnings.simplefilter("ignore")

import os
import sys
import pickle
import json
import random
from IPython.display import display, clear_output
import traceback

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
from matplotlib import rc
rc('animation', html='jshtml')

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

from tqdm.auto import tqdm as tqdm_auto

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# get_ipython().run_line_magic('load_ext', 'tensorboard')

import wandb
wandb.login(key='1b8e8dc9dcf1a34397a04197c4826d3fe7441dae')
# get_ipython().system('wandb login 1b8e8dc9dcf1a34397a04197c4826d3fe7441dae')

import mne

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# In[4]:


sys.path.append(SRC_FOLDER)

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '1')
# get_ipython().run_line_magic('aimport', 'utils')

from utils import SEED
from utils.common import objectName, seed_all, printLog, upd, Config, read_json_with_comments
from utils.models_evaluation import evaluateClassifier, evaluateRegressor, evaluateClassifier_inner_outer_cv
from utils.data_reading import DataReader
from utils.plotting import dataset_hists, plotData, dict_to_df, printDatasetMeta, printDataloaderMeta, plotSamplesFromDataset
from utils.dataset import InMemoryDataset
from utils.logger import Logger
from utils.parser import parse_ml_config
from utils.early_stopper import EarlyStopper

from models import get_model, load_weights_from_wandb

from training import train_eval


# # Train function

# In[5]:


def train(config, verbose=0):
    try:
        if config["log_path"] is not None: logfile = open(os.path.join(config["log_path"], config["model"]["model_description"].replace(" ", "_").replace("/", ".")), "a")
        else: logfile = None

        #############
        #Data reading
        #############
        if verbose - 1 > 0: printLog("Data reading", logfile=logfile)

        #datasets
        #only two setups still supported!
        pretrain_config = config["dataset"]["train"]["pretrain"]
        train_config = config["dataset"]["train"]["train"]
        val_config = config["dataset"]["val"]
        test_config = config["dataset"]["test"]
        if pretrain_config is not None:
            pretrain_reader = DataReader(
                pretrain_config["source"]["file"], 
                dataset_type=pretrain_config["source"]["name"],
                verbose=(verbose-1)
            )
            pretrain_set, _, _ = pretrain_reader.split(
                train_size=pretrain_config["size"], val_size=0, test_size=0
            )
        
        if (pretrain_config["source"] if pretrain_config is not None else {}) != train_config["source"] == val_config["source"] == test_config["source"]:
            reader = DataReader(
                train_config["source"]["file"], 
                dataset_type=train_config["source"]["name"],
                verbose=(verbose-1)
            )
            train_set, val_set, test_set = reader.split(
                train_size=train_config["size"], val_size=val_config["size"], test_size=test_config["size"]
            )    
            
        elif (pretrain_config["source"] if pretrain_config is not None else {}) != train_config["source"] != val_config["source"] == test_config["source"] != pretrain_config["source"]:
            train_reader = DataReader(
                train_config["source"]["file"],
                dataset_type=train_config["source"]["name"],
                verbose=(verbose-1),
            )
            train_set, _, _ = train_reader.split(
                train_size=train_config["size"], val_size=0, test_size=0
            )

            val_test_reader = DataReader(
                test_config["source"]["file"],
                dataset_type=test_config["source"]["name"],
                verbose=(verbose-1),
            )
            _, val_set, test_set = val_test_reader.split(
                train_size=0, val_size=val_config["size"], test_size=test_config["size"]
            )
            
        else:
            raise NotImplementedError("Unsupported datasets configuration")

        if pretrain_config is not None: chunks_pretrain, targets_pretrain = pretrain_set["chunk"], pretrain_set["target"]
        chunks_train, chunks_val, chunks_test = train_set["chunk"], val_set["chunk"], test_set["chunk"]
        targets_train, targets_val, targets_test = train_set["target"], val_set["target"], test_set["target"]

        #TODO: add to upd function ability to add new fields
        if pretrain_config is not None: 
            config["dataset"] = upd(config["dataset"], {
                "train": {
                    "pretrain": {"n_samples": len(chunks_pretrain)}
                }
            })
            
        config["dataset"] = upd(config["dataset"], {
            "samples_shape": chunks_train[0].shape,
            "train": {
                "train": {"n_samples": len(chunks_train)}
            },
            "val": {"n_samples": len(chunks_val)},
            "test": {"n_samples": len(chunks_test)},
        })

        if pretrain_config is not None: 
            pretrain_dataset = InMemoryDataset(
                chunks_pretrain, **pretrain_config["preprocessing"]
            )
        train_dataset = InMemoryDataset(
            chunks_train, **train_config["preprocessing"]
        )
        val_dataset = InMemoryDataset(
            chunks_val, **val_config["preprocessing"]
        )
        test_dataset = InMemoryDataset(
            chunks_test, **test_config["preprocessing"]
        )
    
        if verbose - 2 > 0: 
            printDatasetMeta(train_dataset, val_dataset, test_dataset, pretrain_dataset=None if pretrain_config is None else pretrain_dataset)
            if config.get("display_mode", "terminal") == "ipynb": plotSamplesFromDataset(train_dataset)
    
        #Dataloader
        if pretrain_config is not None: pretrain_dataloader = DataLoader(pretrain_dataset, shuffle=True, **config["dataset"]['dataloader'])
        train_dataloader = DataLoader(train_dataset, shuffle=True, **config["dataset"]['dataloader'])
        val_dataloader = DataLoader(val_dataset, shuffle=False, **config["dataset"]['dataloader'])
        test_dataloader = DataLoader(test_dataset, shuffle=False, **config["dataset"]['dataloader'])
    
        if verbose - 2 > 0: printDataloaderMeta(train_dataloader, val_dataloader, test_dataloader, pretrain_dataloader=None if pretrain_config is None else pretrain_dataloader)
    
        #Model
        config["model"].update({
            "input_dim" : train_dataset[0].shape,
        })
        model, config["model"] = get_model(config["model"])
        model = model.to(device)
        if verbose - 1 > 0: printLog('model ' + config["model"]['model_description'] + ' is created', logfile=logfile)
        if verbose - 2 > 0: printLog(model)
    
        #Download weights
        if "artifact" in config["model"] and "file" in config["model"]:
            model = load_weights_from_wandb(model, config["model"]["artifact"], config["model"]["file"], verbose=verbose)
    
        # TESTS
        model.eval()
        test_data_point = train_dataset[0][None].to(device)
        inference_result = model(test_data_point)
        reconstruct_result = model.reconstruct(test_data_point)
        encode_result = model.encode(test_data_point)
        if verbose - 1 > 0: 
            printLog(f"Test data point shape: {test_data_point.shape}", logfile=logfile)
            printLog(f"Test inference result length: {len(inference_result)}", logfile=logfile)
            printLog(f"Test reconstruct shape: {reconstruct_result.shape}", logfile=logfile)
            printLog(f"Test encode shape: {encode_result.shape}", logfile=logfile)
    
        #optimizer and scheduler
        optimizer = getattr(torch.optim, config["optimizer"]["optimizer"])(model.parameters(), **config["optimizer"]["kwargs"])
        if verbose - 1 > 0: printLog(f'Optimizer {type(optimizer).__name__} is instantiated', logfile=logfile)
    
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["scheduler"])(optimizer, **config["scheduler"]["kwargs"])
        if verbose - 1 > 0: printLog(f'Scheduler {type(scheduler).__name__} is instantiated', logfile=logfile)
    
        logger = Logger(
            log_type=config["logger"]["log_type"], 
            run_name=config["model"]["model_description"],
            save_path=config["save_path"],
            model=model,
            model_name=config["model"]["model"],        
            project_name=config["project_name"],
            config=config,
            model_description=config["model"]["model_description"],
        #         log_dir = OUTPUT_FOLDER + "logs/"
        )

        #print whole config
        printLog('#################### ' + config["model"]["model_description"] + ' ####################', logfile=logfile)
        printLog(json.dumps(config, indent=4), logfile=logfile)
        
        #parse ml config
        #should be just before training because replace names by objects
        config["ml"] = parse_ml_config(config["ml"])
    
        #seed
        seed_all(SEED)
    
        #training
        # best_loss = np.inf
        best_clf_accuracy = -1
        best_model = None
        best_epoch = None
        final_model = None

        for curr_dataloader, dataset_config in zip(
            [None if pretrain_config is None else pretrain_dataloader, train_dataloader],
            [pretrain_config, train_config],  
        ):
            if dataset_config is None:
                continue

            early_stopper = EarlyStopper(**config['train']['early_stopping'])
            for epoch in tqdm_auto(range(dataset_config["steps"]['start_epoch'], dataset_config["steps"]['end_epoch'])):
                if verbose > 0: printLog(f"Epoch {epoch}", logfile=logfile)
                
                #######
                # train
                #######
                if verbose > 0: printLog("##### Training... #####", logfile=logfile)
                model, results = train_eval(
                    curr_dataloader,
                    model,
                    device=device,
                    mode="train",
                    optimizer=optimizer,
                    epoch=epoch,
                    logger=logger,
                    loss_coefs=config["train"]["loss_coefs"],
                    loss_reduction=config["model"]["loss_reduction"],
                    is_mask=(config["train"]["masking"]["n_masks"] != 0 and config["train"]["masking"]["mask_ratio"] != 0),
                    mask_ratio=config["train"]["masking"]["mask_ratio"],
                    step_max=dataset_config["steps"]["step_max"], 
                    verbose=verbose,
                )
                if results == {}: break
                if verbose > 0: 
                    if config.get("display_mode", "terminal") == "ipynb": display(dict_to_df(results))
                    else: print(dict_to_df(results).T)
                    for k in results: 
                        if isinstance(results[k], np.ndarray): results[k] = float(results[k].tolist())
                    print(json.dumps(results, indent=4), file=logfile)
        
                ############
                # validation
                ############
                if verbose > 0: printLog("##### Validation... #####", logfile=logfile)
                model, results = train_eval(
                    val_dataloader,
                    model,
                    device=device,
                    mode="validation",
                    test_dataset=val_dataset,
                    targets_test=targets_val,
                    check_period=config["train"]["validation"]["check_period"] if epoch % config['train']['validation']['check_period_per_epoch'] == 0 else None,
                    plot_period=config["train"]["validation"]["plot_period"] if (epoch % config['train']['validation']['plot_period_per_epoch'] == 0 and config.get("display_mode", "terminal") == "ipynb") else None,
                    epoch=epoch,
                    logger=logger,
                    loss_coefs=config["train"]["loss_coefs"],
                    loss_reduction=config["model"]["loss_reduction"],
                    step_max=dataset_config["steps"]["step_max"], 
                    verbose=verbose,
                    **config["ml"],
                )
                if results == {}: break
                if verbose > 0: 
                    if config.get("display_mode", "terminal") == "ipynb": display(dict_to_df(results))
                    else: print(dict_to_df(results).T)
                    for k in results: 
                        if type(results[k]) == np.ndarray: results[k] = float(results[k].tolist())
                    print(json.dumps(results, indent=4), file=logfile)
        
                scheduler.step(results['loss'])
                # logger.save_model(epoch)
                final_model = model
        
                zero_ml_tag = config["ml"]["ml_eval_function_tag"][0]
                last_tag = "cv" if zero_ml_tag == "cv" else "bs"
                if results.get(f'clf.{zero_ml_tag}.test.accuracy.{last_tag}', -1) > best_clf_accuracy:
                    best_clf_accuracy = results[f'clf.{zero_ml_tag}.test.accuracy.{last_tag}']
                    best_model = model
                    best_epoch = epoch
                    if verbose > 0: printLog(f"New best classifier accuracy = {best_clf_accuracy} on epoch {epoch}", logfile=logfile)
                
                # if results['loss'] < best_loss:
                #     best_loss = results['loss']
                #     best_model = model
                #     best_epoch = best_epoch
                #     if verbose > 0: printLog(f"New best loss = {best_loss} on epoch {epoch}", logfile=logfile)
                
                if early_stopper.early_stop(results['loss']): break
        
            # if curr_dataloader == train_dataloader: logger.save_model(dataset_config["steps"]['end_epoch'])
    
        ######
        # test
        ######
        results_all = {}
        for model, mode in zip([final_model, best_model], ["final", "test"]):
            if verbose > 0: printLog(f"##### Testing in {mode} mode... #####", logfile=logfile)
            _, results = train_eval(
                test_dataloader,
                model,
                device=device,
                mode=mode,
                test_dataset=test_dataset,
                targets_test=targets_test,
                check_period=1e10,
                plot_period=1e10 if config.get("display_mode", "terminal") == "ipynb" else None,
                epoch=train_config["steps"]['end_epoch'],
                logger=logger,
                loss_coefs=config["train"]["loss_coefs"],
                loss_reduction=config["model"]["loss_reduction"],
                step_max=train_config["steps"]["step_max"], 
                verbose=verbose,
                **config["ml"],
            )
            results_all[mode] = results
            if verbose > 0: 
                if config.get("display_mode", "terminal") == "ipynb": display(dict_to_df(results))
                else: print(dict_to_df(results).T)
                for k in results: 
                    if type(results[k]) == np.ndarray: results[k] = float(results[k].tolist())
                print(json.dumps(results, indent=4), file=logfile)
        
        logger.save_model(dataset_config["steps"]['end_epoch'], final_model)
        logger.save_model(best_epoch, best_model)
        logger.update_summary("validation.best_epoch", best_epoch)
        logger.finish()

        logfile.close()
        return results_all
        
    except Exception as error:
        # handle the exception
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback) 
        if logfile is not None: 
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=logfile) 
            logfile.close()
        return {}


# # Config

# In[6]:


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
        "{SEED}": str(SEED)
    }
)

default_config = {
    "project_name": 'EEG_depression_classification',
    "method": "direct restoration",
    "save_path" : OUTPUT_FOLDER + 'model_weights/',
    "log_path" : OUTPUT_FOLDER + "logs/",
    "hash": "0",
    "display_mode": "terminal", #ipynb/terminal
    
    "dataset": dataset_config,
    "model": model_config,
    "optimizer" : optimizer_config,
    "scheduler": scheduler_config,
    "train": train_config,
    "ml": ml_config,
    "logger": logger_config,
}

with open("configs/default_config.json", "w") as f: json.dump(default_config, f, indent=4, ensure_ascii=False)


# # Experiments

# In[7]:


import itertools

experiments = []
for is_pretrain in [False, True]:
    for t in [1, 15, 60]:
        hash = hex(random.getrandbits(32))
        default_config.update({"hash": hash})
        dc = Config(default_config)
        for p_dropout in [0.1, 0.3, 0.5, 0.7]:
            if is_pretrain:
                train_config = {
                    "pretrain": {
                        "source":{
                            "file": TUAB_DIRECTORY + f"fz_cz_pz/dataset_128_{t}.0.pkl",
                        },
                        "steps": {
                            "start_epoch": 0, # including
                            "end_epoch": 10, # excluding,
                        }
                    },
                    "train": {
                        "source":{
                            "file": INHOUSE_DIRECTORY + f"fz_cz_pz/dataset_128_{t}.0.pkl", #TUAB_DIRECTORY + "dataset_128_1.0.pkl",
                        },
                        "steps": {
                            "start_epoch": 10,
                            "end_epoch": 85,
                        }
                    }
                }
            else:
                train_config = {
                    "pretrain": None,
                    "train": {
                        "source":{
                            "file": INHOUSE_DIRECTORY + f"fz_cz_pz/dataset_128_{t}.0.pkl", #TUAB_DIRECTORY + "dataset_128_1.0.pkl",
                        },
                        "steps": {
                            "start_epoch": 0,
                            "end_epoch": 75,
                        }
                    },
                }
            
            cc = dc.upd({
                "model":{
                    "model_description": f"p_dropout {p_dropout}, nta, {'finetune, ' if is_pretrain else ''}{t} s, " + dc.config["model"]["model_description"], #!!CHECK
                    "encoder": {
                        "down_blocks_config": [
                            {"input_dim": 128*t, "seq_length": 3, "num_heads": t, "depth": 1, "mlp_depth": 2, "mlp_ratio": 1, "p_dropout": p_dropout, "act_layer": "PReLU", "norm_layer": "LayerNorm"},
                            {"input_dim": 64*t, "seq_length": 3, "num_heads": t, "depth": 1, "mlp_depth": 2, "mlp_ratio": 1, "p_dropout": p_dropout, "act_layer": "PReLU", "norm_layer": "LayerNorm"},
                            {"input_dim": 32*t, "seq_length": 3, "num_heads": t, "depth": 1, "mlp_depth": 2, "mlp_ratio": 1, "p_dropout": p_dropout, "act_layer": "PReLU", "norm_layer": "LayerNorm"}
                        ],
                        "out_block_config": {"dim": 16*t, "num_heads": t, "mlp_ratio": 1, "mlp_depth": 2, "p_dropout": p_dropout, "activation": "PReLU", "norm_layer": "LayerNorm"}
                    },
                    "decoder":{
                        "in_block_config": {"dim": 16*t, "num_heads": t, "mlp_ratio": 1, "mlp_depth": 2, "p_dropout": p_dropout, "activation": "PReLU", "norm_layer": "LayerNorm"},
                        "up_blocks_config": [
                            {"input_dim": 16*t, "seq_length": 3, "num_heads": t, "depth": 1, "mlp_depth": 2, "mlp_ratio": 1, "p_dropout": p_dropout, "act_layer": "PReLU", "norm_layer": "LayerNorm"},
                            {"input_dim": 32*t, "seq_length": 3, "num_heads": t, "depth": 1, "mlp_depth": 2, "mlp_ratio": 1, "p_dropout": p_dropout, "act_layer": "PReLU", "norm_layer": "LayerNorm"},
                            {"input_dim": 64*t, "seq_length": 3, "num_heads": t, "depth": 1, "mlp_depth": 2, "mlp_ratio": 1, "p_dropout": p_dropout, "act_layer": "PReLU", "norm_layer": "LayerNorm", "with_outro": True}
                        ]
                    }
                },
                "dataset": {
                    "train": train_config,
                    "val": {
                        "source":{
                            "file": INHOUSE_DIRECTORY + f"fz_cz_pz/dataset_128_{t}.0.pkl", #TUAB_DIRECTORY + "dataset_128_1.0.pkl",
                        },
                    },
                    "test": {
                        "source":{
                            "file": INHOUSE_DIRECTORY + f"fz_cz_pz/dataset_128_{t}.0.pkl", #TUAB_DIRECTORY + "dataset_128_1.0.pkl",
                        },
                    },
                }
            })
            experiments.append(cc)


# In[8]:


print("N experiments:", len(experiments))
for exp in experiments:
    print(exp['hash'], exp['model']['model_description'])


# # Training

# In[9]:


all_results = []
for config in experiments:
    exp_results = {
        config['model']["model_description"] : train(config, verbose=3)
    }
    all_results.append(exp_results)
    with open(os.path.join(config["log_path"], "all_results_" + config["model"]["model_description"].replace(" ", "_").replace("/", ".")), "w") as f:
        json.dump(exp_results, f, indent=4, ensure_ascii=False)
    with open("current_results.json", "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)