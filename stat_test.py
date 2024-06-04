'''
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

import warnings
warnings.simplefilter("ignore")

import os
import sys
import json
from IPython.display import display
import traceback

import scipy.stats as sts
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from matplotlib import rc
rc('animation', html='jshtml')

import torch
from torch.utils.data import DataLoader

import wandb
wandb.login(key='1b8e8dc9dcf1a34397a04197c4826d3fe7441dae')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

sys.path.append(SRC_FOLDER)

from src.utils import DEFAULT_SEED
from src.utils.common import seed_all, printLog, upd, read_json_with_comments
from src.data.data_reading import DataReader
from src.utils.plotting import printDatasetMeta, printDataloaderMeta, plotSamplesFromDataset
from src.data.dataset import InMemoryUnsupervisedDataset
from src.utils.parser import parse_ml_config
from src.trainer.evaluate_ml import evaluate_ml

from src.models import get_model, load_weights_from_wandb

def stattest(config, verbose=0):
    try:
        model1_description = config["models"]["model1"]["model_description"].replace(" ", "_").replace("/", ".")
        model2_description = config["models"]["model2"]["model_description"].replace(" ", "_").replace("/", ".")
        description = "stat_test_" + model1_description + "_vs_" + model2_description
        if config["log_path"] is not None: 
            logfile = open(os.path.join(config["log_path"], description), "a")
        else: 
            logfile = None

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
            pretrain_dataset = InMemoryUnsupervisedDataset(
                chunks_pretrain, **pretrain_config["preprocessing"]
            )
        train_dataset = InMemoryUnsupervisedDataset(
            chunks_train, **train_config["preprocessing"]
        )
        val_dataset = InMemoryUnsupervisedDataset(
            chunks_val, **val_config["preprocessing"]
        )
        test_dataset = InMemoryUnsupervisedDataset(
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
    
        #Models
        models = []
        for tag in ["model1", "model2"]:
            config["models"][tag].update({
                "input_dim" : train_dataset[0].shape,
            })
            model, config["models"][tag] = get_model(config["models"][tag])
            model = model.to(device)
            if verbose - 1 > 0: printLog('model ' + config["models"][tag]['model_description'] + ' is created', logfile=logfile)
            if verbose - 2 > 0: printLog(model)
        
            #Download weights
            model = load_weights_from_wandb(model, config["models"][tag]["artifact"], config["models"][tag]["file"], verbose=verbose)
        
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
                
            models.append(model)

        #print whole config
        printLog('#################### ' + description + ' ####################', logfile=logfile)
        printLog(json.dumps(config, indent=4), logfile=logfile)
        
        #parse ml config
        #should be just before training because replace names by objects
        config["ml"] = parse_ml_config(config["ml"])
    
        #seed
        seed_all(DEFAULT_SEED)
          
        ######
        # test
        ######
        results_all = {}
        for model, mode in zip(models, ["model1", "model2"]):
            if verbose > 0: printLog(f"##### Testing in {mode} mode... #####", logfile=logfile)
            results = evaluate_ml(
                model,
                device=device,
                test_dataset=test_dataset,
                targets_test=targets_test,
                verbose=verbose,
                **config["ml"],
            )
            results_all[mode] = results

        #statistical test
        test_results = []
        for metric in results_all["model1"]["d"]["test"]:
            estimates1 = results_all["model1"]["d"]["test"][metric]
            estimates2 = results_all["model2"]["d"]["test"][metric]
            statistic, pvalue = sts.ttest_ind(estimates1, estimates2)
            test_results.append({
                "metric": metric,
                "statistic": statistic,
                "pvalue": pvalue
            })
        if config.get("display_mode", "terminal") == "ipynb": display(pd.DataFrame.from_records(test_results))
        else: print(pd.DataFrame.from_records(test_results).T)
        print(json.dumps(test_results, indent=4), file=logfile)
        
        logfile.close()
        return test_results
        
    except Exception as error:
        # handle the exception
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback) 
        if logfile is not None: 
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=logfile) 
            logfile.close()
        return {}

train_config = read_json_with_comments("stat_test_configs/train_config.json")
logger_config = read_json_with_comments("stat_test_configs/logger_config.json")
dataset_config = read_json_with_comments(
    "stat_test_configs/dataset_config.json",
    {
        "{TUAB_DIRECTORY}": TUAB_DIRECTORY,
        "{INHOUSE_DIRECTORY}": INHOUSE_DIRECTORY,
        "{DEPR_ANON_DIRECTORY}": DEPR_ANON_DIRECTORY,
    }
)
ml_config = read_json_with_comments(
    "stat_test_configs/ml_config.json",
    {
        "{SEED}": str(DEFAULT_SEED)
    }
)

model_config = {
    "model1": {
        "model": "VAE_deep",
        "model_description": "finetune, duration, 60 s, beta-VAE, 3 ch., 4/8/16/32, 7/7/5/3/3/3/3/1, Sigmoid",
        "loss_reduction" : "mean",
        "latent_dim": 16*32,
        "beta": 2,
        "first_decoder_conv_depth": 32,
        "artifact" : "dmitriykornilov_team/EEG_depression_classification-PR-AUC/VAE_deep:v48",
        "file": "85_epoch.pth"
    },
    "model2": {
        "model": "VAE_deep",
        "model_description": "duration, inhouse_dataset, 60 s, beta-VAE, 3 ch., 4/8/16/32, 7/7/5/3/3/3/3/1, Sigmoid",
        "loss_reduction" : "mean",
        "latent_dim": 16*32,
        "beta": 2,
        "first_decoder_conv_depth": 32,
        "artifact" : "dmitriykornilov_team/EEG_depression_classification-PR-AUC/VAE_deep:v24",
        "file": "50_epoch.pth"
    }
}

default_config = {
    "project_name": 'EEG_depression_classification-PR-AUC',
    "method": "direct restoration",
    "save_path" : OUTPUT_FOLDER + 'model_weights/',
    "log_path" : OUTPUT_FOLDER + "logs/",
    "hash": "0",
    "display_mode": "terminal", #ipynb/terminal
    
    "dataset": dataset_config,
    "models": model_config,
    "train": train_config,
    "ml": ml_config,
    "logger": logger_config,
}

with open("stat_test_configs/default_config.json", "w") as f: json.dump(default_config, f, indent=4, ensure_ascii=False)

experiments = [default_config]
all_results = []
for config in experiments:
    model1_description = config["models"]["model1"]["model_description"].replace(" ", "_").replace("/", ".")
    model2_description = config["models"]["model2"]["model_description"].replace(" ", "_").replace("/", ".")
    description = "stat_test_" + model1_description + "_vs_" + model2_description
    
    exp_results = {
        description : stattest(config, verbose=3)
    }
    all_results.append(exp_results)
    with open(os.path.join(config["log_path"], "stat_test_all_results_" + description), "w") as f:
        json.dump(exp_results, f, indent=4, ensure_ascii=False)
    with open("stat_test_current_results.json", "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)