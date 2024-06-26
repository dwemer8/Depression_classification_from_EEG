'''
common libraries
'''

import os
import sys
import json
from copy import deepcopy
from tqdm.auto import tqdm as tqdm_auto

from IPython.display import display
import traceback

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from matplotlib import rc
rc('animation', html='jshtml')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

'''
Project libraries
'''
from src.utils.common import seed_all, printLog, upd, wrap_field, replace_unsupported_path_symbols
from src.data.data_reading import DataReader
from src.utils.plotting import dict_to_df, printDatasetMeta, printDataloaderMeta, plotSamplesFromDataset
from src.data.dataset import InMemoryUnsupervisedDataset, InMemorySupervisedDataset
from src.utils.logger import Logger
from src.utils.parser import parse_ml_config, parse_dataset_preprocessing_config
from src.trainer.early_stopper import EarlyStopper

from src.models import get_model, load_weights_from_wandb, load_sklearn_model_from_wandb

from src.trainer.train_eval import train_eval

'''
Experiment function
'''

def do_experiment(config, device="cpu", verbose=0):
    try:
        if config["log_path"] is not None: 
            logdir = os.path.join(config["log_path"], config["hash"], replace_unsupported_path_symbols(config["run_name"]), config["run_hash"])
            if not os.path.exists(logdir): os.makedirs(logdir)
            logfile = open(os.path.join(logdir, "log.txt"), "a")
        else: 
            logdir = None
            logfile = None

        #############
        #Data reading
        #############
        if verbose - 1 > 0: printLog("Data reading", logfile=logfile)

        #datasets
        #only two setups still supported: pretrain!=train=val=test and pretrain!=train!=val=test
        #TODO: add more setups
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
                train_size=pretrain_config["size"], val_size=0, test_size=0, seed=config["seed"]
            )

        if pretrain_config is None and train_config is None and val_config["source"] == test_config["source"]:
            val_test_reader = DataReader(
                test_config["source"]["file"],
                dataset_type=test_config["source"]["name"],
                verbose=(verbose-1),
            )
            _, val_set, test_set = val_test_reader.split(
                train_size=0, val_size=val_config["size"], test_size=test_config["size"], seed=config["seed"]
            )

        elif wrap_field(pretrain_config, 'source') != train_config["source"] == val_config["source"] == test_config["source"]:
            reader = DataReader(
                train_config["source"]["file"], 
                dataset_type=train_config["source"]["name"],
                verbose=(verbose-1)
            )
            train_set, val_set, test_set = reader.split(
                train_size=train_config["size"], val_size=val_config["size"], test_size=test_config["size"], seed=config["seed"]
            )    
            
        elif wrap_field(pretrain_config, 'source') != train_config["source"] != val_config["source"] == test_config["source"] != wrap_field(pretrain_config, 'source'):
            train_reader = DataReader(
                train_config["source"]["file"],
                dataset_type=train_config["source"]["name"],
                verbose=(verbose-1),
            )
            train_set, _, _ = train_reader.split(
                train_size=train_config["size"], val_size=0, test_size=0, seed=config["seed"]
            )

            val_test_reader = DataReader(
                test_config["source"]["file"],
                dataset_type=test_config["source"]["name"],
                verbose=(verbose-1),
            )
            _, val_set, test_set = val_test_reader.split(
                train_size=0, val_size=val_config["size"], test_size=test_config["size"], seed=config["seed"]
            )
            
        else:
            raise NotImplementedError("Unsupported datasets configuration")

        model_type = config["model"].get("type", "unsupervised")
        if model_type == "unsupervised":
            if pretrain_config is not None: chunks_pretrain, targets_pretrain = pretrain_set["chunk"], pretrain_set["target"]
            if train_config is not None: chunks_train, targets_train = train_set["chunk"], train_set["target"]
            chunks_val, chunks_test = val_set["chunk"], test_set["chunk"]
            targets_val, targets_test = val_set["target"], test_set["target"]

            #TODO: add to upd function ability to add new fields
            if pretrain_config is not None: 
                config["dataset"] = upd(config["dataset"], {
                    "train": {
                        "pretrain": {"n_samples": len(chunks_pretrain)}
                    }
                })
            if train_config is not None:
                config["dataset"] = upd(config["dataset"], {
                    "train": {
                        "train": {"n_samples": len(chunks_train)}
                    }
                })
                
            config["dataset"] = upd(config["dataset"], {
                "samples_shape": chunks_test[0].shape,
                "val": {"n_samples": len(chunks_val)},
                "test": {"n_samples": len(chunks_test)},
            })

            if pretrain_config is not None: 
                pretrain_dataset = InMemoryUnsupervisedDataset(
                    chunks_pretrain, **parse_dataset_preprocessing_config(pretrain_config["preprocessing"])
                )
            if train_config is not None: 
                train_dataset = InMemoryUnsupervisedDataset(
                    chunks_train, **parse_dataset_preprocessing_config(train_config["preprocessing"])
                )
            val_dataset = InMemoryUnsupervisedDataset(
                chunks_val, **parse_dataset_preprocessing_config(val_config["preprocessing"])
            )
            test_dataset = InMemoryUnsupervisedDataset(
                chunks_test, **parse_dataset_preprocessing_config(test_config["preprocessing"])
            )
            
        elif model_type == "supervised":
            #TODO: add to upd function ability to add new fields
            if pretrain_config is not None: 
                config["dataset"] = upd(config["dataset"], {
                    "train": {
                        "pretrain": {"n_samples": len(pretrain_set['chunk'])}
                    }
                })

            if train_config is not None:
                config["dataset"] = upd(config["dataset"], {
                    "train": {
                        "train": {"n_samples": len(train_set['chunk'])}
                    }
                })
                
            config["dataset"] = upd(config["dataset"], {
                "samples_shape": test_set['chunk'][0].shape,
                "val": {"n_samples": len(val_set['chunk'])},
                "test": {"n_samples": len(test_set['chunk'])},
            })

            if pretrain_config is not None: 
                pretrain_dataset = InMemorySupervisedDataset(
                    list(zip(pretrain_set['chunk'], pretrain_set["target"])) , **parse_dataset_preprocessing_config(pretrain_config["preprocessing"])
                )
            if train_config is not None: 
                train_dataset = InMemorySupervisedDataset(
                    list(zip(train_set['chunk'], train_set["target"])), **parse_dataset_preprocessing_config(train_config["preprocessing"])
                )
            val_dataset = InMemorySupervisedDataset(
                list(zip(val_set['chunk'], val_set["target"])), **parse_dataset_preprocessing_config(val_config["preprocessing"])
            )
            test_dataset = InMemorySupervisedDataset(
                list(zip(test_set['chunk'], test_set["target"])), **parse_dataset_preprocessing_config(test_config["preprocessing"])
            )
            
        else:
            raise ValueError("Unsupported training type")
        
    
        if verbose - 2 > 0 and model_type == "unsupervised": 
            printDatasetMeta(val_dataset, test_dataset, train_dataset=None if train_config is None else train_dataset, pretrain_dataset=None if pretrain_config is None else pretrain_dataset, logfile=logfile)
            if config.get("display_mode", "terminal") == "ipynb": plotSamplesFromDataset(test_dataset)
    
        #Dataloader
        if pretrain_config is not None: pretrain_dataloader = DataLoader(pretrain_dataset, shuffle=True, **config["dataset"]['dataloader'])
        if train_config is not None: train_dataloader = DataLoader(train_dataset, shuffle=True, **config["dataset"]['dataloader'])
        val_dataloader = DataLoader(val_dataset, shuffle=False, **config["dataset"]['dataloader'])
        test_dataloader = DataLoader(test_dataset, shuffle=False, **config["dataset"]['dataloader'])
    
        if verbose - 2 > 0: printDataloaderMeta(val_dataloader, test_dataloader, train_dataloader=None if train_config is None else train_dataloader, pretrain_dataloader=None if pretrain_config is None else pretrain_dataloader, logfile=logfile)
    
        #Model and environment
        config["model"].update({
            "input_dim" : test_dataset[0].shape if model_type == "unsupervised" else test_dataset[0][0].shape,
        })
        model, config["model"] = get_model(config["model"])
        model = model.to(device)
        if verbose - 1 > 0: printLog('model ' + config["model"].get('model_description', config["model"]["model"])  + ' is created', logfile=logfile)
        if verbose - 2 > 0: printLog(model, logfile=logfile)

        #NB: Should be placed before Logger since it uses wandb.init() and wandb.finish()
        #Download weights
        if "artifact" in config["model"] and "file" in config["model"]:
            model = load_weights_from_wandb(model, config["model"]["artifact"], config["model"]["file"], verbose=verbose, device=device)
        
        ml_model = None #will be used further to update config
        if "ml_artifact" in config["model"] and "ml_file" in config["model"]:
            ml_model = load_sklearn_model_from_wandb(config["model"]["ml_artifact"], config["model"]["ml_file"], verbose=verbose)

        # TESTS
        model.eval()
        test_data_point = test_dataset[0][None].to(device) if model_type == "unsupervised" else test_dataset[0][0][None].to(device)
        inference_result = model(test_data_point)
        if model_type == "unsupervised":
            reconstruct_result = model.reconstruct(test_data_point)
            encode_result = model.encode(test_data_point)
        if verbose - 1 > 0: 
            printLog(f"Test data point shape: {test_data_point.shape}", logfile=logfile)
            printLog(f"Test inference result length: {len(inference_result)}", logfile=logfile)
            if model_type == "unsupervised":
                printLog(f"Test reconstruct shape: {reconstruct_result.shape}", logfile=logfile)
                printLog(f"Test encode shape: {encode_result.shape}", logfile=logfile)
    
        #optimizer and scheduler
        optimizer = getattr(torch.optim, config["optimizer"]["optimizer"])(model.parameters(), **config["optimizer"]["kwargs"])
        if verbose - 1 > 0: printLog(f'Optimizer {type(optimizer).__name__} is instantiated', logfile=logfile)
    
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["scheduler"])(optimizer, **config["scheduler"]["kwargs"])
        if verbose - 1 > 0: printLog(f'Scheduler {type(scheduler).__name__} is instantiated', logfile=logfile)

        #loss
        pretrain_criterion = None
        if model_type == "supervised" and pretrain_config is not None:
            alpha = np.sum(pretrain_set["target"]) / len(pretrain_set["target"]) #0 or 1
            weight = torch.tensor([alpha, 1 - alpha]).to(torch.float32).to(device)
            pretrain_criterion = nn.CrossEntropyLoss(weight=weight)

        train_criterion = None
        if model_type == "supervised" and train_config is not None:
            alpha = np.sum(train_set["target"]) / len(train_set["target"]) #0 or 1
            weight = torch.tensor([alpha, 1 - alpha]).to(torch.float32).to(device)
            train_criterion = nn.CrossEntropyLoss(weight=weight)
    
        logger = Logger(
            log_type=config["logger"]["log_type"], 
            run_name=config["run_name"],
            save_path=config["save_path"],
            model=model,
            model_name=config["model"]["model"],        
            project_name=config["project_name"],
            config=config,
            model_description=config["model"]["model_description"],
        #         log_dir = OUTPUT_FOLDER + "logs/"
        )

        #NB:Cannot be placed before get_model() since it updates config
        #print whole config
        printLog('#################### ' + config["run_name"] + ' ####################', logfile=logfile)
        printLog(json.dumps(config, indent=4), logfile=logfile)
        
        #NB:should be just before training because replace names by objects. Cannot be placed before Logger since it logs config
        #parse ml config
        config["ml"] = parse_ml_config(config["ml"])
        config["ml_validation"] = parse_ml_config(config["ml_validation"])
        if ml_model is not None:
            config["ml"]["ml_model"] = deepcopy(ml_model)
            config["ml_validation"]["ml_model"] = deepcopy(ml_model)
    
        #seed
        seed_all(config["seed"])
    
        #training
        best_loss = np.inf
        best_clf_accuracy = 0
        best_model = None
        best_ml_model = None
        best_epoch = None
        final_epoch = None
        final_model = None
        final_ml_model = None
        last_model = None
        last_ml_model = None
        zero_ml_tag = config["ml_validation"]["ml_eval_function_tag"][0]

        for curr_dataloader, dataset_config, criterion in zip(
            [None if pretrain_config is None else pretrain_dataloader, None if train_config is None else train_dataloader],
            [pretrain_config, train_config], 
            [pretrain_criterion, train_criterion] 
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
                model, trained_ml_models, results = train_eval( #None in trained_ml_model since no test_dataset and targets_test
                    curr_dataloader,
                    model,
                    device=device,
                    mode="train",
                    optimizer=optimizer,
                    epoch=epoch,
                    logger=logger,
                    loss_coefs=config["train"]["loss_coefs"],
                    loss_reduction=config["model"]["loss_reduction"] if model_type == "unsupervised" else None,
                    is_mask=(config["train"]["masking"]["n_masks"] != 0 and config["train"]["masking"]["mask_ratio"] != 0),
                    mask_ratio=config["train"]["masking"]["mask_ratio"],
                    step_max=dataset_config["steps"]["step_max"], 
                    verbose=verbose,
                    logfile=logfile,
                    model_type=model_type,
                    criterion=criterion,
                )
                if results == {}: break
                if verbose > 0: 
                    #printLog doesn't used in order to display nice table
                    if config.get("display_mode", "terminal") == "ipynb": display(dict_to_df(results))
                    else: print(dict_to_df(results).T)
                    for k in results: 
                        if isinstance(results[k], np.ndarray): results[k] = float(results[k].tolist())
                    print(json.dumps(results, indent=4), file=logfile)
        
                ############
                # validation
                ############
                if verbose > 0: printLog("##### Validation... #####", logfile=logfile)
                model, trained_ml_models, results = train_eval(
                    val_dataloader,
                    model,
                    device=device,
                    mode="validation",
                    test_dataset=val_dataset if model_type == "unsupervised" else val_set["chunk"],
                    targets_test=targets_val if model_type == "unsupervised" else val_set["target"],
                    check_period=config["train"]["validation"]["check_period"] if epoch % config['train']['validation']['check_period_per_epoch'] == 0 else None,
                    plot_period=config["train"]["validation"]["plot_period"] if epoch % config['train']['validation']['plot_period_per_epoch'] == 0 else None,
                    epoch=epoch,
                    logger=logger,
                    loss_coefs=config["train"]["loss_coefs"],
                    loss_reduction=config["model"]["loss_reduction"] if model_type == "unsupervised" else None,
                    step_max=dataset_config["steps"]["step_max"], 
                    verbose=verbose,
                    logfile=logfile,
                    logdir=logdir,
                    model_type=model_type,
                    criterion=criterion,
                    **config["ml_validation"],
                )
                if results == {}: break
                if verbose > 0: 
                    if config.get("display_mode", "terminal") == "ipynb": display(dict_to_df(results))
                    else: print(dict_to_df(results).T)
                    for k in results: 
                        if type(results[k]) == np.ndarray: results[k] = float(results[k].tolist())
                    print(json.dumps(results, indent=4), file=logfile)
        
                scheduler.step(results['loss'])

                if results['loss'] < best_loss and dataset_config == train_config: #validation loss from results is here 
                    best_loss = results['loss']
                    final_model = deepcopy(model)
                    final_ml_model = deepcopy(trained_ml_models[zero_ml_tag]) if trained_ml_models is not None else None
                    final_epoch = epoch
                    if verbose > 0: printLog(f"New best loss = {best_loss} on epoch {epoch}", logfile=logfile)

                last_tag = "cv" if zero_ml_tag == "cv" else "bs"
                accuracy_tag = f'clf.{zero_ml_tag}.test.{last_tag}.balanced_accuracy'
                if epoch % config['train']['validation']['check_period_per_epoch'] == 0 and dataset_config == train_config:
                    if results.get(accuracy_tag, 0) > best_clf_accuracy: #metric is computed and classifier is learnt only every n epochs
                        best_clf_accuracy = results[accuracy_tag]
                        best_model = deepcopy(model)
                        best_ml_model = deepcopy(trained_ml_models[zero_ml_tag]) if trained_ml_models is not None else None
                        best_epoch = epoch
                        if verbose > 0: printLog(f"New best classifier accuracy = {best_clf_accuracy} on epoch {epoch}", logfile=logfile)
                
                if early_stopper.early_stop(results['loss']): break #validation loss is here

            logger.save_model(
                dataset_config["steps"]['end_epoch'] - 1, 
                model=model, 
                model_postfix="_pretrain" if dataset_config == pretrain_config else "_train", #dataset_config isn't None since condition at the beginning
                ml_model=trained_ml_models[zero_ml_tag] if trained_ml_models is not None else None, 
                ml_model_postfix="_pretrain" if dataset_config == pretrain_config else "_train" #trained_ml_models isn't needed to be checked for None, see method implementation
            )

        if train_config is not None or pretrain_config is not None:
            if final_epoch is not None:
                logger.save_model(final_epoch, model=final_model, ml_model=final_ml_model, model_postfix="_final", ml_model_postfix="_final")
                logger.update_summary("validation.final_epoch", final_epoch)

            if best_epoch is not None:
                logger.save_model(best_epoch, model=best_model, ml_model=best_ml_model, model_postfix="_test", ml_model_postfix="_test")
                logger.update_summary("validation.best_epoch", best_epoch)
        else:
            print("INFO: Since there is no pretrain or train, final_model is loaded from current model, final_ml_model is loaded from current ml_model (and can be None) and test_model with test_ml_model are Nones.")
            final_model = model
            final_ml_model = ml_model
            final_epoch = 0
    
        ######
        # test
        ######
        results_all = {}
        not_trained_ml_model = deepcopy(config["ml"]["ml_model"])

        for tested_model, tested_ml_model, mode, epoch in zip(
            [final_model, best_model], [final_ml_model, best_ml_model], ["final", "test"], [final_epoch, best_epoch]
        ):
            if verbose > 0: printLog(f"##### Testing in {mode} mode... #####", logfile=logfile)
            if tested_model is not None:
                if tested_ml_model is not None and config["ml"]["ml_to_train"] == False:
                    printLog(f"INFO:{mode}_ml_model present: {tested_ml_model}", logfile=logfile)
                    config["ml"]["ml_model"] = tested_ml_model
                    config["ml"]["ml_to_train"] = False

                elif tested_ml_model is not None and config["ml"]["ml_to_train"] == True:
                    printLog(f"WARNING:{mode}_ml_model present, but ml_to_train==True, new {mode}_ml_model will be trained on test dataset!", logfile=logfile)
                    config["ml"]["ml_model"] = not_trained_ml_model #is needed in case if we changed model on previous step, but on this we have None
                    config["ml"]["ml_to_train"] = True

                else:
                    printLog(f"INFO:{mode}_ml_model is None, raw ml model {not_trained_ml_model} is loaded from config and will be trained on test dataset!", logfile=logfile)
                    config["ml"]["ml_model"] = not_trained_ml_model #is needed in case if we changed model on previous step, but on this we have None
                    config["ml"]["ml_to_train"] = True

                tested_model, trained_ml_models, results = train_eval(
                    test_dataloader,
                    tested_model,
                    device=device,
                    mode=mode,
                    test_dataset=test_dataset if model_type == "unsupervised" else test_set["chunk"],
                    targets_test=targets_test if model_type == "unsupervised" else test_set["target"],
                    check_period=1e10,
                    plot_period=1e10,
                    epoch=train_config["steps"]['end_epoch'] if train_config is not None else 0,
                    logger=logger,
                    loss_coefs=config["train"]["loss_coefs"],
                    loss_reduction=config["model"]["loss_reduction"] if model_type == "unsupervised" else None,
                    step_max=train_config["steps"]["step_max"] if train_config is not None else None, 
                    verbose=verbose,
                    logfile=logfile,
                    logdir=logdir,
                    model_type=model_type,
                    criterion=train_criterion,
                    **config["ml"],
                )
                results_all[mode] = results
                if results == {}: break
                if verbose > 0: 
                    if config.get("display_mode", "terminal") == "ipynb": display(dict_to_df(results))
                    else: print(dict_to_df(results).T)
                    for k in results: 
                        if type(results[k]) == np.ndarray: results[k] = float(results[k].tolist())
                    print(json.dumps(results, indent=4), file=logfile)

                if config.get("save_after_test", False):
                    printLog(f"INFO: Saving models after test in {mode} mode.")
                    is_retrained = config["ml"]["ml_to_train"]
                    logger.save_model(
                        epoch, 
                        model=tested_model, 
                        model_postfix=f"_{mode}_", 
                        ml_model=trained_ml_models[zero_ml_tag] if trained_ml_models is not None else None, 
                        ml_model_postfix=f"_{mode}_{'' if is_retrained else 'not_'}retrained_" #run_hash will be added after postfix 
                    )
            else:
                printLog(f"WARNING:{mode} model is None", logfile=logfile)
        
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