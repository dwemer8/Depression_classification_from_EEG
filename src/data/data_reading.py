import os
from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from src.utils import DEFAULT_SEED

##########################################################################
#files
##########################################################################

def fileExtension(fname):
    return fname.split('.')[-1]

def fileName(fname):
    last_point_idx = len(fname) - 1 - fname[::-1].find(".")
    return fname[:last_point_idx]

def readDataExt_one(folder, ext="csv", exclude={}, add_filename=True, n_first=None, verbose=0, is_list=False, file_preprocess=None, file_postprocess=None):
    '''
    Function reads tabular data from one folder into one table or list of tables
    '''
    data = []
    file_names = os.listdir(folder)
    if n_first != None: file_names = file_names[:n_first+1]
        
    for file_name in tqdm(file_names):  
        if fileExtension(file_name) == ext and not(file_name in exclude):
            if verbose > 0: print(file_name)
            if file_preprocess != None: file_preprocess(folder + file_name) 
            new_part = pd.read_csv(os.path.join(folder, file_name))
            if file_postprocess != None: file_postprocess(folder + file_name) 
            if add_filename: new_part = pd.concat([new_part, pd.Series([file_name]*len(new_part), name="file_name")], axis=1) 
            data.append(deepcopy(new_part)) 
    
    if is_list: return data
    else: return pd.concat([*data], axis=0, ignore_index=True)

def readDataExt_mul(folders, verbose=0, **kwargs):
    '''
    Function reads tabular data from multiple folders into list of tables or list of lists of tables
    '''
    dfs = []
    for folder in folders:
        dfs.append(readDataExt_one(
                folder, 
                verbose=max(verbose - 1, 0),
                **kwargs
            )
        )
        if verbose: print(f"{folder} folder was processed")
    
    return dfs

class DataReader:
    '''
    Class for reading and splitting data from pickle file
    '''
    def __init__(self, file, verbose=0, dataset_type="depression_anonymized"):
        if dataset_type not in ["inhouse_dataset", "depression_anonymized", "TUAB"]:
            raise NotImplementedError(f"Unkown dataset type {dataset_type}")
        
        self.dataset_type = dataset_type
        self.verbose = verbose
        with open(file, "rb") as f:
            self.chunks_list = pickle.load(f)
        
        if self.verbose > 0:
            if self.dataset_type in ["inhouse_dataset", "depression_anonymized"]: 
                print("\nChunks shape:", self.chunks_list[0]["chunk"].shape, ", length:", len(self.chunks_list), ", keys:", self.chunks_list[0].keys())
            elif self.dataset_type == "TUAB":
                print("\nChunks shape:", self.chunks_list["chunks_train"][0].shape, ", length:", len(self.chunks_list["chunks_train"]), ", keys:", self.chunks_list.keys())

    def split(self, train_size=None, val_size=0.1, test_size=0.1, seed=DEFAULT_SEED):
        if self.dataset_type in ["depression_anonymized", "inhouse_dataset"]:
            '''
            We need to divide data in such way that records from the same patient should be only in train or test or validation
            '''
            #reading patients tags and associated targets
            patients_targets = pd.DataFrame.from_records([{"patient": x["patient"], "target": x["target"]} for x in self.chunks_list]).drop_duplicates().reset_index(drop=True)
            if self.verbose > 0: print(f"N patients = {len(patients_targets)}")

            #split to train, val and test
            def empty_patients_df():
                return pd.DataFrame({"patient": [], "target": []})

            if (train_size is None or train_size != 0) and val_size == 0 and test_size == 0:
                if train_size is not None: patients_train = patients_targets[:train_size]
                else: patients_train = patients_targets
                patients_val = empty_patients_df()
                patients_test = empty_patients_df()
                
            elif train_size == 0 and val_size != 0 and test_size != 0:
                patients_train = empty_patients_df()
                patients_val, patients_test = train_test_split(patients_targets, test_size=(test_size/(test_size+val_size)), random_state=seed, stratify=patients_targets["target"], shuffle=True)
                
            elif (train_size is None or train_size != 0) and val_size != 0 and test_size != 0:
                patients_train, patients_val_test = train_test_split(patients_targets, test_size=(val_size + test_size), random_state=seed, stratify=patients_targets["target"], shuffle=True)
                patients_val, patients_test = train_test_split(patients_val_test, test_size=(test_size/(test_size+val_size)), random_state=seed, stratify=patients_val_test["target"], shuffle=True)
                if train_size is not None: patients_train = patients_train[:train_size]
            
            else:
                raise NotImplementedError(f"Unsupported combination of train_size={train_size}, val_size={val_size} and test_size={test_size}")
            

            #type conversion
            patients_train, patients_val, patients_test = patients_train["patient"].values, patients_val["patient"].values, patients_test["patient"].values
            if self.verbose > 0: print(f"Train={len(patients_train)}, validation={len(patients_val)}, test={len(patients_test)}")

            #go through dataset and distribution of data points to datasets
            reset = {"chunk": [], "target": [], "patient": []}
            train_set, val_set, test_set = deepcopy(reset), deepcopy(reset), deepcopy(reset)

            def add_chunk_to_dataset(chunk, verbose=1, prev_chunk_patient_id = ""):
                patient_id = chunk["patient"]
                was_added = False
                for data_set, patients in zip([train_set, val_set, test_set], [patients_train, patients_val, patients_test]):
                    if patient_id in patients:
                        for tag in ["chunk", "target", "patient"]: data_set[tag].append(chunk[tag])
                        was_added = True
                        break
                if verbose and not was_added and patient_id != prev_chunk_patient_id:
                    print (f"WARNING: Patient data with id {patient_id} wasn't added to any dataset")
                return patient_id

            if self.verbose > 0:
                prev_chunk_patient_id = ""
                for chunk in tqdm(self.chunks_list):
                    prev_chunk_patient_id = add_chunk_to_dataset(chunk, self.verbose, prev_chunk_patient_id=prev_chunk_patient_id)
            else:
                for chunk in self.chunks_list:
                    add_chunk_to_dataset(chunk, self.verbose)

            #type conversion
            for data_set in [train_set, val_set, test_set]:
                data_set["chunk"] = np.array(data_set["chunk"])
                data_set["target"] = np.array(data_set["target"])
            
            if self.verbose > 0: 
                if len(train_set["chunk"]) > 0: print("Train:", len(train_set["chunk"]), train_set["chunk"][0].shape)
                if len(val_set["chunk"]) > 0: print("Validation:", len(val_set["chunk"]), val_set["chunk"][0].shape)
                if len(test_set["chunk"]) > 0: print("Test:", len(test_set["chunk"]), test_set["chunk"][0].shape)
    
            return train_set, val_set, test_set

        elif self.dataset_type == "TUAB":
            print("WARNING: train_size, val_size and test_size were ignored")
            train_set = {"chunk" : self.chunks_list["chunks_train"], "target" : self.chunks_list["targets_train"]}
            val_set = {"chunk" : self.chunks_list["chunks_val"], "target" : self.chunks_list["targets_val"]}
            test_set = {"chunk" : self.chunks_list["chunks_test"], "target" : self.chunks_list["targets_test"]}

            return train_set, val_set, test_set

        else:
            raise NotImplementedError(f"Unkown dataset type {self.dataset_type}")