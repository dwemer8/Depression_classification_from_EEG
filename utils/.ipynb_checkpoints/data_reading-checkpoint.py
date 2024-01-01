import os
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

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
    Function reads tabular data from one folder
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
    Function reads tabular data from multiple folders
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