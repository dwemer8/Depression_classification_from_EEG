import os
import numpy as np
import pandas as pd
import mne
from tqdm.auto import tqdm
import pickle

from .data_reading import readDataExt_one, readDataExt_mul, fileExtension, fileName
from utils import DEFAULT_SEED

SAMPLING_FREQUENCY = 250 #Hz
L_FREQ = 0.1 #Hz
H_FREQ = 30 #Hz
AMPLITUDE_THRESHOLD = 0.005 #V
STD_THRESHOLD = 5
TRANSIENT_STD_THRESHOLD = 3

def save_preprocessed_data(chunks_list, file):
    with open(file, "wb") as f:
        pickle.dump(chunks_list, f)

def pickChunks(df, ch_names, n_samples_per_chunk=128, n_chunks_max=None):
    chunks = []
    start_idx = 0
    end_idx = start_idx + n_samples_per_chunk

    while end_idx <= df.shape[0]:
        if n_chunks_max != None and len(chunks) >= n_chunks_max:
            break
        
        chunk = df.iloc[start_idx:end_idx]
        if len(chunk) != n_samples_per_chunk:
            print(f"WARNING: chunk shape = {chunk.shape}")
            start_idx = end_idx
            end_idx += n_samples_per_chunk
            continue

        #std = 5 threshold
        if chunk[ch_names].to_numpy().std() >= 5:
            start_idx = end_idx
            end_idx += n_samples_per_chunk
            continue

        # std t, t+1 threshold
        # drop = False
        # for col in chunk[ch_names]:
        #     timeseries = chunk[col].to_numpy()
        #     for i in range(len(timeseries) - 1):
        #         if np.std([timeseries[i], timeseries[i + 1]]) >= 3:
        #             drop = True
        #             break
        #     if drop:
        #         break
        # if drop:
        #     continue

        chunks.append(chunk)
        start_idx = end_idx
        end_idx += n_samples_per_chunk

    return chunks

def processPatientData(
    df,
    l_freq = L_FREQ,
    h_freq = H_FREQ,
    source_freq = 125,
    target_freq = 125,
    divider = 1e6,
    amplitude_threshold = AMPLITUDE_THRESHOLD,
    **kwargs
):
    '''
    This function:
        - make mne Raw from df, 
        - apply average reference
        - filter
        - resample
        - convert to df
        - clip
        - normalize
        - divide df to chunks
    It returns array of chunks in pd.DataFrame format. Perform preprocessing over all channels.
    '''
    
    ch_names = df.columns.to_list()[1:] #delete 'time' #['t6', 't4', 'o1', 'f8', 'p4', 'c4', 't3', 'f7', 'f3', 'o2', 'f4', 'c3', 'p3', 't5', 'cz', 'fp1', 'fp2', 'pz', 'fz']
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=source_freq)
    raw = mne.io.RawArray(df[ch_names].to_numpy(copy=True).T / divider, info, verbose=False) #data in microvolts

    #average reference, filtration
    raw, _ = mne.set_eeg_reference(raw, ref_channels='average', verbose=False) #average reference
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', verbose=False) #filtration
    raw = raw.resample(target_freq, npad='auto')
    
    df = raw.to_data_frame() #in microvolts
    #clipping
    df.loc[:, ch_names].clip(-amplitude_threshold, amplitude_threshold, inplace=True)
    #normalization
    df.loc[:, ch_names] = (df[ch_names] - df[ch_names].mean())/df[ch_names].std()

    return pickChunks(df, ch_names, **kwargs)

### Depression Anonymized ###

def preprocessDepressionAnonymizedData(
    directory, 
    epoch_folders, 
    picked_channels, 
    **kwargs
):
    chunks_list = []
    
    for epoch_folder in epoch_folders:
        print(epoch_folder)
        data_epoch_descr = pd.read_csv(directory + epoch_folder + "path_file.csv")
        data_epoch_list = readDataExt_one(directory + epoch_folder, exclude={"path_file.csv"}, is_list=True)
    
        for df in tqdm(data_epoch_list):
            # if data_epoch_descr[data_epoch_descr["fn"] == df['file_name'].iloc[0]].iloc[0]['target'] == 1: #take only healthy
            #     continue
    
            # print(df['file_name'][0])
            chunks_from_patient = processPatientData(df.drop(['file_name'], axis=1), **kwargs)
            file_mask = data_epoch_descr["fn"] == df['file_name'].iloc[0]
            target = data_epoch_descr[file_mask].iloc[0]['target']
    
            for chunk in chunks_from_patient:
                image = chunk[picked_channels].to_numpy().T
                chunks_list.append({
                    "chunk": image,
                    "target": target,
                    "patient": df['file_name'].iloc[0]
                })
    
    print("\nChunks shape:", chunks_list[0]["chunk"].shape, ", chunks number:", len(chunks_list))
    return chunks_list

### Inhouse Dataset ###

def preprocessInhouseDatasetData(
    directory,
    data_folders,
    picked_channels,
    **kwargs
):

    #read path_file.csv with targets and patients tags
    data_epoch_descr = pd.read_csv(os.path.join(directory, "path_file.csv")).drop(["Unnamed: 0"], axis="columns")
    data_epoch_descr["fn"] = data_epoch_descr["fn"].map(lambda x: x.split("/")[-1]) #remove dir from filenames

    #read data from MMD and Healthy dirs
    dirs = [os.path.join(directory, x) for x in data_folders] 
    data_epoch_list = readDataExt_mul(dirs, is_list=True)

    #flat data list [N_dirs x N_dfs] -> [N_dirs*N_dfs]
    data_epoch_list_ = []
    for df_list in data_epoch_list: data_epoch_list_.extend(df_list)
    data_epoch_list = data_epoch_list_

    #divide dfs to chunks
    chunks_list = []
    for df in tqdm(data_epoch_list):
        chunks_from_patient = processPatientData(df.drop(['file_name', "Unnamed: 0"], axis=1), **kwargs)
        file_mask = data_epoch_descr["fn"] == df['file_name'].iloc[0]
        target = data_epoch_descr[file_mask].iloc[0]['target']
    
        for chunk in chunks_from_patient:
            image = chunk[picked_channels].to_numpy().T
            chunks_list.append({
                "chunk": image,
                "target": target,
                "patient": df['file_name'].iloc[0]
            })
    
    print("\nChunks shape:", chunks_list[0]["chunk"].shape, "length:", len(chunks_list))
    return chunks_list

### TUAB ###

def getAge(file_name):
    f = open(file_name, "r", encoding="utf-8")
    try:
        buffer = "1234"
        while buffer != "Age:":
            buffer = buffer[1:] + f.read(1)

        age = f.read(2)
        f.close()
        return int(age)

    except (UnicodeDecodeError, KeyboardInterrupt, ValueError) as error:
        f.close()
        return "error"

def preprocessRecord(
        file_name,
        verbose=False,
        channels_to_drop=['EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR'],
        target_freq=125,
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        ampl_thresh=AMPLITUDE_THRESHOLD,
        **kwargs
):
    #average reference, filtration
    raw = mne.io.read_raw_edf(file_name, preload=True, verbose=verbose) #data in microvolts
    raw = raw.drop_channels(channels_to_drop, on_missing="warn")
    
    raw, _ = mne.set_eeg_reference(raw, ref_channels='average', verbose=verbose) #average reference
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', verbose=verbose) #filtration
    raw = raw.resample(target_freq, npad='auto')
    
    df = raw.to_data_frame() #in microvolts
    ch_names = raw.ch_names
    #clipping
    df.loc[:, ch_names].clip(-ampl_thresh, ampl_thresh, inplace=True)
    #normalization
    df.loc[:, ch_names] = (df[ch_names] - df[ch_names].mean())/df[ch_names].std()

    return pickChunks(df, ch_names, **kwargs) #time is also among columns

def processTUABdataDirectory(
    directory,
    picked_channels,
    n_files=None,
    file_type = "edf",
    chunks_file_suffix = "_chunks_fz_cz_pz_3x128",
    targets_file_suffix = "_targets",
    is_save = False,
    force_recompute = False,
    **kwargs
):
    n_files_read = 0
    n_files_passed = 0
    
    if n_files is not None:
        file_names = os.listdir(directory)[:n_files]
    else:
        file_names = os.listdir(directory)
    
    bunch_of_chunks_list = []
    bunch_of_targets_list = []
    
    for file_name in tqdm(file_names):
        if fileExtension(file_name) == file_type and \
            (force_recompute or \
             not (os.path.exists(directory + fileName(file_name) + chunks_file_suffix + ".npy") and \
                  os.path.exists(directory + fileName(file_name) + targets_file_suffix + ".npy")\
                 )\
            ):
            chunks_list = []
            targets_list = []
    
            #get age
            age = getAge(directory + file_name)
            if age == "error":
                n_files_passed += 1
                print(f"File {file_name} was passed, passed files: {n_files_passed}, read files: {n_files_read}")
                continue
            n_files_read += 1
    
            #get chunks from .edf
            chunks_from_record = preprocessRecord(directory + file_name, **kwargs)
    
            #append all chunks with targets to dataset
            for chunk in chunks_from_record:
                image = chunk[picked_channels].to_numpy().T
                chunks_list.append(image)
                targets_list.append(age)
            chunks = np.array(chunks_list)
            targets = np.array(targets_list)
            bunch_of_chunks_list.append(chunks)
            bunch_of_targets_list.append(targets)
    
            if is_save:
                np.save(directory + fileName(file_name) + chunks_file_suffix, chunks)
                np.save(directory + fileName(file_name) + targets_file_suffix, targets)
    
    print(f"Read files: {n_files_read}, passed files: {n_files_passed}")
    
    return np.concatenate(bunch_of_chunks_list), np.concatenate(bunch_of_targets_list)

def preprocessTUABdata(
    picked_channels, 
    chunks_file_name, 
    targets_file_name, 
    TUAB_TRAIN, 
    TUAB_EVAL, 
    force_recompute=True,
    **kwargs
):
    for directory in [TUAB_TRAIN, TUAB_EVAL]:
        chunks, targets = processTUABdataDirectory(
            directory,
            picked_channels,
            force_recompute=force_recompute,
            chunks_file_suffix="_"+chunks_file_name,
            targets_file_suffix="_"+targets_file_name,
            **kwargs
        )
        np.save(directory + chunks_file_name, chunks)
        np.save(directory + targets_file_name, targets)

def concatenateTUABpreprocessedFiles(
    TUAB_TRAIN,
    TUAB_EVAL,
    chunks_file_name,
    targets_file_name,
    n_samples_per_chunk=128,
    n_channels=3
):
    '''
    # Concatenate .npy files from each record together. Very helpful if preprocessing crashes anywhere
    '''
    for directory in [TUAB_TRAIN, TUAB_EVAL]:
        print(directory)
        chunks_list = []
        targets_list = []
        for file_name in [directory + chunks_file_name + ".npy", directory + targets_file_name + ".npy"]: #delete already concatenated chunks
            if os.path.exists(file_name):
                os.remove(file_name)
    
        for i, file_name in enumerate(tqdm(os.listdir(directory))): #iterate through files from every file
            if chunks_file_name in file_name or targets_file_name in file_name:
                data = np.load(directory + file_name)
                if targets_file_name in file_name:
                    targets_list.append(data)
                else:
                    chunks_list.append(data)
    
        chunks = np.array(chunks_list).reshape(-1, 1, n_channels, n_samples_per_chunk)
        targets = np.array(targets_list).reshape(-1)
        
        print("Chunks:", chunks.shape, "targets:", targets.shape)
        
        np.save(directory + "../" + chunks_file_name, chunks)
        np.save(directory + "../" + targets_file_name, targets)

from sklearn.model_selection import train_test_split

def getTUABdataset(
    TUAB_TRAIN, 
    TUAB_EVAL, 
    chunks_file_name, 
    targets_file_name,
    n_channels=3,
    n_samples_per_chunk=128,
    val_size=0.1,
    SEED=DEFAULT_SEED,
):
    chunks_train_val = np.load(TUAB_TRAIN + chunks_file_name + ".npy").reshape(-1, n_channels, n_samples_per_chunk)
    targets_train_val = np.load(TUAB_TRAIN + targets_file_name + ".npy")
    chunks_test = np.load(TUAB_EVAL + chunks_file_name + ".npy").reshape(-1, n_channels, n_samples_per_chunk)
    targets_test = np.load(TUAB_EVAL + targets_file_name + ".npy")
    
    chunks_train, chunks_val, targets_train, targets_val = train_test_split(chunks_train_val, targets_train_val, test_size=val_size, random_state=DEFAULT_SEED, shuffle=True)
    
    print(chunks_train.shape, targets_train.shape, chunks_val.shape, targets_val.shape, chunks_test.shape, targets_test.shape)
    
    return {
        "chunks_train": chunks_train,
        "targets_train": targets_train,
        "chunks_val": chunks_val,
        "targets_val": targets_val,
        "chunks_test": chunks_test,
        "targets_test": targets_test
    }

### DEPRECATED
# def readCsv(file_name):
#     file_obj = open(file_name, "r")
#     age = int(file_obj.readline()[8:-1])
#     data = file_obj.readlines()
#     file_obj.close()
#     data[0] = data[0].replace("# ", "")
#     processed_file_name = str(age) + ".csv"
#     processed_file_obj = open(processed_file_name, "w")
#     processed_file_obj.writelines(data)
#     df = pd.read_csv(processed_file_name)
#     processed_file_obj.close()
#     os.remove(processed_file_name)
#     return df, age

# def df2edf(df, sfreq=SAMPLING_FREQUENCY):
#     ch_names = df.columns.to_list()
#     ch_types = ['eeg'] * len(ch_names)
#     info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
#     return mne.io.RawArray(df[ch_names].to_numpy(copy=True).T / 1e6, info, verbose=False) #data in microvolts