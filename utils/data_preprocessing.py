import numpy as np
import pandas as pd
import mne
from tqdm.auto import tqdm
import pickle

from .data_reading import readDataExt_one

SAMPLING_FREQUENCY = 250 #Hz
L_FREQ = 0.1 #Hz
H_FREQ = 30 #Hz
AMPLITUDE_THRESHOLD = 0.005 #V
STD_THRESHOLD = 5
TRANSIENT_STD_THRESHOLD = 3

def pickChunks(df, ch_names, n_samples_per_chunk=124):
    chunks = []
    start_idx = 0
    end_idx = start_idx + n_samples_per_chunk

    while end_idx <= df.shape[0]:
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
    #average reference, filtration
    ch_names = df.columns.to_list()[1:] #delete 'time' #['t6', 't4', 'o1', 'f8', 'p4', 'c4', 't3', 'f7', 'f3', 'o2', 'f4', 'c3', 'p3', 't5', 'cz', 'fp1', 'fp2', 'pz', 'fz']
    ch_types = ['eeg'] * 19
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=source_freq)
    raw = mne.io.RawArray(df[ch_names].to_numpy(copy=True).T / divider, info, verbose=False) #data in microvolts
    raw, _ = mne.set_eeg_reference(raw, ref_channels='average', verbose=False) #average reference
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', verbose=False) #filtration
    raw = raw.resample(target_freq, npad='auto')
    df = raw.to_data_frame() #in microvolts

    #clipping
    df.loc[:, ch_names].clip(-amplitude_threshold, amplitude_threshold, inplace=True)

    #normalization
    df.loc[:, ch_names] = (df[ch_names] - df[ch_names].mean())/df[ch_names].std()

    return pickChunks(df, ch_names, **kwargs)

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

def save_preprocessed_data(chunks_list, file):
    with open(file, "wb") as f:
        pickle.dump(chunks_list, f)