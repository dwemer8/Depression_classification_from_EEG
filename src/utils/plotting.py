from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, periodogram
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, \
                            accuracy_score, f1_score, recall_score, precision_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

from src.utils import DEFAULT_SEED
from src.utils.common import printLog

def dict_to_df(results):
    return pd.DataFrame(dict([(key, [f"{value:.3f}"]) for key, value in results.items()]))

##########################################################################
#graphs
##########################################################################
def plotSamplesImshow(data, nrows=5, ncols=5, figsize=(10, 10), title=None, axes_title=""):
    '''
    Plotting EEG recordings as images
    '''
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(title)

    for i in range(nrows):
        for j in range(ncols):
            if ncols == 1 and nrows == 1:
                axes.imshow(data[i].cpu().squeeze(0), cmap='gray')
                # axes.axis('off')
                axes.set_title(axes_title + f"({i},{j})", fontsize=10)
            
            elif ncols == 1:
                axes[i].imshow(data[i].cpu().squeeze(0), cmap='gray')
                axes[i].axis('off')
                axes[i].set_title(axes_title + f"({i},{j})", fontsize=10)
                
            elif nrows == 1:
                axes[j].imshow(data[j].cpu().squeeze(0), cmap='gray')
                axes[j].axis('off')
                axes[j].set_title(axes_title + f"({i},{j})", fontsize=10)
                
            else:
                axes[i][j].imshow(data[i*nrows +j].cpu().squeeze(0), cmap='gray')
                axes[i][j].axis('off')
                axes[i][j].set_title(axes_title + f"({i},{j})", fontsize=10)
    plt.tight_layout(pad=0.)
    
def plotSamplesPlot(data, nrows=5, ncols=5, figsize=(10, 10), title=None, axes_title="", axis_regime="on"):
    '''
    Plotting EEG recordings as plots
    '''
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(title)

    for i in range(nrows):
        for j in range(ncols):
            if ncols == 1 and nrows == 1:
                axes.plot(data[i].cpu().squeeze(0).T)
                axes.axis(axis_regime)
                axes.set_title(axes_title + f"({i},{j})", fontsize=10)
            
            elif ncols == 1:
                axes[i].plot(data[i].cpu().squeeze(0).T)
                axes[i].axis(axis_regime)
                axes[i].set_title(axes_title + f"({i},{j})", fontsize=10)
                
            elif nrows == 1:
                axes[j].plot(data[j].cpu().squeeze(0).T)
                axes[j].axis(axis_regime)
                axes[j].set_title(axes_title + f"({i},{j})", fontsize=10)
                
            else:
                axes[i][j].plot(data[i*nrows +j].cpu().squeeze(0).T)
                axes[i][j].axis(axis_regime)
                axes[i][j].set_title(axes_title + f"({i},{j})", fontsize=10)
    plt.tight_layout(pad=0.5)

#scipy functions

def plotSpectrogram(timeseries, fs=125, **kwargs):
    f, t, Sxx = spectrogram(timeseries, fs=125, **kwargs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
def plotPsd(timeseries, fs=125, xlim=None, label=None, **kwargs):
    f, P = periodogram(timeseries, fs=fs, scaling='density', **kwargs)
    plt.plot(f, P, label=label)
    plt.xlim(xlim)

def plotROC(y_test, y_prob, ax):
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    ax.step(fpr, tpr, color='b', alpha=0.2,
             where='post')
    ax.fill_between(fpr, tpr, alpha=0.2, color='b')

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC AUC={0:0.2f}'.format(
              roc_auc_score(y_test, y_prob)))  

# sci-kit learn functions
    
def plotPR(y_test, y_prob, ax):
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    
    ax.plot(rec, prec, color='b', alpha=0.2)
    ax.fill_between(rec, prec, 0, color='b', alpha=0.2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR-AUC={0:0.2f}'.format(average_precision_score(y_test, y_prob)))  

def printScores(y_tr, y_pr, is_display=True):
    acc = accuracy_score(y_tr, y_pr)
    f1 = f1_score(y_tr, y_pr)
    prec = precision_score(y_tr, y_pr)
    rec = recall_score(y_tr, y_pr)
    if is_display:
        display(pd.DataFrame({"F1": [f1], "Recall": [rec], "Precision": [prec], "Accuracy": [acc]}))
    else:
        print(f"F1: {f1:0.3f}, precision: {prec:0.3f}, recall: {rec:0.3f}, accuracy: {acc:0.3f}") 

# dataset

def dataset_hists(train_set, val_set, test_set, chunk_bins=20, target_bins=3, chunk_range=None, target_range=None):
    for data_set in [train_set, val_set, test_set]:
        n_channels = data_set["chunk"].shape[-2]
        fig, ax = plt.subplots(nrows=1, ncols=(n_channels+1), figsize=(16, 2))
        fig.suptitle("train" if data_set is train_set else "val" if data_set is val_set else "test")
        for i in range(n_channels): 
            ax[i].hist(data_set["chunk"].squeeze()[:, i, :].flatten(), bins=chunk_bins, range=chunk_range)
            ax[i].set_title(f"Channel {i}")
        ax[3].hist(data_set["target"], bins=target_bins, range=target_range)
        ax[3].set_title(f"Target")
        plt.show()

# embeddings visualization

def plotData(X, y, method="pca", ax=plt, plot_type="regression", seed=DEFAULT_SEED, **kwargs):
    n_components = 2
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=seed, whiten=True, svd_solver="full", **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=seed, **kwargs)
    elif method == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=seed, **kwargs)
    else:
        raise "NotImplementedError"

    X_reduced = reducer.fit_transform(X)

    if method == "pca":
        display(pd.DataFrame({"Explained variance": reducer.explained_variance_, "Ratio": reducer.explained_variance_ratio_}))
    
    if plot_type == "regression":
        return sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], ax=ax, alpha=y)
    elif plot_type == "classification":
        return sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], ax=ax, alpha=0.3, hue=y)
    else:
        raise ValueError(f"Unexpected type {plot_type}")

def plotSamplesFromDataset(dataset):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 2))
    fig.suptitle("Samples")
    for i in range(3): ax[i].plot(dataset[np.random.choice(list(range(len(dataset))))].squeeze().T)
    plt.show()

def printDatasetMeta(val_dataset, test_dataset, train_dataset=None, pretrain_dataset=None, logfile=None):
    if pretrain_dataset is not None: print("Pretrain dataset:", len(pretrain_dataset))
    if train_dataset is not None: printLog("Train dataset:", len(train_dataset), logfile=logfile)
    printLog("Val dataset:", len(val_dataset), logfile=logfile)
    printLog("Test dataset:", len(test_dataset), logfile=logfile)

    if pretrain_dataset is not None: print("Pretrain sample shape:", pretrain_dataset[0].shape)
    if train_dataset is not None: printLog("Train sample shape:", train_dataset[0].shape, logfile=logfile)
    printLog("Val sample shape:", val_dataset[0].shape, logfile=logfile)
    printLog("Test sample shape:", test_dataset[0].shape, logfile=logfile)

    if pretrain_dataset is not None: print("Pretrain sample type:", pretrain_dataset[0].type())
    if train_dataset is not None: printLog("Train sample type:", train_dataset[0].type(), logfile=logfile)
    printLog("Val sample type:", val_dataset[0].type(), logfile=logfile)
    printLog("Test sample type:", test_dataset[0].type(), logfile=logfile)

def printDataloaderMeta(val_dataloader, test_dataloader, train_dataloader=None, pretrain_dataloader=None, logfile=None):
    if pretrain_dataloader is not None: print("Pretrain dataloader:", len(pretrain_dataloader))
    if train_dataloader is not None: printLog("Train dataloader:", len(train_dataloader), logfile=logfile)
    printLog("Val dataloader:", len(val_dataloader), logfile=logfile)
    printLog("Test dataloader:", len(test_dataloader), logfile=logfile)

def plot_se(x, y, y_se=None, label=None, ax=plt, alpha=0.3, marker="."):
    ax.plot(x, y, label=label, marker=marker)
    if y_se is not None:
        ax.fill_between(x, y - y_se, y + y_se, alpha=alpha)