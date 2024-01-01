from IPython.display import display
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, periodogram
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, \
                            accuracy_score, f1_score, recall_score, precision_score
                             

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