import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from IPython.display import display

from confusion_matrix.cf_matrix import make_confusion_matrix
from ..utils.plotting import printScores, plotROC, plotPR
from ..utils.common import get_object_name

def compute_bootstrapped_score(y_test, y_prob, scorer, m_sample=None, stratum_vals=None):
    idx = np.array(range(len(y_test)))
    if m_sample is None: m_sample = len(y_test) #bootstrap sample size
        
    if stratum_vals is not None: #select equal number of samples from each category
        idx_bs = [] 
        for val in set(stratum_vals):
            stratum_idx = idx[stratum_vals == val] 
            idx_bs += np.random.choice(stratum_idx, size=len(stratum_idx), replace=True).tolist()
    else:
        idx_bs = np.random.choice(idx, size=m_sample, replace=True)

    try:
        return scorer(y_test[idx_bs], y_prob[idx_bs])
    except:
        return np.nan

def get_bootstrap_estimates(
    y_test, 
    y_prob, 
    stratum_vals=None, 
    n_bootstraps=1000, 
    m_sample=None, 
    scorer=accuracy_score, 
    verbose=1, 
):
    assert len(y_test) == len(y_prob), "y_test and y_prob should have the same lengths"
        
    scores = []
    if verbose > 0:
        print("Bootstrap scores computing...")
        for _ in tqdm(range(n_bootstraps)):
            scores.append(compute_bootstrapped_score(
                y_test, 
                y_prob, 
                scorer, 
                stratum_vals=stratum_vals, 
                m_sample=m_sample
            ))
    else:
        for _ in range(n_bootstraps):
            scores.append(compute_bootstrapped_score(
                y_test, 
                y_prob, 
                scorer, 
                stratum_vals=stratum_vals, 
                m_sample=m_sample
            ))
    return np.array(scores)

def get_bootstrap_estimates_for_metrics(
    y_test, 
    y_prob, 
    stratum_vals=None, 
    threshold=0.5, 
    verbose=1, 
    n_bootstraps=1000,
    metrics_for_CI = [],# [(average_precision_score, "soft"), (accuracy_score, "hard")],
):
    estimates = {}
    y_pred = y_prob > threshold

    #computing CIs for metrics
    for method, method_type in metrics_for_CI:
        method_name = get_object_name(method)
        if method_type == "soft": y_pred_ = y_prob
        elif method_type == "hard": y_pred_ = y_pred
        else: raise ValueError(f"Unknown metric type {method_type}")
        estimates[method_name] = get_bootstrap_estimates(
            y_test, 
            y_pred_, 
            stratum_vals=stratum_vals,
            n_bootstraps=n_bootstraps, 
            scorer=method, 
            verbose=(verbose-1),
        )
    
    return estimates

def evalConfInt(
    y_test, 
    y_prob, 
    stratum_vals=None, 
    n_bootstraps=1000, 
    m_sample=None, 
    scorer=average_precision_score, 
    alpha=0.05,
    verbose=1, 
):
    assert len(y_test) == len(y_prob), "y_test and y_prob should have the same lengths"
        
    scores = []
    if verbose > 0:
        print("Bootstrap scores computing...")
        for _ in tqdm(range(n_bootstraps)):
            scores.append(compute_bootstrapped_score(y_test, y_prob, scorer, stratum_vals=stratum_vals, m_sample=m_sample))
    else:
        for _ in range(n_bootstraps):
            scores.append(compute_bootstrapped_score(y_test, y_prob, scorer, stratum_vals=stratum_vals, m_sample=m_sample))
            
    scores = np.array(scores)
    if verbose > 0: print(f"N samples = {len(scores)}")
    
    nans_share = np.sum(np.isnan(scores).astype(int))/len(scores)
    if nans_share > 0.5:
        raise ValueError(f"There is {nans_share*100:.0f}% NaNs in bootstrapped scores.")
    
    perc = sts.norm.ppf(1 - alpha/2)
    estimation = np.nanmean(scores)
    se = np.nanstd(scores) * perc
    
    if verbose > 0:
        plt.figure(figsize=(4, 2.5))
        plt.hist(scores, bins=50)
        plt.axvline(x = estimation, color = 'tab:orange', label = 'mean')
        plt.axvline(x = estimation - se, color = 'tab:red', label = f'mean - se_{1-alpha:.2f}')
        plt.axvline(x = estimation + se, color = 'tab:red', label = f'mean + se_{1-alpha:.2f}')
        plt.show()
    
    return estimation, se

def compute_accuracy_for_each_measurement(y_test, y_prob, msmnt_idx, threshold=0.5, verbose=1):
    y_m_df = pd.DataFrame()
    y_m_df["y_test"]=y_test
    y_m_df["y_prob"]=y_prob
    y_m_df["msmnt_idx"]=msmnt_idx

    if verbose > 0: print("Idx accuracies:")
    idx_accs = {}
    for idx in y_m_df["msmnt_idx"].value_counts().index:
        y_m_df_idx=y_m_df[y_m_df["msmnt_idx"] == idx]
        y_tr = y_m_df_idx["y_test"].values
        y_pr = y_m_df_idx["y_prob"].values>threshold
        idx_accs[idx] = accuracy_score(y_tr, y_pr)
    return pd.DataFrame(sorted(idx_accs.items(), key=lambda x: x[0]), columns=["Measurement index", "Accuracy"])

def compute_sensitivity_specificity(y_test, y_prob, threshold):
    cf_mtrx = confusion_matrix(y_test, y_prob > threshold)
    sensitivity = cf_mtrx[1,1]/(cf_mtrx[1,0]+cf_mtrx[1,1])
    specificity = cf_mtrx[0,0]/(cf_mtrx[0,0]+cf_mtrx[0,1])
    return sensitivity, specificity
    
def evaluateMetrics(
    y_test, 
    y_prob, 
    stratum_vals=None, 
    threshold=0.5, 
    verbose=1, 
    n_bootstraps=1000,
    alpha=0.05,
    metrics_for_CI = [],# [(average_precision_score, "soft"), (accuracy_score, "hard")],
    metrics = [],#[(average_precision_score, "soft"), (accuracy_score, "hard")],
):
    estimates = {}
    y_pred = (y_prob > threshold).astype(int)

    #computing metrics
    for method, method_type in metrics:
        method_name = get_object_name(method)
        if method_type == "soft": y_pred_ = y_prob
        elif method_type == "hard": y_pred_ = y_pred
        else: raise ValueError(f"Unknown metric type {method_type}")
        est = method(y_test, y_pred_)
        estimates[method_name] = est
        if verbose > 0: print(f"{method_name}: {est:0.3f}")

    #computing CIs for metrics
    for method, method_type in metrics_for_CI:
        method_name = get_object_name(method)
        if method_type == "soft": y_pred_ = y_prob
        elif method_type == "hard": y_pred_ = y_pred
        else: raise ValueError(f"Unknown metric type {method_type}")
        est, se = evalConfInt(
            y_test, y_pred_, stratum_vals, n_bootstraps=n_bootstraps, alpha=alpha, verbose=(verbose-1), scorer=method
        )
        estimates["bs." + method_name] = est
        estimates["bs." + method_name + ".se"] = se
        if verbose > 0: print(f"{method_name}, 0.95% interval from bootstrap: {est:0.3f}+-{se:0.3f}")
            
    ### Additional metrics ###
    
    #Scores
    if verbose - 1 > 0:
        printScores(y_test, y_prob > threshold)
    
        #Print scores for each type of measurement
        if stratum_vals is not None:
            accuracies = compute_accuracy_for_each_measurement(y_test, y_prob, stratum_vals, threshold=0.5, verbose=(verbose-1))
            display(accuracies)
            
    #specificity, sensitivity
    estimates["sensitivity"], estimates["specificity"] = compute_sensitivity_specificity(y_test, y_prob, threshold)
    sensitivity = estimates["sensitivity"]
    specificity = estimates["specificity"]
    if verbose - 1 > 0:
        print(f'Sensitivity : {sensitivity:0.3f}')
        print(f'Specificity : {specificity:0.3f}')
    
    ### Plots ###
    if verbose - 1 > 0:
        n_plots = 4
        add_row = 1 if n_plots//4 != n_plots/4 else 0
        fig, axes = plt.subplots(n_plots//4+add_row, 4, figsize=(20, 5))
        i = 0

        #confusion matrix
        cf_mtrx = confusion_matrix(y_test, y_prob > threshold)
        labels = ['True Neg','False Pos','False Neg','True Pos'] # TN, FP = confusion[0, 0], confusion[0, 1]; FN, TP = confusion[1, 0], confusion[1, 1]
        make_confusion_matrix(cf_mtrx, group_names=labels, sum_stats=False, ax=axes[i])
        i += 1

        # score-threshold plots
        thresholds = np.linspace(0, 1, 100)
        axes[i].plot(thresholds, [accuracy_score(y_test, y_prob > threshold) for threshold in thresholds], label="Accuracy")
        axes[i].plot(thresholds, [f1_score(y_test, y_prob > threshold) for threshold in thresholds], label="f1")
        axes[i].legend()
        i += 1

        #ROC
        plotROC(y_test, y_prob, axes[i])
        i +=1

        #PR
        plotPR(y_test, y_prob, axes[i])
        i += 1

        plt.tight_layout()
        plt.show()
    
    return estimates

def evalConfInt_cv(clf, X, y, cv, scorer, alpha=0.05, verbose=1):
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scorer, verbose=max(verbose-1, 0))
    perc = sts.norm.ppf(1 - alpha/2)
    est = scores.mean()
    se = scores.std() * perc
    return est, se

def evaluateMetrics_cv(
    clf,
    X,
    y,
    cv,
    metrics,
    alpha=0.05,
    verbose=1
):
    estimates = {}
    for method, method_type in metrics:
        method_name = get_object_name(method)
            
        if method_type == "soft": scorer = make_scorer(method, needs_proba=True)
        elif method_type == "hard": scorer = make_scorer(method)
            
        est, se = evalConfInt_cv(clf, X, y, cv, scorer, alpha=alpha, verbose=(verbose-1))
        
        estimates["cv." + method_name] = est
        estimates["cv." + method_name + ".se"] = se
        if verbose > 0: print(f"{method_name}, {1 - alpha:.02f}% interval from cross-validation: {est:0.3f}+-{se:0.3f}")
    return estimates
