import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, average_precision_score, accuracy_score, f1_score

from confusion_matrix.cf_matrix import make_confusion_matrix
from .plotting import printScores, plotROC, plotPR

def evalConfInt(y_test, y_prob, stratum_idx_vals=None, n_bootstraps=1000, m_sample=None, scorer=average_precision_score, verbose=2, n_sigma=2):
    assert len(y_test) == len(y_prob), "y_test and y_prob should have the same lengths"
    if not m_sample:
        m_sample = len(y_test)
    idx = np.array(range(len(y_test)))
    
    scores = []
    if verbose >=2 :
        print("Bootstrap scores computing...")
        for _ in tqdm(range(n_bootstraps)):
            if type(stratum_idx_vals) != type(None):
                idx_bs = []
                for val in set(stratum_idx_vals):
                    stratum_idx = idx[stratum_idx_vals == val]
                    idx_bs += np.random.choice(stratum_idx, size=len(stratum_idx), replace=True).tolist()
            else:
                idx_bs = np.random.choice(idx, size=m_sample, replace=True)
            scores.append(scorer(y_test[idx_bs], y_prob[idx_bs]))
    else:
        for _ in range(n_bootstraps):
            if type(stratum_idx_vals) != type(None):
                idx_bs = []
                for val in set(stratum_idx_vals):
                    stratum_idx = idx[stratum_idx_vals == val]
                    idx_bs += np.random.choice(stratum_idx, size=len(stratum_idx), replace=True).tolist()
            else:
                idx_bs = np.random.choice(idx, size=m_sample, replace=True)
            scores.append(scorer(y_test[idx_bs], y_prob[idx_bs]))

    scores = np.array(scores)
    
    if verbose >= 2:
        plt.figure(figsize=(4, 2.5))
        plt.hist(scores, bins=50)
        plt.axvline(x = scores.mean(), color = 'tab:orange', label = 'mean')
        plt.axvline(x = scores.mean() - n_sigma*scores.std(), color = 'tab:red', label = f'mean - {n_sigma}*sigma')
        plt.axvline(x = scores.mean() + n_sigma*scores.std(), color = 'tab:red', label = f'mean + {n_sigma}*sigma')
        plt.show()
    
    return scores.mean(), n_sigma*scores.std()
    
def evaluateMetrics(y_test, y_prob, msmnt_idx=None, threshold=0.5, verbose=2):
    #Scores
    if verbose >= 2:
        printScores(y_test, y_prob > threshold)
    
        #Scores for each record
        if type(msmnt_idx) != type(None):
            y_m_df = pd.DataFrame()
            y_m_df["y_test"]=y_test
            y_m_df["y_prob"]=y_prob
            y_m_df["msmnt_idx"]=msmnt_idx

            print("Idx accuracies:")
            idx_accs = {}
            for idx in y_m_df["msmnt_idx"].value_counts().index:
                y_m_df_idx=y_m_df[y_m_df["msmnt_idx"] == idx]
                y_tr = y_m_df_idx["y_test"].values
                y_pr = y_m_df_idx["y_prob"].values>threshold
                idx_accs[idx] = accuracy_score(y_tr, y_pr)
            display(pd.DataFrame(sorted(idx_accs.items(), key=lambda x: x[0]), columns=["Measurement index", "Accuracy"]))
            
    #confusion matrix
    cf_mtrx = confusion_matrix(y_test, y_prob > threshold)
    if verbose >= 2:
        print(f'Sensitivity : {cf_mtrx[1,1]/(cf_mtrx[1,0]+cf_mtrx[1,1]):0.3f}')
        print(f'Specificity : {cf_mtrx[0,0]/(cf_mtrx[0,0]+cf_mtrx[0,1]):0.3f}')
            
    #pr auc
    n_sigma = 2
    pr_auc_mean, pr_auc_std = evalConfInt(y_test, y_prob, msmnt_idx, n_bootstraps=1000, n_sigma=n_sigma, verbose=verbose)
    if verbose:
        print(f"PR AUC, n_sigma={n_sigma}: {pr_auc_mean:0.3f}+-{pr_auc_std:0.3f}")
        
    #Plots
    if verbose:
        n_plots = 4
        add_row = 1 if n_plots//4 != n_plots/4 else 0
        fig, axes = plt.subplots(n_plots//4+add_row, 4, figsize=(20, 5))
        i = 0

        #confusion matrix
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
    
    return pr_auc_mean, pr_auc_std