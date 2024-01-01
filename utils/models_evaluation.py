from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, accuracy_score, mean_squared_error as mse
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from copy import deepcopy

from .metrics_evaluation import evaluateMetrics
from utils import SEED

##########################################################################################
# model evaluation
##########################################################################################

def evaluateClassifier(
    X,
    y,
    model, 
    param_grid, 
    verbose=2,
    test_size=0.33,
    SEED=SEED
):
    def evaluate(clf, X, y):
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)[:, 1]
        pr_auc_mean, pr_auc_std = evaluateMetrics(y, y_proba, verbose=verbose)
        accuracy = accuracy_score(y, y_pred)
        return pr_auc_mean, pr_auc_std, accuracy
    
    model = deepcopy(model)

    if verbose: print("\nData split") 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED, shuffle=True)

    if verbose: print("\nGridSearchCV")
    clf = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=make_scorer(f1_score),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
        verbose=verbose
    ) 

    clf.fit(X_train, y_train)

    if verbose: print("\nEvaluation on the train data")
    pr_auc_mean_train, pr_auc_std_train, accuracy_train = evaluate(clf, X_train, y_train)

    if verbose >= 2:
        print("Best classifier:")
        print("Parameters:", clf.best_params_)
        print("Score:", clf.best_score_)
        print(f"Train accuracy: {accuracy_train}")
    
    if verbose: print("\nEvaluation on the test data")
    pr_auc_mean_test, pr_auc_std_test, accuracy_test = evaluate(clf, X_test, y_test)
    
    return {
        "clf": clf, 

        "pr_auc_mean_test": pr_auc_mean_test, 
        "pr_auc_std_test": pr_auc_std_test, 
        "accuracy_test": accuracy_test,

        "f1_val": clf.best_score_, 
        "pr_auc_mean_train": pr_auc_mean_train, 
        "pr_auc_std_train": pr_auc_std_train, 
        "accuracy_train": accuracy_train,
    }

def evaluateRegressor(
    X,
    y,
    model, 
    param_grid, 
    verbose=2,
    test_size=0.33,
    SEED=SEED
):
    def evaluate(reg, X, y):
        y_pred = reg.predict(X)
        return mse(y, y_pred)
    
    model = deepcopy(model)

    if verbose: print("\nData split") 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED, shuffle=True)

    if verbose: print("\nGridSearchCV")
    reg = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=make_scorer(mse, greater_is_better=False),
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
        verbose=verbose
    ) 

    reg.fit(X_train, y_train)

    if verbose: print("\nEvaluation on the train data")
    mse_train = evaluate(reg, X_train, y_train)
    
    if verbose >= 2:
        print("Best classifier:")
        print("Parameters:", reg.best_params_)
        print("Score:", reg.best_score_)
    
    if verbose: print("\nEvaluation on the test data")
    mse_test = evaluate(reg, X_test, y_test)

    return {
        "reg": reg, 
        "mse_test": mse_test,
        "mse_val": reg.best_score_,
        "mse_train": mse_train,
    }