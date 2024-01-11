from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, accuracy_score, average_precision_score, roc_auc_score, mean_squared_error as mse
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from copy import deepcopy

from .metrics_evaluation import evaluateMetrics, evaluateMetrics_cv
from .common import get_object_name
from utils import SEED

##########################################################################################
# model evaluation
##########################################################################################

def evaluateClassifier(
    X,
    y,
    model, 
    param_grid, 
    verbose=1,
    test_size=0.33,
    SEED=SEED,
    cv_scorer=accuracy_score,
    metrics = [], #[(average_precision_score, "soft"), (roc_auc_score, "soft"), (accuracy_score, "hard"), (f1_score, "hard")],
    metrics_for_CI = [], #[(average_precision_score, "soft"), (roc_auc_score, "soft"), (accuracy_score, "hard"), (f1_score, "hard")],
    n_bootstraps = 1000,
):
    def evaluate(clf, X, y, metrics_for_CI=metrics_for_CI):
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)[:, 1]
        estimates = evaluateMetrics(
            y, 
            y_proba, 
            verbose=(verbose-1), 
            metrics=metrics,
            metrics_for_CI=metrics_for_CI,
            n_bootstraps=n_bootstraps,
        )
        return estimates
    
    model = deepcopy(model)

    if verbose > 0: print("Data split") 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED, shuffle=True)

    if verbose > 0: print("GridSearchCV")
    clf = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=make_scorer(cv_scorer),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
        verbose=verbose
    ) 

    clf.fit(X_train, y_train)

    if verbose > 0: print("Evaluation on the train data")
    estimates_train = evaluate(clf, X_train, y_train)

    if verbose - 1 > 0:
        print("Best classifier:")
        print("Parameters:", clf.best_params_)
        print("Score:", clf.best_score_)
        print(f"Train accuracy: {accuracy_train}")
    
    if verbose > 0: print("Evaluation on the test data")
    estimates_test = evaluate(clf, X_test, y_test)
    
    return {
        "train": estimates_train,
        "test": estimates_test
    }

def evaluateClassifier_inner_outer_cv(
    X,
    y,
    model, 
    param_grid, 
    verbose=1,
    SEED=SEED,
    cv_scorer=accuracy_score,
    metrics = [], #[(average_precision_score, "soft"), (roc_auc_score, "soft"), (accuracy_score, "hard"), (f1_score, "hard")],
    n_splits=10,
):
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED+1)

    if verbose > 0: print("GridSearchCV")
    clf = GridSearchCV(
        estimator=model,  
        param_grid=param_grid,
        scoring=make_scorer(cv_scorer),
        cv=inner_cv,
        n_jobs=-1,
        verbose=(verbose-1)
    )
    clf.fit(X, y)

    if verbose > 0: print("Evaluation on the train data")
    estimates_train = evaluateMetrics_cv(clf, X, y, inner_cv, metrics, verbose=(verbose-1))
    if verbose > 0: print("Evaluation on the test data")
    estimates_test = evaluateMetrics_cv(clf, X, y, outer_cv, metrics, verbose=(verbose-1))
    
    return {
        "train": estimates_train,
        "test": estimates_test
    }

def evaluateRegressor(
    X,
    y,
    model, 
    param_grid, 
    verbose=1,
    test_size=0.33,
    SEED=SEED
):
    '''
    TODO: update
    '''
    def evaluate(reg, X, y):
        y_pred = reg.predict(X)
        return mse(y, y_pred)
    
    model = deepcopy(model)

    if verbose > 0: print("Data split") 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED, shuffle=True)

    if verbose > 0: print("GridSearchCV")
    reg = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=make_scorer(mse, greater_is_better=False),
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
        verbose=(verbose-1)
    ) 

    reg.fit(X_train, y_train)

    if verbose > 0: print("Evaluation on the train data")
    mse_train = evaluate(reg, X_train, y_train)
    
    if verbose - 1 > 0:
        print("Best classifier:")
        print("Parameters:", reg.best_params_)
        print("Score:", reg.best_score_)
    
    if verbose > 0: print("Evaluation on the test data")
    mse_test = evaluate(reg, X_test, y_test)

    return {
        "test" : {"mse" : mse_test},
        "val" : {"mse" : reg.best_score_},
        "train" : {"mse" : mse_train},
    }