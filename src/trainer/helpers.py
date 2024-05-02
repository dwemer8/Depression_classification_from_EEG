import time 
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from src.utils.common import printLog
from src.utils.plotting import plotData

def save_sklearn_model(model, filename):
    dump(model, filename)

def load_sklearn_model(filename):
    return load(filename)

def get_embeddings(
    model, 
    test_dataset, 
    targets_test, 
    avg_over_time=False,
    mode="eval",
    device="cpu",
    logfile=None
):
    if mode == "train": model.eval()
    embeddings_test = model.encode(test_dataset[:].to(device)).detach().cpu().numpy()
    if avg_over_time: embeddings_test = embeddings_test.mean(-1) #TAKE MEAN OVER TIME AXIS
    embeddings_test = embeddings_test.reshape(len(test_dataset), -1)
    if mode == "train": model.train()

    X = np.array(embeddings_test)
    y = np.array(targets_test)

    if np.any(np.isnan(X)): 
        nan_indexes = np.where(np.isnan(X))
        printLog("Test dataset sample:", test_dataset[nan_indexes[0]], "Embedding:", embeddings_test[nan_indexes[0]], logfile=logfile)
        raise ValueError("NaN in embeddings")
    if np.any(np.isnan(y)): 
        raise ValueError("NaN in targets")

    indices = list(range(len(X)))
    np.random.shuffle(indices)
    return X[indices], y[indices]

def vizualize(
    model, 
    imgs,
    test_dataset, 
    targets_test, 
    avg_embeddings_over_time, 
    mode, 
    device, 
    logfile, 
    epoch, 
    step, 
    buffer_max, 
    buffer_cnt,
    plot_type,
    verbose
):
    start_plotting_time = time.time()
    
    if buffer_cnt >= buffer_max:
        clear_output(wait=True)
        buffer_cnt = 0
    buffer_cnt += 1

    printLog(f"Epoch {epoch}, step {step}", logfile=logfile)

    #pca embeddnigs
    if verbose - 1 > 0: printLog("Plotting PCA...", logfile=logfile)
    X, y = get_embeddings(
        model, 
        test_dataset, 
        targets_test, 
        avg_over_time=avg_embeddings_over_time, 
        mode=mode, 
        device=device, 
        logfile=logfile,
    )
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    plotData(X, y, method="pca", ax=ax[0], plot_type=plot_type)
    
    #plot reconstruction
    if verbose - 1 > 0: printLog("Plotting reconstruction...", logfile=logfile)
    ax[1].plot(imgs[0].squeeze()[0], label="data", color="b", marker="o")
    if mode == "train": model.eval()
    imgs_reconstructed = model.reconstruct(imgs.to(device))
    if mode == "train": model.train()
    ax[1].plot(imgs_reconstructed[0].squeeze()[0].detach().cpu(), label="approximation", color="r")
    plt.show()

    end_plotting_time = time.time()
    if verbose - 2 > 0: printLog(f"Plotting time: {end_plotting_time - start_plotting_time} s", logfile=logfile)

def eval_ml_model(
    model,
    test_dataset,
    targets_test,
    logger,
    logfile,
    avg_embeddings_over_time,
    mode,
    device,
    verbose,
    ml_model,
    ml_param_grid,
    ml_eval_function,
    ml_eval_function_kwargs,
    ml_eval_function_tag,
    ml_metric_prefix
):
    start_evaluation_time = time.time()

    if ml_model is None or ml_param_grid is None or ml_eval_function is None: raise ValueError("Some ml parameter is not defined")

    if verbose - 1 > 0: printLog("Classifier/regressor metrics evaluation...", logfile=logfile)
    X, y = get_embeddings(
        model, 
        test_dataset, 
        targets_test, 
        avg_over_time=avg_embeddings_over_time, 
        mode=mode, 
        device=device, 
        logfile=logfile,
    )
    if verbose - 2 > 0: printLog("Embeddings shape:", X.shape, logfile=logfile)
    results = {}
    for func, kwargs, tag in zip(ml_eval_function, ml_eval_function_kwargs, ml_eval_function_tag):
        results[tag] = (func(X, y, ml_model, ml_param_grid, logfile=logfile, **kwargs))
    logger.update_other({ml_metric_prefix: results})

    end_evaluation_time = time.time()
    if verbose - 2 > 0: printLog(f"Evaluation time: {end_evaluation_time - start_evaluation_time} s", logfile=logfile)