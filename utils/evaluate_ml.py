import time
import numpy as np

def evaluate_ml(
    model,
    device="cuda",
    test_dataset=None,
    targets_test=None,
    
    ml_model=None,
    ml_param_grid=None, 
    ml_eval_function=None,
    ml_eval_function_kwargs=None,
    ml_eval_function_tag=None,

    avg_embeddings_over_time=False,
    verbose=0,

    **kwargs
):  
    model.eval()
    if verbose - 1 > 0: print("Model is in evaluation mode")

    try:
        
        #embeddings
        def get_embeddings(
            model, 
            test_dataset, 
            targets_test, 
            avg_over_time=False
        ):
            embeddings_test = model.encode(test_dataset[:].to(device)).detach().cpu().numpy()
            if avg_over_time: embeddings_test = embeddings_test.mean(-1) #TAKE MEAN OVER TIME AXIS
            embeddings_test = embeddings_test.reshape(len(test_dataset), -1)

            X = np.array(embeddings_test)
            y = np.array(targets_test)

            if np.any(np.isnan(X)): 
                nan_indexes = np.where(np.isnan(X))
                print("Test dataset sample:", test_dataset[nan_indexes[0]], "Embedding:", embeddings_test[nan_indexes[0]])
                raise ValueError("NaN in embeddings")
            if np.any(np.isnan(y)): 
                raise ValueError("NaN in targets")

            indices = list(range(len(X)))
            np.random.shuffle(indices)
            return X[indices], y[indices]

        #classifier/regressor metrics evaluation
        start_evaluation_time = time.time()
        
        if ml_model is None or ml_param_grid is None or ml_eval_function is None: raise ValueError("Some ml parameter is not defined")

        if verbose - 1 > 0: print("Getting bootstrap values...")
        X, y = get_embeddings(model, test_dataset, targets_test, avg_over_time=avg_embeddings_over_time)
        if verbose - 2 > 0: print("Embeddings shape:", X.shape)
        results = {}
        for func, kwargs, tag in zip(ml_eval_function, ml_eval_function_kwargs, ml_eval_function_tag):
            results[tag] = func(X, y, ml_model, ml_param_grid, **kwargs)

        end_evaluation_time = time.time()
        if verbose - 2 > 0: print(f"Evaluation time: {end_evaluation_time - start_evaluation_time} s")
        
        return results

    except KeyboardInterrupt:
        return {}