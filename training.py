import time

import torch
import matplotlib.pyplot as plt
import numpy as np

from models.VAE import VAE, BetaVAE_H, BetaVAE_B
from models.AE import AE_framework
from utils.plotting import plotData

def check_instance(object, types):
    for class_type in types:
        if isinstance(object, class_type):
            return True
    return False

def mask_chunks(chunks, mask_ratio = 0.5):
    if mask_ratio == 0:
        return chunks
        
    B = chunks.shape[0]
    length = chunks.shape[-1]
    mask_length = int(length*mask_ratio)
    max_mask_idx = length - mask_length
    mask_idxs = (max_mask_idx * torch.rand(*list(chunks.shape[:-1]))).type(torch.int)
    
    chunks = chunks.clone().detach()
    chunks_mean = chunks.mean(dim=-1, keepdim=True)
    chunks_std = chunks.std(dim=-1, keepdim=True)
    noise = torch.randn(*chunks.shape[:-1], mask_length)*chunks_std + chunks_mean
    
    if (noise.max() > 1 or noise.min() < 0) and noise.max() - noise.min() != 0:
            noise = (noise - noise.min())/(noise.max() - noise.min()) 
            
    if len(chunks.shape) == 3:
        for chunk, noise_chunk, chunk_mask_idxs in zip(chunks, noise, mask_idxs):
            for ch, noise_ch, ch_mask_idx in zip(chunk, noise_chunk, chunk_mask_idxs):
                ch[ch_mask_idx:ch_mask_idx + mask_length] = noise_ch
                
    elif len(chunks.shape) == 4:
        for chunk, noise_chunk, chunk_mask_idxs in zip(chunks, noise, mask_idxs):
            for ch_d, noise_ch_d, ch_d_mask_idxs in zip(chunk, noise_chunk, chunk_mask_idxs):
                for ch, noise_ch, ch_mask_idx in zip(ch_d, noise_ch_d, ch_d_mask_idxs):
                    ch[ch_mask_idx:ch_mask_idx + mask_length] = noise_ch
                    
    else:
        raise ValueError(f"Unexpected number of dimensions in chunks: {len(chunks.shape)}")
                
    return chunks

def train_eval(
    dataloader,
    model,
    device="cuda",
    mode="train", #train or something else
    optimizer=None,
    test_dataset=None,
    targets_test=None,
    
    step_max=None,
    
    check_period=None,
    plot_period=None,
    epoch=0,
    check_steps=None,
    plot_steps=None,
    buffer_max=10,
    logger=None,
    
    ml_model=None,
    ml_param_grid=None, 
    ml_eval_function=None,
    ml_eval_function_kwargs=None,
    ml_eval_function_tag=None,
    ml_metric_prefix=None,
    plot_type="regression", #"regression"/"classification"
    
    loss_coefs=None,
    loss_reduction="mean",
    
    is_mask=False,
    mask_ratio=0,

    avg_embeddings_over_time=False,
    verbose=0,
):  
    if mode == "train":
        model.train()
        if verbose - 1 > 0: print("Model is in train mode")
    else:
        model.eval()
        if verbose - 1 > 0: print("Model is in evaluation mode")

    try:
        logger.reset()
        buffer_cnt = 0
        values_keys = []

        start_epoch_time = time.time()
        for step, imgs in enumerate(dataloader):
            #masking
            if is_mask: imgs_masked = mask_chunks(imgs, mask_ratio=mask_ratio)
            
            #results
            if check_instance(model, [VAE, BetaVAE_H, BetaVAE_B, AE_framework]):
                if is_mask: results = model(imgs_masked.to(device))
                else: results = model(imgs.to(device))
                outputs = results["decoded_imgs"]
            else:
                if is_mask: outputs = model(imgs_masked.to(device))
                else: outputs = model(imgs.to(device))
    
            #Loss parts computation
            sum_dims = tuple(range(1, len(imgs.shape)))
            reduce_func = getattr(torch, loss_reduction)
            
            #amplitude
            loss_ampl = reduce_func((outputs - imgs.to(device))**2, sum_dims).mean()
    
            #velocity
            outputs_vel = (outputs[:, :, :-1] - outputs[:, :, 1:])
            imgs_vel = (imgs.to(device)[:, :, :-1] - imgs.to(device)[:, :, 1:])
            loss_vel = reduce_func((outputs_vel - imgs_vel)**2, sum_dims).mean()
    
            #acceleration
            outputs_acc = (outputs_vel[:, :, :-1] - outputs_vel[:, :, 1:])
            imgs_acc = (imgs_vel[:, :, :-1] - imgs_vel[:, :, 1:])
            loss_acc = reduce_func((outputs_acc - imgs_acc)**2, sum_dims).mean()
    
            #frequency
            Foutputs = torch.fft.fft(outputs)
            Foutputs_abs = torch.nn.functional.normalize(torch.abs(outputs), dim=sum_dims[-1])
            Fimgs = torch.fft.fft(imgs.to(device))
            Fimgs_abs = torch.nn.functional.normalize(torch.abs(Fimgs), dim=sum_dims[-1])
            loss_frq = reduce_func((Foutputs_abs - Fimgs_abs)**2, sum_dims).mean()
    
            #summation
            if check_instance(model, [VAE, BetaVAE_H, BetaVAE_B]):
                loss_mean = (
                    loss_coefs["ampl"]*loss_ampl + 
                    loss_coefs["vel"]*loss_vel + 
                    loss_coefs["acc"]*loss_acc + 
                    loss_coefs["frq"]*loss_frq +
                    loss_coefs["kl"]*results['-log p(x|z)']
                )/np.sum(list(loss_coefs.values()))
            else:
                loss_mean = (
                    loss_coefs["ampl"]*loss_ampl + 
                    loss_coefs["vel"]*loss_vel + 
                    loss_coefs["acc"]*loss_acc + 
                    loss_coefs["frq"]*loss_frq
                )/(loss_coefs["ampl"] + loss_coefs["vel"] + loss_coefs["acc"] + loss_coefs["frq"]) #now you can use another keys along with these ones
            
            if isinstance(model, BetaVAE_B): model.update_global_iter(n=1)

            #learning step
            if mode == "train":
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()
                
            #metrics
            decoded_imgs = outputs.cpu()
            
            ##pearson correlation
            corr_running = 0
            for img, decoded_img in zip(imgs, decoded_imgs): #s * 1 * c * t
                for ch_img, ch_d_img in zip(img.squeeze(), decoded_img.squeeze()):
                    corr_coef = torch.corrcoef(torch.stack([ch_img, ch_d_img]))[0][1]
                    corr_running += corr_coef
            corr_avg = corr_running/torch.prod(torch.tensor(imgs.shape[:-1]))

            ##metric
            err = reduce_func((imgs - decoded_imgs)**2, sum_dims)
            if loss_reduction == "mean": max_diff_norm = 1
            else: max_diff_norm = torch.prod(torch.tensor(imgs.shape[1:]))
            metric = 1 - err / max_diff_norm #in [0, 1]
            
            ##snr
            noise = imgs - decoded_imgs
            signal_power = torch.linalg.vector_norm(imgs, dim=len(imgs.shape)-1)**2
            noise_power = torch.linalg.vector_norm(noise, dim=len(noise.shape)-1)**2
            snr_db = 10*torch.log10(signal_power/noise_power).mean()
            
            #logging 'per step' values
            logger.update({
                'loss': loss_mean.cpu(),
                'metric': metric.mean(),
                'pearson_correlation': corr_avg,
                'snr_db': snr_db,
                'loss_ampl': loss_ampl.cpu(),
                'loss_vel': loss_vel.cpu(),
                'loss_acc': loss_acc.cpu(),
                'loss_frq': loss_frq.cpu(),
                "RMSE" : torch.sqrt(loss_ampl.cpu()),
            })
            if check_instance(model, [VAE, BetaVAE_H, BetaVAE_B]):
                for key in ['-log p(x|z)', "kl"]: logger._append(key, results[key])

            #embeddings
            def get_embeddings(
                model, 
                test_dataset, 
                targets_test, 
                avg_over_time=False
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
                    print("Test dataset sample:", test_dataset[nan_indexes[0]], "Embedding:", embeddings_test[nan_indexes[0]])
                    raise ValueError("NaN in embeddings")
                if np.any(np.isnan(y)): 
                    raise ValueError("NaN in targets")

                indices = list(range(len(X)))
                np.random.shuffle(indices)
                return X[indices], y[indices]

            #plotting
            if (plot_period is not None and step % plot_period == 0) or\
            (plot_steps is not None and step in plot_steps):
                start_plotting_time = time.time()
                
                if buffer_cnt >= buffer_max:
                    clear_output(wait=True)
                    buffer_cnt = 0
                buffer_cnt += 1

                print(f"Epoch {epoch}, step {step}")

                #pca embeddnigs
                if verbose - 1 > 0: print("Plotting PCA...")
                X, y = get_embeddings(model, test_dataset, targets_test, avg_over_time=avg_embeddings_over_time)
                fig, ax = plt.subplots(1, 2, figsize=(12, 3))
                plotData(X, y, method="pca", ax=ax[0], plot_type=plot_type)
                
                #plot reconstruction
                if verbose - 1 > 0: print("Plotting reconstruction...")
                ax[1].plot(imgs[0].squeeze()[0], label="data", color="b", marker="o")
                if mode == "train": model.eval()
                imgs_reconstructed = model.reconstruct(imgs.to(device))
                if mode == "train": model.train()
                ax[1].plot(imgs_reconstructed[0].squeeze()[0].detach().cpu(), label="approximation", color="r")
                plt.show()

                end_plotting_time = time.time()
                if verbose - 2 > 0: print(f"Plotting time: {end_plotting_time - start_plotting_time} s")

            #classifier/regressor metrics evaluation
            if (check_period is not None and step % check_period == 0) or\
            (check_steps is not None and step in check_steps):
                start_evaluation_time = time.time()
                
                if ml_model is None or ml_param_grid is None or ml_eval_function is None: raise ValueError("Some ml parameter is not defined")

                if verbose - 1 > 0: print("Classifier/regressor metrics evaluation...")
                X, y = get_embeddings(model, test_dataset, targets_test, avg_over_time=avg_embeddings_over_time)
                if verbose - 2 > 0: print("Embeddings shape:", X.shape)
                results = {}
                for func, kwargs, tag in zip(ml_eval_function, ml_eval_function_kwargs, ml_eval_function_tag):
                    results[tag] = (func(X, y, ml_model, ml_param_grid, **kwargs))
                logger.update_other({ml_metric_prefix: results})

                end_evaluation_time = time.time()
                if verbose - 2 > 0: print(f"Evaluation time: {end_evaluation_time - start_evaluation_time} s")

            #break
            if step_max is not None and step >= step_max:
                break

        end_epoch_time = time.time()
        if verbose - 2 > 0: print(f"Epoch time: {end_epoch_time - start_epoch_time} s")
        
        #logging final results
        logger.average()
        if optimizer is not None: logger._append("lr", optimizer.param_groups[0]['lr'])
        if isinstance(model, BetaVAE_B): logger._append("C", model.get_C())
        logger.log(mode, epoch)
        
        return model, logger.get()

    except KeyboardInterrupt:
        return model, {}