import time
import torch
import numpy as np
import wandb

from src.models.VAE import VAE, BetaVAE_H, BetaVAE_B
from src.models.AE import AE_framework
from src.utils.common import printLog, check_instance
from src.utils.callbacks import mask_chunks
from .helpers import vizualize, eval_ml_model

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
    ml_to_train=None,
    plot_type="classification", #"regression"/"classification"
    
    loss_coefs=None,
    loss_reduction="mean",
    
    is_mask=False,
    mask_ratio=0,

    avg_embeddings_over_time=False,
    verbose=0,
    logfile=None,
    logdir=None,
):  
    if mode == "train":
        model.train()
        if verbose - 1 > 0: printLog("Model is in train mode", logfile=logfile)
    else:
        model.eval()
        if verbose - 1 > 0: printLog("Model is in evaluation mode", logfile=logfile)

    try:
        logger.reset() #drops internal counters related to averaging
        buffer_cnt = 0
        trained_ml_models = None

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
            Foutputs_abs = torch.nn.functional.normalize(torch.abs(Foutputs), dim=sum_dims[-1])
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

            #plotting
            if (plot_period is not None and step % plot_period == 0) or\
            (plot_steps is not None and step in plot_steps):
                vizualize(
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
                    verbose,
                    logdir,
                    logger
                )

            #classifier/regressor metrics evaluation
            #it will be true for every check_period != None when step == 0
            if (check_period is not None and step % check_period == 0) or\
            (check_steps is not None and step in check_steps):
                    trained_ml_models, _ = eval_ml_model(
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
                    ml_metric_prefix,
                    ml_to_train,
                )

            #break
            if step_max is not None and step >= step_max:
                break

        end_epoch_time = time.time()
        if verbose - 2 > 0: printLog(f"Epoch time: {end_epoch_time - start_epoch_time} s", logfile=logfile)
        
        #logging final results
        logger.average()
        if optimizer is not None: logger._append("lr", optimizer.param_groups[0]['lr'])
        if isinstance(model, BetaVAE_B): logger._append("C", model.get_C())
        logger.log(mode, epoch)
        
        results = {k: v for k, v in logger.get().items() if not isinstance(v, wandb.Image)}
        return model, trained_ml_models, results

    except KeyboardInterrupt:
        return model, None, {}