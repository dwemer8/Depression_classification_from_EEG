import os
import wandb
import torch

from .modules import encoder_conv, decoder_conv, encoder_conv4, decoder_conv4
from .VAE import VAE, BetaVAE_H, BetaVAE_B
from .AE import AE, AE_framework
from .UNet import UNet

def get_model(model_config):
    model, config = getattr(ModelsZoo, "get_" + model_config["model"])(model_config)
    return model, config

def load_weights_from_wandb(model, artifact, file, verbose=1):
    if verbose > 0:
        weights_file = os.path.join(artifact, file)
        print(f"Loading weights from {weights_file}...")
    run = wandb.init()
    artifact = run.use_artifact(artifact, type='model')
    artifact_dir = artifact.download()
    model.load_state_dict(torch.load(os.path.join(artifact_dir, file)))
    wandb.finish()
    if verbose > 0: 
        print("Weights were loaded")
    return model

# models #######################################################################

##################
# shallow networks
##################

class ModelsZoo:
    def get_VAE(config=None):
        if config is None: config = {}
        model_config = {
            "latent_dim": 10,
            "beta": 2,
            "input_dim": (1, 3, 128),
            "latent_std": 1,
            "latent_multiplier" : 2,
            "loss_reduction" : "mean", #"mean"/"sum,
            "model_description": "beta-VAE, 3 ch., 32/32, 6/6, PReLU",
        }
        model_config.update(config)
        model = VAE(
            encoder_conv(model_config),
            decoder_conv(model_config),
            **model_config
        )
        return model, model_config
    
    def get_BetaVAE_B(config=None):
        if config is None: config = {}
        model_config = {
            "latent_dim": 10,
            "beta": 100,
            "input_dim": (1, 3, 128),
            "latent_std": 1,
            "C_max": 100,
            "C_stop_iter": len(train_dataloader)*50,
            "device": device,
            "latent_multiplier" : 2,
            "loss_reduction" : "mean", #"mean"/"sum
            "model_description": "0 train, inhouse, beta-VAE-B, 3 ch., 32/32, 6/6, PReLU",
        }
        model_config.update(config)
        model = BetaVAE_B(
            encoder_conv(model_config),
            decoder_conv(model_config),
            **model_config
        )
        return model, model_config
    
    def get_BetaVAE_H(config=None):
        if config is None: config = {}
        model_config = {
            "latent_dim": 10,
            "beta": 2,
            "input_dim": (1, 3, 128),
            "latent_std": 1,
            "latent_multiplier" : 2,
            "loss_reduction" : "mean", #"mean"/"sum
            "model_description": "inhouse, beta-VAE-H, 3 ch., 32/32, 6/6, PReLU",
        }
        model_config.update(config)
        model = BetaVAE_H(
            encoder_conv(model_config),
            decoder_conv(model_config),
            **model_config
        )
        return model, model_config
    
    def get_AE_framework(config):
        if config is None: config = {}
        model_config = {
            "latent_dim": 10,
            "input_dim": (1, 3, 128),
            "latent_multiplier": 1,
            "loss_reduction" : "mean", #"mean"/"sum
            "model_description": "AE, 3 ch., 32/32, 6/6, PReLU",
        }
        model_config.update(config)
        model = AE_framework(
            encoder_conv(model_config),
            decoder_conv(model_config),
            **model_config
        )
        return model, model_config
    
    ###############
    # deep networks
    ###############

    def get_VAE_deep(config):
        if config is None: config = {}
        model_config = {
            "latent_dim": 15*32,
            "beta": 2,
            "input_dim": (3, 128),
            "latent_std": 1,
            "n_channels": 3,
            "n_classes": 3,
            "first_decoder_conv_depth": 32,
            "loss_reduction" : "mean", #"mean"/"sum
            "model_description": "depr. anon., beta-VAE, 3 ch., 4/8/16/32, 7/7/5/3/3/3/3/1, Sigmoid",
        }
        model_config.update(config)
        model = VAE(
            encoder_conv4(model_config["n_channels"]),
            decoder_conv4(model_config["n_classes"]),
            **model_config
        )
        return model, model_config
    
    def get_AE(config):
        if config is None: config = {}
        model_config = {
            "input_dim": (3, 128),
            "loss_reduction" : "mean", #"mean"/"sum,
            'model_description': "depr. anon., AE, 3 ch., 4/8/16/32, 7/7/5/3/3/3/3/1, Sigmoid",
            "n_channels" : 3,
            "n_classes": 3,
        }
        model_config.update(config)
        model = AE(
            n_channels=model_config["n_channels"],
            n_classes=model_config["n_classes"],
        )
        return model, model_config
    
    def get_UNet(config):
        if config is None: config = {}
        model_config = {
            "input_dim": (3, 128),
            "c1": 3,
            "c2": 3,
            "c3": 2,
            "c_neck": 1,
            "loss_reduction" : "mean", #"mean"/"sum
            "model_description": "UNet, 3 ch., 3/3/2/1, 7/7/5/3/3/3/3/1, PReLU",
            "n_channels" : 3,
            "n_classes": 3,
        }
        model_config.update(config)
        model = UNet(
            n_channels=model_config["n_channels"],
            n_classes=model_config["n_classes"],
            c1=model_config["c1"],
            c2=model_config["c2"],
            c3=model_config["c3"],
            c_neck=model_config["c_neck"],
        )
        return model, model_config