import os
import wandb
import torch

from .modules import \
    encoder_conv, \
    decoder_conv, \
    encoder_conv_bVAE, \
    decoder_conv_bVAE, \
    ConvEncoder, \
    ConvDecoder, \
    TransformerEncoder, \
    TransformerDecoder
from .VAE import VAE, BetaVAE_H, BetaVAE_B
from .AE import AE, AE_framework
from .UNet import UNet
from src.utils.common import upd

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
            #for framework
            "latent_dim": 16*32,
            "beta": 2,
            "first_decoder_conv_depth": 32,
            "loss_reduction" : "mean", #"mean"/"sum

            #for encoder end decoder
            "n_channels": 3,
            "n_classes": 3,

            #for log only
            "latent_std": 1,
            "model_description": "beta-VAE, 3 ch., 4/8/16/32, 7/7/5/3/3/3/3/1, Sigmoid",
        }
        model_config.update(config)
        model = VAE(
            encoder_conv_bVAE(model_config["n_channels"]),
            decoder_conv_bVAE(model_config["n_classes"]),
            **model_config
        )
        return model, model_config

    def get_VAE_parametrized(config):
        if config is None: config = {}
        framework_config = {
            "latent_dim": 16*32,
            "beta": 2,
            "first_decoder_conv_depth": 32,
            "loss_reduction" : "mean", #"mean"/"sum
        }
        encoder_config = {
            "down_blocks_config": [
                {"in_channels": 3, "out_channels": 4, "kernel_size": 7, "n_convs": 2, "activation": "Sigmoid"},
                {"in_channels": 4, "out_channels": 8, "kernel_size": 7, "n_convs": 2, "activation": "Sigmoid"},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 5, "n_convs": 2, "activation": "Sigmoid"},
            ],
            "out_conv_config": {"in_channels": 16, "out_channels": 64, "kernel_size": 3, "n_convs": 2, "activation": "Sigmoid", "normalize_last": False},
        }
        decoder_config = {
            "in_conv_config": {"in_channels": 32, "out_channels": 16, "kernel_size": 3, "n_convs": 2, "activation": "Sigmoid"},
            "up_blocks_config": [
                {"in_channels": 16, "out_channels": 8, "kernel_size": 3, "n_convs": 2, "activation": "Sigmoid"},
                {"in_channels": 8, "out_channels": 4, "kernel_size": 3, "n_convs": 2, "activation": "Sigmoid"},
                {"in_channels": 4, "out_channels": 3, "kernel_size": 1, "n_convs": 2, "activation": "Sigmoid", "normalize_last": False},
            ],
        }
        model_config = {
            "framework": framework_config,
            "encoder": encoder_config,
            "decoder": decoder_config,
            
            "model_description": "beta-VAE",
            "model": "VAE_parametrized",
            "loss_reduction" : framework_config["loss_reduction"],  #for compatibility with train function
        }
        model_config = upd(model_config, config)
        model = VAE(
            ConvEncoder(**model_config["encoder"]),
            ConvDecoder(**model_config["decoder"]),
            **model_config["framework"],
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

    def get_AE_parametrized(config):
        if config is None: config = {}
        framework_config = {
            "first_decoder_conv_depth": 32,
            "loss_reduction" : "mean", #"mean"/"sum
        }
        encoder_config = {
            "down_blocks_config": [
                {"in_channels": 3, "out_channels": 4, "kernel_size": 7, "n_convs": 2, "activation": "Sigmoid"},
                {"in_channels": 4, "out_channels": 8, "kernel_size": 7, "n_convs": 2, "activation": "Sigmoid"},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 5, "n_convs": 2, "activation": "Sigmoid"},
            ],
            "out_conv_config": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "n_convs": 2, "activation": "Sigmoid", "normalize_last": False},
        }
        decoder_config = {
            "in_conv_config": {"in_channels": 32, "out_channels": 16, "kernel_size": 3, "n_convs": 2, "activation": "Sigmoid"},
            "up_blocks_config": [
                {"in_channels": 16, "out_channels": 8, "kernel_size": 3, "n_convs": 2, "activation": "Sigmoid"},
                {"in_channels": 8, "out_channels": 4, "kernel_size": 3, "n_convs": 2, "activation": "Sigmoid"},
                {"in_channels": 4, "out_channels": 3, "kernel_size": 1, "n_convs": 2, "activation": "Sigmoid", "normalize_last": False},
            ],
        }
        model_config = {
            "framework": framework_config,
            "encoder": encoder_config,
            "decoder": decoder_config,
            
            "model_description": "AE",
            "model": "AE_parametrized",
            "loss_reduction" : framework_config["loss_reduction"],  #for compatibility with train function
        }
        model_config = upd(model_config, config)
        model = AE_framework(
            ConvEncoder(**model_config["encoder"]),
            ConvDecoder(**model_config["decoder"]),
            **model_config["framework"],
        )
        return model, model_config

    def get_transformer_AE_parametrized(config):
        if config is None: config = {}
        framework_config = {
            "loss_reduction" : "mean", #"mean"/"sum
        }
        encoder_config = {
            "down_blocks_config": [
                {"input_dim": 128, "seq_length": 3, "num_heads": 1, "depth": 1, "mlp_depth": 2, "mlp_ratio": 4, "p_dropout":0.1, "act_layer": 'PReLU', "norm_layer": "LayerNorm"},
                {"input_dim": 64, "seq_length": 3, "num_heads": 1, "depth": 1, "mlp_depth": 2, "mlp_ratio": 4, "p_dropout":0.1, "act_layer": 'PReLU', "norm_layer": "LayerNorm"},
                {"input_dim": 32, "seq_length": 3, "num_heads": 1, "depth": 1, "mlp_depth": 2, "mlp_ratio": 4, "p_dropout":0.1, "act_layer": 'PReLU', "norm_layer": "LayerNorm"},
            ],
            "out_block_config": {"dim": 16, "num_heads": 1, "mlp_ratio": 4, "mlp_depth": 2, "p_dropout": 0.1, "act_layer": "PReLU", "norm_layer": "LayerNorm"},
        }
        decoder_config = {
            "in_block_config": {"dim": 16, "num_heads": 1, "mlp_ratio": 4, "mlp_depth": 2, "p_dropout": 0.1, "act_layer": "PReLU", "norm_layer": "LayerNorm"},
            "up_blocks_config": [
                {"input_dim": 16, "seq_length": 3, "num_heads": 1, "depth": 1, "mlp_depth": 2, "mlp_ratio": 4, "p_dropout":0.1, "act_layer": 'PReLU', "norm_layer": "LayerNorm"},
                {"input_dim": 32, "seq_length": 3, "num_heads": 1, "depth": 1, "mlp_depth": 2, "mlp_ratio": 4, "p_dropout":0.1, "act_layer": 'PReLU', "norm_layer": "LayerNorm"},
                {"input_dim": 64, "seq_length": 3, "num_heads": 1, "depth": 1, "mlp_depth": 2, "mlp_ratio": 4, "p_dropout":0.1, "act_layer": 'PReLU', "norm_layer": "LayerNorm", "with_outro": True}
            ],
        }
        model_config = {
            "framework": framework_config,
            "encoder": encoder_config,
            "decoder": decoder_config,
            
            "model_description": "transformer AE",
            "model": "transformer_AE_parametrized",
            "loss_reduction" : framework_config["loss_reduction"],  #for compatibility with train function
        }
        model_config = upd(model_config, config)
        model = AE_framework(
            TransformerEncoder(**model_config["encoder"]),
            TransformerDecoder(**model_config["decoder"]),
            **model_config["framework"],
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