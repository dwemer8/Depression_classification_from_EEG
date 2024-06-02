'''
Reimplementation of encoder and decoder from "REPRESENTATION LEARNING FOR IMPROVED INTERPRETABILITY AND CLASSIFICATION ACCURACY OF CLINICAL FACTORS FROM EEG"
'''

import torch
import torch.nn as nn
import typing as t
from .modules import FlattenLinear, UnflattenLinear

class Encoder(nn.Module):
    def __init__(
        self,
        convs_params : t.List[t.Dict] = [
            {"in_channels": 6, "out_channels": 32, "kernel_size": 6, "stride": 2},
            {"in_channels": 32, "out_channels": 6, "kernel_size": 6, "stride": 2},
        ],
        activation : str = "ReLU",
        linear_params : t.List[t.Dict] = [
            {"in_features" : 366, "out_features": 128},
            {"in_features" : 128, "out_features": 10},
        ]
    ):
        super().__init__()
        
        modules = []

        for i, params in enumerate(convs_params):
            if i != 0:
                modules.extend([
                    getattr(nn, activation)(),
                    nn.Conv1d(**params),
                ])
            else:
                modules.append(nn.Conv1d(**params))

        for i, params in enumerate(linear_params):
            if i != 0:
                modules.extend([
                    getattr(nn, activation)(),
                    nn.Linear(**params),
                ])
            else:
                modules.extend([
                    getattr(nn, activation)(),
                    FlattenLinear(**params),
                ])

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
class Decoder(nn.Module):
    def __init__(
        self,
        activation : str = "ReLU",
        linear_params : t.List[t.Dict] = [
            {"in_features" : 10, "out_features": 128},
            {"in_features" : 128, "out_features": 366, "penultimate_dim": 6},
        ],
        convs_params : t.List[t.Dict] = [
            {"in_channels": 6, "out_channels": 32, "kernel_size": 6, "stride": 2},
            {"in_channels": 32, "out_channels": 6, "kernel_size": 6, "stride": 2},
        ],
    ):
        super().__init__()
        
        modules = []

        for i, params in enumerate(linear_params):
            if i != len(linear_params) - 1:
                modules.extend([
                    nn.Linear(**params),
                    getattr(nn, activation)(),
                ])
            else:
                modules.append(UnflattenLinear(**params))
                if len(convs_params) > 0:
                    modules.append(getattr(nn, activation)())

        for i, params in enumerate(convs_params):
            if i != len(convs_params) - 1:
                modules.extend([
                    nn.ConvTranspose1d(**params),
                    getattr(nn, activation)(),
                ])
            else:
                modules.append(nn.ConvTranspose1d(**params))

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        return self.net(x)