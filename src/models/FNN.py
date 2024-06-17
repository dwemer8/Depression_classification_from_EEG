'''
Reimplementation of architecture from "Improving Multichannel Raw Electroencephalography-based Diagnosis of Major Depressive Disorder via Transfer Learning with Single Channel Sleep Stage Data"
'''

import torch
import torch.nn as nn
import typing as t
from . import modules as modules

def get_module(module_name):
    try:
        return getattr(nn, module_name)
    except:
        pass

    try:
        return getattr(modules, module_name)
    except:
        pass

    raise ValueError(f"Unknown module: {module_name}")

class FNN(nn.Module):
    def __init__(
        self,
        modules_params : t.List[t.Tuple[str, t.Dict]] = [
            ("Conv1d", {"in_channels": 3, "out_channels" : 5, "kernel_size" : 10, "stride": 1}),
            ("ELU", {}),
            ("MaxPool1d", {"kernel_size": 2, "stride": 2}),
            ("BatchNorm1d", {"num_features": 5}),

            ("Conv1d", {"in_channels": 5, "out_channels" : 10, "kernel_size" : 10, "stride": 1}),
            ("ELU", {}),
            ("MaxPool1d", {"kernel_size": 2, "stride": 2}),
            ("BatchNorm1d", {"num_features": 10}),

            ("Conv1d", {"in_channels": 10, "out_channels" : 10, "kernel_size" : 10, "stride": 1}),
            ("ELU", {}),
            ("MaxPool1d", {"kernel_size": 2, "stride": 2}),
            ("BatchNorm1d", {"num_features": 10}),

            ("Conv1d", {"in_channels": 10, "out_channels" : 15, "kernel_size" : 5, "stride": 1}),
            ("ELU", {}),
            ("MaxPool1d", {"kernel_size": 2, "stride": 2}),
            ("BatchNorm1d", {"num_features": 15}),

            ("Dropout", {"p": 0.5}),
            ("FlattenLazyLinear", {"out_features": 64}),
            ("ELU", {}),

            ("Dropout", {"p": 0.5}),
            ("LazyLinear", {"out_features": 32}),
            ("ELU", {}),

            ("Dropout", {"p": 0.5}),
            ("LazyLinear", {"out_features": 2}),

            ("Softmax", {"dim" : 1})
        ]
    ):
        super().__init__()
        modules = []

        for layer, params in modules_params:
            modules.append(get_module(layer)(**params))

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        return self.net(x)