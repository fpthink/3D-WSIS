# Copyright (c) Gorilla-Lab. All rights reserved.
from types import ModuleType
from typing import Dict, Type
import torch.nn as nn

NAME_MAP = {
    # "BN":     "BatchNorm2d",
    "BN1d": "BatchNorm1d",
    "BN2d": "BatchNorm2d",
    "BN3d": "BatchNorm3d",
    "SyncBN": "SyncBatchNorm",
    "GN": "GroupNorm",
    "LN": "LayerNorm",
    "IN": "InstanceNorm2d",
    "IN1d": "InstanceNorm1d",
    "IN2d": "InstanceNorm2d",
    "IN3d": "InstanceNorm3d",
    "deconv": "ConvTranspose2d"
}
# add torch.nn's module name into NAME_MAP
for module_name in dir(nn):
    NAME_MAP[module_name] = module_name


def update_args(args: Dict, **kwargs) -> Dict:
    if kwargs is not None:
        for name, value in kwargs.items():
            args.setdefault(name, value)
    return args


def get_torch_layer_caller(type_name: str) -> Type:
    layer_type = NAME_MAP[type_name] if type_name in NAME_MAP else type_name
    return getattr(nn, layer_type)


def build_from_package(type_name: str,
                      args: Dict,
                      pack: ModuleType=nn,
                      **kwargs) -> object:
    layer_type = NAME_MAP[type_name] if type_name in NAME_MAP else type_name
    try:
        layer = getattr(pack, layer_type)
        args = update_args(args, **kwargs)
        return layer(**args)
    except:
        raise ImportError(f"Unrecognized `layer_type: {layer_type}`")
