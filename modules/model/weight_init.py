# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np
import torch.nn as nn


def constant_init(module: nn.Module, val, bias=0, **kwargs):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module: nn.Module, gain=1, bias=0, distribution="normal", **kwargs):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module, mean=0, std=1, bias=0, **kwargs):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module: nn.Module, a=0, b=1, bias=0, **kwargs):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module: nn.Module,
                 a=0,
                 mode="fan_out",
                 nonlinearity="relu",
                 bias=0,
                 distribution="normal",
                 **kwargs):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(module.weight,
                                 a=a,
                                 mode=mode,
                                 nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight,
                                a=a,
                                mode=mode,
                                nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def c2_xavier_init(module: nn.Module) -> None:
    r"""
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)  # pyre-ignore
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def c2_msra_init(module: nn.Module, **kwargs) -> None:
    r"""
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (nn.Module): module to initialize.
    """
    # pyre-ignore
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def geometric_init(module: nn.Linear,
                   geometric_radius: int=1.0,
                   last: bool=False,
                   **kwargs) -> None:
    r"""Author: lei.jiabao
    gepometric_init defined from SAL paper, work for MLP

    Args:
        module (nn.Linear): MLP layer
        geometric_radius (int, optional): radius. Defaults to 1.0.
        last (bool, optional): the last MLP layer. Defaults to False.
    """
    # get input and output dim
    in_dim = module.weight.shape[1]
    out_dim = module.weight.shape[0]

    if last: # last layer
        nn.init.constant_(module.weight, np.sqrt(np.pi) / np.sqrt(in_dim))
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, -geometric_radius)
    else: # hidden layer
        nn.init.normal_(module.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def bias_init_with_prob(prior_prob: float):
    r"""initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
