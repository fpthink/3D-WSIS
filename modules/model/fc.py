# Copyright (c) Gorilla-Lab. All rights reserved.
from copy import deepcopy
from typing import List, Dict, Union, Callable, Optional

import torch
import torch.nn as nn

from layer_builder import get_torch_layer_caller
from weight_init import *  # constant_init, kaiming_init and so on

class GorillaFC(nn.Sequential):
    r"""A FC block that bundles FC/norm/activation layers.

    This block simplifies the usage of fully connect layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon build method: "get_torch_layer_caller"

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the fully connect layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_features (int): Same as nn.Conv2d.
        out_features (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        FC_cfg (dict): Config dict for fully connect layer. Default: None
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type="ReLU").
        order (tuple[str]): The order of FC/norm/activation layers. It is a
            sequence of "FC", "norm" and "act". Common examples are
            ("FC", "norm", "act") and ("act", "FC", "norm").
            Default: ("FC", "norm", "act").
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool=True,
                 name: str="",
                 norm_cfg: Optional[Dict]=dict(type="BN1d"),
                 act_cfg: Optional[Dict]=dict(type="ReLU", inplace=True),
                 dropout: Optional[float]=None,
                 init: Union[str, Callable]="kaiming",
                 order: List[str]=["FC", "norm", "act", "dropout"]):
        super().__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)

        assert set(order).difference(set(["FC", "norm", "act", "dropout"])) == set()

        self.order = deepcopy(order)
        self.norm_cfg = deepcopy(norm_cfg)
        self.act_cfg = deepcopy(act_cfg)

        # if the FC layer is before a norm layer, bias is unnecessary.\
        with_norm = (self.norm_cfg is not None)
        # if with_norm:
        #     bias = False

        # build FC layer
        FC = nn.Linear(in_features, out_features, bias)

        # build normalization layers
        norm = None
        if with_norm:
            if "BN" in self.norm_cfg.get("type"):
                if self.order.index("norm") > self.order.index("FC"):
                    num_features = FC.out_features
                else:
                    num_features = FC.in_features
                self.norm_cfg.update(num_features=num_features)
            norm_caller = get_torch_layer_caller(self.norm_cfg.pop("type"))
            norm = norm_caller(**self.norm_cfg)
        else:
            if "norm" in self.order:
                self.order.remove("norm")

        # init FC and norm
        if init is not None:
            if isinstance(init, str):
                init_func = globals()[f"{init}_init"]
            elif isinstance(init, Callable):
                init_func = init
            init_func(FC)
        if "norm" in self.order:
            constant_init(norm, 1, bias=0)

        # build activation layer
        with_act = (self.act_cfg is not None)
        act = None
        if with_act:
            act_caller = get_torch_layer_caller(self.act_cfg.pop("type"))
            act = act_caller(**self.act_cfg)
        else:
            if "act" in self.order:
                self.order.remove("act")

        # build dropout layer
        with_dropout = (dropout is not None)
        if with_dropout:
            dropout = nn.Dropout(p=dropout)
        else:
            if "dropout" in self.order:
                self.order.remove("dropout")

        for layer in self.order:
            self.add_module(name + layer, eval(layer))


class MultiFC(nn.Sequential):
    def __init__(
            self,
            nodes: List[int],
            bias: Union[List[bool], bool]=True,
            name: Union[List[str], str]="",
            norm_cfg: Optional[Union[List[Dict], Dict]]=dict(type="BN1d"),
            act_cfg: Optional[Union[List[Dict], Dict]]=dict(type="ReLU", inplace=True),
            dropout: Optional[Union[List[float], float]]=None,
            init: Union[str, Callable]="kaiming",
            order: List[str]=["FC", "norm", "act", "dropout"],
            drop_last: bool=True):
        r"""Author: liang.zhihao
        Build the multi FC layer easily

        Args:
            nodes (List[int]): The num of nodes of each layer (including input layer and output layer).
            bias (Union[List[bool], bool], optional): With bias or not. Defaults to True. Refer to GoirllaFC.
            name (Union[List[str], str], optional): Name of each FC. Defaults to "". Refer to GoirllaFC.
            norm_cfg (Optional[Union[List[Dict], Dict]], optional): Norm cfg of each FC. Defaults to dict(type="BN1d"). Refer to GoirllaFC.
            act_cfg (Optional[Union[List[Dict], Dict]], optional): Activation cfg of each FC. Defaults to dict(type="ReLU", inplace=True). Refer to GoirllaFC.
            dropout (Optional[Union[List[float], float]], optional): Dropout ratio of each FC. Defaults to None. Refer to GoirllaFC.
            init (Union[str, Callable], optional): Init func or init_func name. Defaults to "kaiming".
            order (List[str], optional): FC layer order. Defaults to ["FC", "norm", "act", "dropout"].
            drop_last (bool, optional): Drop out last layer's norm and activation or not. Defaults to True.

        Returns:
            [type]: [description]
        """
        super().__init__()
        num_of_linears = len(nodes) - 1

        def list_wrapper(x):
            if isinstance(x, List):
                assert len(x) == num_of_linears
                return x
            else:
                return [x] * num_of_linears

        bias_list = list_wrapper(bias)
        name_list = list_wrapper(name)
        act_cfg_list = list_wrapper(act_cfg)
        norm_cfg_list = list_wrapper(norm_cfg)
        dropout_list = list_wrapper(dropout)

        # if drop last, remove the last layer's activation and norm
        if drop_last:
            act_cfg_list[-1] = None
            norm_cfg_list[-1] = None

        for idx, (in_features, out_features, b, n, norm, act, drop) in \
            enumerate(zip(nodes[:-1], nodes[1:], bias_list, name_list, norm_cfg_list, act_cfg_list, dropout_list)):
            self.add_module(
                str(idx),
                GorillaFC(
                    in_features=in_features,
                    out_features=out_features,
                    bias=b,
                    name=n,
                    norm_cfg=norm,
                    act_cfg=act,
                    dropout=drop,
                    init=init,
                    order=order))


class DenseFC(nn.Module):
    def __init__(
            self,
            nodes: List[int],
            arc_tale: List[List[int]],
            arc_tm_shape: List[List[int]],
            init: Union[str, Callable]="geometric"):
        r"""Author: lei.jiabao, liang.zhihao
        initialization for MLP of arbitrary architecture (specialized for cuam library)

        Args:
            nodes (list): The num of nodes of each layer (including input layer and output layer).
                For example, nodes = [3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1].

            arc_tale (list): A list of list which specifies the architecture of MLP. 
                For example, arc_tale = [[0], [1, 1, 0], [0], [1, 3, 0], [0], [1, 5, 0], [0], [1, 7, 0], [0], [0]].
                Note: 
                    len(arc_tale) should be equal to the num of hidden layers, i.e. len(nodes)-2

                    'i' refer to arc_table id in the under example
                    arc_tale[i][0] should be the number of incoming adding skip connection (0 <= i < len(arc_tale))
                    len(arc_tale[i]) should be of length 1+2*arc_tale[i][0]
                    arc_tale[i][1+2*j] should be the index of the source of the adding skip connection (0 <= j < arc_tale[i][0])
                        refer to source id in the under example
                    arc_tale[i][2+2*j] should be the index of the transformation matrix (0 <= j < arc_tale[i][0])

                Example: [[0], [1, 1, 0], [0], [1, 3, 0], [0], [1, 5, 0], [0], [1, 7, 0], [0], [0]]

                         ┌───────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
                         |               ↓|              ↓|              ↓|              ↓
                [in] -> [h0] -> [h1] -> [h2] -> [h3] -> [h4] -> [h5] -> [h6] -> [h7] -> [h8] -> [h9] -> [out]
     source_id:  0       1       2       3       4       5       6       7       8       9       10      11
  arc_table_id:                  0       1       2       3       4       5       6       7       8       9

                        [[0], [1, 1, 0], [0], [2, 3, 0, 1, 0], [0], [2, 5, 0, 1, 0], [0], [1, 7, 0], [0], [0]]


                         ┌───────────────────────────────────────────────┐
                         ├───────────────────────────────┐               |
                         ├───────────────┐┌──────────────┤┌──────────────┤┌──────────────┐
                         |               ↓|              ↓|              ↓|              ↓
                [in] -> [h0] -> [h1] -> [h2] -> [h3] -> [h4] -> [h5] -> [h6] -> [h7] -> [h8] -> [h9] -> [out]
     source_id:  0       1       2       3       4       5       6       7       8       9       10      11
  arc_table_id:                  0       1       2       3       4       5       6       7       8       9

                TIP: [.] represent the layers' features, 'in' and 'out' mean input and output 'h{num}' means the hidden layer
                     '-' means the fc layer and '>' means the forward direction

            arc_tm_shape (list): list of the shape of transformation matrices. 
                For example, arc_tm_shape = [[0, 0]].
                Note:
                    if it is an identity matrix, please use shape==[0, 0] for performance optimization
                    
            init (str, optional): choose the method to initialize the parameters. options are "kaiming" or "geometric".
                Defaults to "geometric".
                Note:
                    "kaiming" is the well-known kaiming initialization strategy
                    "geometric" is Geometric network initialization strategy described in 
                        paper `SAL: Sign Agnostic Learning of Shapes from Raw Data`. 
                        It is suitable for fitting a Signed Distance Field from raw point cloud (without normals or any sign).
        
        Note:
            the arc_tm_shape and the transform idx in arc_tale should be correpsond
        """
        super().__init__()
        self.nodes = nodes
        self.arc_tale = arc_tale
        self.arc_tm_shape = arc_tm_shape

        # build linears
        self.num_of_linears = len(self.nodes) - 1
        self.linears = torch.nn.ModuleList()
        for linear_idx in range(self.num_of_linears):
            # init the linear linear
            linear = torch.nn.Linear(self.nodes[linear_idx], self.nodes[linear_idx + 1], bias=True)
            # get the related init function and init
            last = (linear_idx == self.num_of_linears)
            if isinstance(init, str):
                init_func = globals()[f"{init}_init"]
            elif isinstance(init, Callable):
                init_func = init
            else:
                raise TypeError(f"init must be 'str' or 'Callable', but got {type(init)}")
            init_func(linear, last)
            self.linears.append(linear)

        # activation
        self.num_of_acts = self.num_of_linears
        self.acts = torch.nn.ModuleList()
        for i in range(self.num_of_acts):
            if i != self.num_of_acts - 1:
                self.acts.append(torch.nn.ReLU(inplace=True))
            else:
                self.acts.append(torch.nn.Identity())


        # transform matrix
        self.num_of_tms = len(self.arc_tm_shape)
        self.tms = torch.nn.ModuleList()
        for tm_shape in self.arc_tm_shape:
            if tm_shape[0] == 0 and tm_shape[1] == 0:
                self.tms.append(torch.nn.Identity())
            else:
                self.tms.append(torch.nn.Linear(in_features=tm_shape[1], out_features=tm_shape[0], bias=False))

        # source of transform
        self.srcs = {}
        for arc in self.arc_tale:
            for i in range(arc[0]):
                self.srcs[arc[1 + 2 * i]] = None

        self.outputs_list = []

    def forward(self,
                x: torch.Tensor,
                requires_outputs_list: bool=False) -> torch.Tensor:
        if requires_outputs_list:
            self.outputs_list.clear()

        for linear_idx in range(self.num_of_linears):
            # record the layer features
            if linear_idx in self.srcs.keys():
                self.srcs[linear_idx] = x

            # linear forward
            x = self.linears[linear_idx](x)

            if linear_idx >= 1:
                skip_arc = self.arc_tale[linear_idx - 1]
                for j in range(skip_arc[0]):
                    # get the former feature map
                    src = self.srcs[skip_arc[1 + 2 * j]]
                    # transformer forward
                    transform = self.tms[skip_arc[2 + 2 * j]]
                    # skip add
                    x = x + transform(src)

            x = self.acts[linear_idx](x)

            # record output
            if linear_idx != self.num_of_linears - 1:
                if requires_outputs_list:
                    self.outputs_list.append(x)

        return x

    def get_info(self) -> Dict:

        weights = []
        for i in range(self.num_of_linears):
            weights.append(self.linears[i].weight)

        biases = []
        for i in range(self.num_of_linears):
            biases.append(self.linears[i].bias)

        arc_tm = []
        for i in range(self.num_of_tms):
            tms = self.tms[i]
            if isinstance(tms, torch.nn.Identity):
                arc_tm.append(torch.zeros([0, 0]))
            elif isinstance(tms, torch.nn.Linear):
                arc_tm.append(tms.weight)

        arc_table_width = max([len(arc) for arc in self.arc_tale])
        arc_table = torch.zeros([len(self.arc_tale), arc_table_width], dtype=torch.int32)
        for r in range(len(self.arc_tale)):
            for c in range(len(self.arc_tale[r])):
                arc_table[r, c] = self.arc_tale[r][c]

        return {"weights": weights, "biases": biases, "arc_tm": arc_tm, "arc_table": arc_table}
