import functools
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

import spconv
from spconv.modules import SparseModule

from layer_builder import get_torch_layer_caller

def conv1x3(in_planes: int,
            out_planes: int,
            stride: int=1,
            indice_key: Optional[str]=None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=(1, 3, 3),
                             stride=stride,
                             padding=(0, 1, 1),
                             bias=False,
                             indice_key=indice_key)


def conv3x1(in_planes: int,
            out_planes: int,
            stride: int=1,
            indice_key: Optional[str]=None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=(3, 1, 3),
                             stride=stride,
                             padding=(1, 0, 1),
                             bias=False,
                             indice_key=indice_key)



class AsymResidualBlock(SparseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_fn: Union[Callable, Dict]=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key: Optional[str]=None,
                 normalize_before: bool=False):
        super().__init__()
        
        if isinstance(norm_fn, Dict):
            # norm_caller = gorilla.nn.get_torch_layer_caller(norm_fn.pop("type"))  # le
            norm_caller = get_torch_layer_caller(norm_fn.pop("type"))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        if normalize_before:
            self.conv_1 = spconv.SparseSequential(
                norm_fn(in_channels),
                nn.LeakyReLU(),
                conv3x1(in_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                conv1x3(out_channels, out_channels, indice_key=indice_key),
                )
            self.conv_2 = spconv.SparseSequential(
                norm_fn(in_channels),
                nn.LeakyReLU(),
                conv1x3(in_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                conv3x1(out_channels, out_channels, indice_key=indice_key),
                )
        else:
            self.conv_1 = spconv.SparseSequential(
                conv3x1(in_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                conv1x3(out_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                )
            self.conv_2 = spconv.SparseSequential(
                conv1x3(in_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                conv3x1(out_channels, out_channels, indice_key=indice_key),
                norm_fn(out_channels),
                nn.LeakyReLU(),
                )
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv_1(input)
        output.features += self.conv_2(input).features

        return output


class ResidualBlock(SparseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_fn: Union[Callable, Dict]=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key: Optional[str]=None,
                 normalize_before: bool=True):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  bias=False))
        
        if isinstance(norm_fn, Dict):
            # norm_caller = gorilla.nn.get_torch_layer_caller(norm_fn.pop("type"))  # le
            norm_caller = get_torch_layer_caller(norm_fn.pop("type"))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels),
                nn.ReLU(),
                spconv.SubMConv3d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
                spconv.SubMConv3d(out_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key))
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
                spconv.SubMConv3d(out_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU())

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features,
                                           input.indices,
                                           input.spatial_shape,
                                           input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_fn: Union[Callable, Dict]=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key: Optional[str]=None,
                 normalize_before: bool=True):
        super().__init__()
        
        if isinstance(norm_fn, Dict):
            # norm_caller = gorilla.nn.get_torch_layer_caller(norm_fn.pop("type"))  # le
            norm_caller = get_torch_layer_caller(norm_fn.pop("type"))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        if normalize_before:
            self.conv_layers = spconv.SparseSequential(
                norm_fn(in_channels),
                nn.ReLU(),
                spconv.SubMConv3d(in_channels,
                                out_channels,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                                indice_key=indice_key))
        else:
            self.conv_layers = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels,
                                out_channels,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                                indice_key=indice_key),
                norm_fn(in_channels),
                nn.ReLU())

    def forward(self, input):
        return self.conv_layers(input)

class UBlock(nn.Module):
    def __init__(self,
                 nPlanes: List[int],
                 norm_fn: Union[Dict, Callable]=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 block_reps: int=2,
                 block: Union[str, Callable]=ResidualBlock,
                 indice_key_id: int=1,
                 normalize_before: bool=True,
                 return_blocks: bool=False,):

        super().__init__()

        self.return_blocks = return_blocks
        self.nPlanes = nPlanes

        # process block and norm_fn caller
        if isinstance(block, str):
            area = ["residual", "vgg", "asym"]
            assert block in area, f"block must be in {area}, but got {block}"
            if block == "residual":
                block = ResidualBlock
            elif block == "vgg":
                block = VGGBlock
            elif block == "asym":
                block = AsymResidualBlock
        
        if isinstance(norm_fn, Dict):
            # norm_caller = gorilla.nn.get_torch_layer_caller(norm_fn.pop("type"))  # le
            norm_caller = get_torch_layer_caller(norm_fn.pop("type"))
            norm_fn = functools.partial(norm_caller, **norm_fn)
            
        blocks = {
            f"block{i}":
            block(nPlanes[0],
                  nPlanes[0],
                  norm_fn,
                  normalize_before=normalize_before,
                  indice_key=f"subm{indice_key_id}")
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(nPlanes[0]),
                    nn.ReLU(),
                    spconv.SparseConv3d(
                        nPlanes[0],
                        nPlanes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}"))
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        nPlanes[0],
                        nPlanes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}"),
                    norm_fn(nPlanes[1]),
                    nn.ReLU())

            self.u = UBlock(nPlanes[1:],
                            norm_fn,
                            block_reps,
                            block,
                            indice_key_id=indice_key_id + 1,
                            normalize_before=normalize_before,
                            return_blocks=return_blocks)

            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(nPlanes[1]),
                    nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        nPlanes[1],
                        nPlanes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}")
                    )
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        nPlanes[1],
                        nPlanes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}"),
                    norm_fn(nPlanes[0]),
                    nn.ReLU())

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f"block{i}"] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key=f"subm{indice_key_id}",
                    normalize_before=normalize_before)
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self,
                input,
                previous_outputs: Optional[List]=None):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features,
                                           output.indices,
                                           output.spatial_shape,
                                           output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat(
                (identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        if self.return_blocks:
            # NOTE: to avoid the residual bug
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output




