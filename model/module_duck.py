import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from collections.abc import Sequence

from typing import List, Optional, Sequence, Tuple, Union
from torch.nn.functional import interpolate
from monai.networks.blocks.dynunet_block import UnetOutBlock, UnetBasicBlock, UnetResBlock


def get_dilated_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    dilation: Sequence[int] | int = 1,  
    act: tuple | str | None = Act.PRELU,
    norm: tuple | str | None = Norm.INSTANCE,
    dropout: tuple | str | float | None = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_dilated_padding(kernel_size, stride, dilation) 
    output_padding = None
    if is_transposed:
        output_padding = get_output_dilated_padding(kernel_size, stride, padding, dilation) 
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        dilation=dilation,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )

def get_dilated_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int, dilation: Sequence[int] | int) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    dilation_np = np.atleast_1d(dilation)
    padding_np = ((kernel_size_np - 1) * dilation_np + 1 - stride_np) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size, stride, or dilation.")
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]

def get_output_dilated_padding(
    kernel_size: Sequence[int] | int, stride: Sequence[int] | int, padding: Sequence[int] | int, dilation: Sequence[int] | int ) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    dilation_np = np.atleast_1d(dilation)
    out_padding_np = 2 * padding_np + stride_np - (kernel_size_np - 1) * dilation_np - 1
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size, stride, padding, or dilation.")
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]

def get_seperated_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 7,
    stride: Sequence[int] | int = 1,
    dilation: Sequence[int] | int = 1,  
    act: tuple | str | None = Act.PRELU,
    norm: tuple | str | None = Norm.INSTANCE,
    dropout: tuple | str | float | None = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_seperated_padding(kernel_size, stride, dilation) 
    output_padding = None
    if is_transposed:
        output_padding = get_output_seperated_padding(kernel_size, stride, padding, dilation)  
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        dilation=dilation,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )

def get_seperated_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int, dilation: Sequence[int] | int) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    dilation_np = np.atleast_1d(dilation)
    padding_np = ((kernel_size_np - 1) * dilation_np + 1 - stride_np) / 2
    if np.any(padding_np < 0):
        raise ValueError("Padding value should not be negative. Please adjust the kernel size, stride, or dilation.")
    padding = tuple(int(p) for p in padding_np)
    return padding

def get_output_seperated_padding(
    kernel_size: Sequence[int] | int, stride: Sequence[int] | int, padding: Sequence[int] | int, dilation: Sequence[int] | int ) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    dilation_np = np.atleast_1d(dilation)
    out_padding_np = 2 * padding_np + stride_np - (kernel_size_np - 1) * dilation_np - 1
    if np.any(out_padding_np < 0):
        raise ValueError("Output padding value should not be negative. Please adjust the kernel size, stride, padding, or dilation.")
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding

def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    act: tuple | str | None = Act.PRELU,
    norm: tuple | str | None = Norm.INSTANCE,
    dropout: tuple | str | float | None = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Sequence[int] | int, stride: Sequence[int] | int, padding: Sequence[int] | int
) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


