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

from model.module_duck import get_dilated_conv_layer, get_seperated_conv_layer, get_conv_laye

class WideUnetBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv1 = get_dilated_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation= 1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_dilated_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation= 2,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv3 = get_dilated_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation= 3,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.lrelu(out)
        return out

class MidUnetBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv1 = get_dilated_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation= 1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_dilated_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation= 2,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out

class SeperatedUnetBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv1 = get_seperated_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=(1, 1, kernel_size),
            stride=1,
            dilation= 1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_seperated_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size, 1),
            stride=1,
            dilation= 1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv3 = get_seperated_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1, 1),
            stride=1,
            dilation= 1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.lrelu(out)
        return out

class DuckBlock3D(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super(DuckBlock3D, self).__init__()
        
        self.wide_unet_block = WideUnetBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout)
        self.mid_unet_block = MidUnetBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout)
        self.separated_unet_block = SeperatedUnetBlock(spatial_dims, in_channels, out_channels, 5 , stride, norm_name, act_name, dropout)
        
        self.res_unet_block_1 = UnetResBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout)
        
        self.res_unet_block_2 = nn.Sequential(
            UnetResBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout),
            UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout)
        )
        
        self.res_unet_block_3 = nn.Sequential(
            UnetResBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout),
            UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout),
            UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout)
        )
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, x):
        norm_1 = self.norm1(x)
        wide_out = self.wide_unet_block(norm_1)
        mid_out = self.mid_unet_block(norm_1)
        separated_out = self.separated_unet_block(norm_1)
        res_out_1 = self.res_unet_block_1(norm_1)
        res_out_2 = self.res_unet_block_2(norm_1)
        res_out_3 = self.res_unet_block_3(norm_1)
        out = mid_out  + separated_out + res_out_2 + res_out_1
        return out

class Residualx2_BottleNeck(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super(Residualx2_BottleNeck, self).__init__()
        
        self.res_unet_block_2 = nn.Sequential(
            UnetResBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout),
            UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size, 1, norm_name, act_name, dropout)
        )

    def forward(self, x):
        out = self.res_unet_block_2(x)
        return out

class UnetBasicDuckBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = DuckBlock3D(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act_name=act_name,
            norm_name=norm_name
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out

class UnetUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        self.conv_block = DuckBlock3D(
            spatial_dims,
            out_channels + out_channels,
            out_channels , 
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1) # Sửa ở đây -----------------------------------
#         out = torch.add(out, skip)
        out = self.conv_block(out)
        return out
