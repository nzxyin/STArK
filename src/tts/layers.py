from typing import Tuple, Union
import torch
import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(PointwiseConv, self).__init__()
        self.pointwise = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(x)

class DepthwiseConv1d(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            kernel_size: Union[int, Tuple[int]], 
            stride: Union[int, Tuple[int]] = 1, 
            padding: Union[str, Union[int, Tuple[int]]] = 0,
        ):
        super(DepthwiseConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, 
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise(x)

class SepConv1d(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: Union[int, Tuple[int]], 
            stride: Union[int, Tuple[int]] = 1, 
            padding: Union[str, Union[int, Tuple[int]]] = 0
        ):
        super(SepConv1d, self).__init__()
        self.depthwise = DepthwiseConv1d(in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pointwise = PointwiseConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class ConvNorm(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: Union[int, Tuple[int]], 
            stride: Union[int, Tuple[int]] = 1, 
            padding: Union[str, Union[int, Tuple[int]]] = 0
        ):
        super(ConvNorm, self).__init__()
        self.conv = SepConv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x.transpose(1,2)).transpose(1,2))
        # return self.conv(x.transpose(1,2)).transpose(1,2)