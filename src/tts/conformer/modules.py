from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import PointwiseConv, DepthwiseConv1d, SepConv1d
from .attention import MultiHeadAttention

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model, n_head, dropout, position_emb='rotary'):
        super(MultiHeadSelfAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout, position_emb=position_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        """
        inputs:
        x : (B, L, d_model)
        key_padding_mask : (B, L)
        """
        B, L, _ = x.shape
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, L), f"{(B,L)}, {key_padding_mask.shape}"
        x = self.layer_norm(x)
        x = self.dropout(self.attn(x, key_padding_mask))
        return x

class DepthwiseConvModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super(DepthwiseConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv_1 = PointwiseConv(in_channels=d_model, out_channels=d_model * 2)
        self.pointwise_conv_2 = PointwiseConv(in_channels=d_model, out_channels=d_model)
        self.depthwise_conv = DepthwiseConv1d(in_channels=d_model, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        inputs:
        x : (B, L, d_model)
        """
        x = self.layer_norm(x).transpose(1,2)
        x = F.glu(self.pointwise_conv_1(x), dim=1)
        x = F.silu(self.batch_norm(self.depthwise_conv(x)))
        x = self.dropout(self.pointwise_conv_2(x))
        return x.transpose(1,2)
    
class ConvFeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dropout):
        super(ConvFeedForwardModule, self).__init__()
        self.conv_1 = SepConv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv_2 = SepConv1d(in_channels=d_ff, out_channels=d_model, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        inputs:
        x : (B, L, d_model)
        """
        x = self.layer_norm(x).transpose(1,2)
        x = self.dropout(F.silu((self.conv_1(x))))
        x = self.dropout(self.conv_2(x))
        return x.transpose(1,2)
    
