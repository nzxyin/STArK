from typing import Optional
import torch
import torch.nn as nn
from .modules import ConvFeedForwardModule, DepthwiseConvModule, MultiHeadSelfAttentionModule

class ConformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, ff_kernel_size, n_head, dropout, position_emb='rotary'):
        super(ConformerBlock, self).__init__()
        self.ffn_1 = ConvFeedForwardModule(d_model=d_model, d_ff=d_ff, kernel_size=ff_kernel_size, dropout=dropout)
        self.ffn_2 = ConvFeedForwardModule(d_model=d_model, d_ff=d_ff, kernel_size=ff_kernel_size, dropout=dropout)
        self.conv = DepthwiseConvModule(d_model=d_model, kernel_size=kernel_size, dropout=dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model=d_model, n_head=n_head, dropout=dropout, position_emb=position_emb)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        input
        x : (B, L, C)
        attn_mask : None
        key_padding_mask : (B, L)
        """
        x = x + 0.5 * self.ffn_1(x)
        x = x + self.conv(x)
        x = x + self.mhsa(x, key_padding_mask)
        x = self.layer_norm(x + 0.5 * self.ffn_2(x))
        return x

class Conformer(nn.Module):
    def __init__(self, n_layers: int, d_model, d_ff, kernel_size, ff_kernel_size, n_head, dropout, position_emb='rotary'):
        super(Conformer, self).__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, d_ff, kernel_size, ff_kernel_size, n_head, dropout, position_emb=position_emb)
            for _ in range(n_layers)
        ])

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
        return x