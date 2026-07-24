import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_head=4, dropout=0.1, position_emb='rotary'):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        d_qkv = d_model // n_head
        assert d_qkv * n_head == d_model, 'd_model must be divisible by n_head'
        assert d_qkv % 2 == 0, 'd_qkv must be even for rotary embeddings'
        self.d_qkv = d_qkv

        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)

        self.dropout = nn.Dropout(dropout)
        if position_emb == 'rotary':
            self.rotary_emb = RotaryEmbedding(dim=d_qkv//2, )
        else:
            self.rotary_emb = None

    def forward(self, x, attention_mask=None):
        """Runs the multi-head self-attention layer.

        Args:
        x: the input to the layer, a tensor of shape [batch_size, length, d_model]
        attention_mask: a tensor of shape [batch_size, length]
        Returns:
        A single tensor containing the output from this layer
        """

        q = torch.einsum('btf,hfa->bhta', x, self.w_q)
        k = torch.einsum('btf,hfa->bhta', x, self.w_k)
        v = torch.einsum('btf,hfa->bhta', x, self.w_v)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb.rotate_queries_or_keys(q), self.rotary_emb.rotate_queries_or_keys(k)

        logits = torch.einsum('bhqa,bhka->bhqk', q, k) / (self.d_qkv ** 0.5)

        q_pos = q.permute(2,0,1,3) #bhqd->qbhd
        l,b,h,d = q_pos.size()
        # position_logits, _ = self.relative_positional(q_pos.reshape(l,b*h,d))
        # (bh)qk
        # logits = logits + position_logits.view(b,h,l,l)

        if attention_mask is not None:
            assert attention_mask.shape == (b, l), f"{(b,l)}, {attention_mask.shape}"
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat(1, h, 1).unsqueeze(-1)
            logits.masked_fill(attention_mask.to(torch.bool), -1e8)

        probs = F.softmax(logits, dim=-1)
        probs = self.dropout(probs)
        o = torch.einsum('bhqk,bhka->bhqa', probs, v)
        out = torch.einsum('bhta,haf->btf', o, self.w_o)
        return out
