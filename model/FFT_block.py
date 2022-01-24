import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Linear, Conv1d, Conv2d

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pe', self._get_pe_matrix(d_model, max_len))

    def forward(self, x):

        return x + self.pe[:x.size(0)].unsqueeze(1)

    def _get_pe_matrix(self, d_model, max_len):

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        return pe
        

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(FeedForwardNetwork, self).__init__()

        self.layer = nn.Sequential(
            Conv1d(in_dim, hidden_dim, 9, padding=(9 - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Conv1d(hidden_dim, out_dim, 9, padding=(9 - 1) // 2))

    def forward(self, x):
        
        return self.layer(x)


class ConditionalFFTBlock(nn.Module):
    def __init__(self, in_dim, n_heads, out_dim, ffn_dim, dropout, style_dim):
        super(ConditionalFFTBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(in_dim, n_heads, dropout=dropout)

        self.ffn = FeedForwardNetwork(
            in_dim=in_dim + style_dim, hidden_dim=ffn_dim, out_dim=out_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(normalized_shape=in_dim, eps=1e-12)
        self.norm2 = nn.LayerNorm(normalized_shape=out_dim, eps=1e-12)        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, style_embedding, mask):
        
        # x: [B, T, Dim] 
        residual = x         
        x = x.transpose(0,1) # [T, B, D] for attention input
        att, dec_attn = self.self_attn(x, x, x, attn_mask=None, key_padding_mask=mask) # [T, B, D], [B, T, T]
        x = residual + self.dropout1(att.transpose(0,1))
        x = self.norm1(x)

        residual = x
        if style_embedding is not None:
            x = torch.cat([x, style_embedding.unsqueeze(1).repeat(1,x.size(1),1)], dim=-1) # [B, T, D(model+style)]
        x = residual + self.dropout2(self.ffn(x.transpose(1,2)).transpose(1,2)) # [B, T, D(model)]
        x = self.norm2(x)

        return x, dec_attn
