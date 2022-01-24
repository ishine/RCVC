import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.layers import Linear, Conv1d, Conv2d
from model.FFT_block import PositionalEncoding, ConditionalFFTBlock

class Decoder(nn.Module):
    def __init__(self, hp):
        super(Decoder, self).__init__()        

        # attention
        in_dim          = hp.duration_predictor_dim
        hidden_dim      = hp.hidden_dim
        out_dim         = hp.model_dim
        n_layers        = hp.decoder_attn_n_layers
        n_heads         = hp.decoder_attn_n_heads

        # fft
        ffn_dim         = hp.decoder_ffn_dim
        ffn_dropout     = hp.decoder_dropout
        ffn_style_dim   = hp.style_dim

        self.register_buffer('pe', PositionalEncoding(hp.upsampled_dim).pe)
        self.alpha      = nn.Parameter(torch.ones(1))
        self.dropout    = nn.Dropout(0.1)
        
        self.in_layer   = Linear(in_dim, hidden_dim)
        self.decoder_layer = nn.ModuleList([
            ConditionalFFTBlock(hidden_dim, n_heads, hidden_dim, ffn_dim, ffn_dropout, ffn_style_dim) 
            for i in range(n_layers)])
        self.out_layer  = Linear(hidden_dim, out_dim)

    def forward(self, x, style_embedding, mask):        
        
        # Positional encoding
        pos_embedding = self.alpha * self.pe[:x.size(1)].unsqueeze(0) # [1, T', D]
        x = self.dropout(x + pos_embedding)

        # Attention & FFT
        x = self.in_layer(x) # [B, T, hidden_dim]
        for dec_layers in self.decoder_layer:
            x, dec_attn = dec_layers(x, style_embedding, mask) #  [B, T', hidden_dim], [B, T', T']
        x = self.out_layer(x) # [B, T', D]

        return x
