
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.layers import Linear, Conv1d
from model.FFT_block import PositionalEncoding, ConditionalFFTBlock

class Prenet(nn.Module):
    def __init__(self, hp):
        super(Prenet, self).__init__()

        in_dim          = hp.mel_dim
        hidden_dim      = hp.prenet_dim
        out_dim         = hp.prenet_dim 
        dropout         = hp.prenet_dropout

        self.layer = nn.Sequential(
            Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, out_dim))

    def forward(self, x):
        
        return self.layer(x) # [B, T, D]
        
class ContentEncoder(nn.Module):
    def __init__(self, hp):
        super(ContentEncoder, self).__init__()
        
        # attention                
        in_dim              = hp.prenet_dim
        hidden_dim          = hp.hidden_dim
        out_dim             = hp.model_dim         
        n_layers            = hp.encoder_attn_n_layers 
        n_heads             = hp.encoder_attn_n_heads

        # fft
        ffn_dim             = hp.encoder_ffn_dim
        ffn_dropout         = hp.encoder_ffn_dropout
        ffn_style_dim       = hp.style_dim

        # prenet, positional encoding       
        self.prenet         = Prenet(hp)
        self.register_buffer('pe', PositionalEncoding(in_dim).pe)
        self.alpha          = nn.Parameter(torch.ones(1))
        self.dropout        = nn.Dropout(hp.prenet_dropout)

        # encoder
        self.in_linear      = Linear(in_dim, hidden_dim)
        self.encoder_layer  = nn.ModuleList([
            ConditionalFFTBlock(hidden_dim, n_heads, hidden_dim, ffn_dim, ffn_dropout, ffn_style_dim)
            for i in range(n_layers)])
        self.out_linear     = Linear(hidden_dim, out_dim)

    def forward(self, mel, style_embedding, mask):       

        # mel: [B, T, 80]
        # style embedding: [B, D(speaker)]

        prenet_output = self.prenet(mel) # [B, T, D(model)]
        pos_embedding = self.alpha * self.pe[:mel.size(1)].unsqueeze(0) # [1, T, D]
        x = self.dropout(prenet_output + pos_embedding)

        # attention & FFT
        x = self.in_linear(x) # [B, T, D(hidden)]
        for enc_layers in self.encoder_layer:
            x, enc_attn = enc_layers(x, style_embedding, mask) # [B, T, D(hidden)], [B, T, T]
        x = self.out_linear(x) # [B, T, D(model)]

        return x
