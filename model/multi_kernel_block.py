import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import LightweightConv
from model.layers import Linear, Conv1d, Conv2d

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout): 
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            Conv1d(in_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.SiLU())
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, enc_input):

        enc_output = enc_input.contiguous().transpose(1, 2) # [B, D(in_channels), seq_len]
        enc_output = F.dropout(self.conv_layer(enc_output), self.dropout, self.training) # [B, D(out_channels), seq_len]
        enc_output = self.layer_norm(enc_output.contiguous().transpose(1, 2)) # [B, seq_len, D(out_channels)]

        return enc_output

class MultiConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, trg_kernel, src_kernel):
        super(MultiConvBlock, self).__init__()

        src_padding, trg_padding = list(), list()
        for trg_k, src_k in zip(trg_kernel, src_kernel):
            trg_padding.append(0 if trg_k == 0 else (trg_k - 1) // 2)
            src_padding.append(0 if src_k == 0 else (src_k - 1) // 2)

        self.conv1 = Conv2d(in_dim, out_dim, kernel_size=(trg_kernel[0], src_kernel[0]), padding=(trg_padding[0], src_padding[0]))
        self.conv2 = Conv2d(in_dim, out_dim, kernel_size=(trg_kernel[1], src_kernel[1]), padding=(trg_padding[1], src_padding[1]))
        self.conv3 = Conv2d(in_dim, out_dim, kernel_size=(trg_kernel[2], src_kernel[2]), padding=(trg_padding[2], src_padding[2]))
        self.conv4 = Conv2d(in_dim, out_dim, kernel_size=(trg_kernel[3], src_kernel[3]), padding=(trg_padding[3], src_padding[3]))
        
        self.linear = Linear(out_dim * 4, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        
        # x: [B, trg_len, src_len, D]
        x = x.permute(0,3,1,2) # [B, D, trg_len, src_len]
        x = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=1) # [2, D*4, 28, 516]
        x = x.permute(0,2,3,1) # [B, trg_len, src_len, D]

        x = F.dropout(self.linear(F.silu(x)), p=0.1)
        x = self.norm(x)

        return x

class MultiConvBlocks(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, trg_kernel, src_kernel):
        super(MultiConvBlocks, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.convblock1 = MultiConvBlock(in_dim, out_dim, trg_kernel, src_kernel)
        self.convblock2 = MultiConvBlock(out_dim, out_dim, trg_kernel, src_kernel) 

    def forward(self, S, E, V):
        
        x = torch.cat([
            S.unsqueeze(-1), E.unsqueeze(-1), V.unsqueeze(1).expand(-1, E.size(1), -1, -1)], dim=-1) # [B, L, T, 1+1+mkb_dim (10)]
        x = self.dropout(x)
        
        x = self.convblock1(x)
        x = self.convblock2(x)

        return x

class LConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size, num_heads, dropout, weight_softmax=True):
        super(LConvBlock, self).__init__()

        embed_dim = d_model
        padding_l = (kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2))

        self.act_linear = Linear(embed_dim, 2* embed_dim)
        self.act = nn.GLU()

        self.conv_layer = LightweightConv(
            embed_dim, kernel_size, padding_l=padding_l, weight_softmax=weight_softmax,
            num_heads=num_heads, weight_dropout=dropout)

        self.fc1 = Linear(embed_dim, 4 * embed_dim)
        self.fc2 = Linear(4 * embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):

        x = x.contiguous().transpose(0, 1) # [seq_len, B, D]

        residual = x
        x = self.act_linear(x) # [seq_len, B, D * 2]
        x = self.act(x) # [seq_len, B, D]
        if mask is not None:
            if mask.dim() == 2:
                mask_ = mask.transpose(0,1).unsqueeze(2) #  [seq_len, B, 1]: [[F, F, F,..., T, T]] 
            else:
                NotImplementedError("Check mask shape!")
            x = x.masked_fill(mask_, 0)

        x = self.conv_layer(x) # [seq_len, B, D]
        x = residual + x

        residual = x
        x = F.relu(self.fc1(x)) # [seq_len, B, D * 4]
        x = self.fc2(x) # [seq_len, B, D]
        x = residual + x

        x = x.contiguous().transpose(0, 1) # [B, seq_len, D]
        x = self.layer_norm(x)

        if mask is not None:
            x = x.masked_fill(mask_.transpose(0,1), 0)

        return x
