import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.multi_kernel_block import ConvBlock, LConvBlock, MultiConvBlocks
from model.loss import get_mask_from_lengths
from model.layers import Linear, Conv1d

class Downsampling(nn.Module):
    def __init__(self, hp):
        super(Downsampling, self).__init__()

        # input linear & duration predictor
        input_dim                   = hp.model_dim + hp.style_dim
        hidden_dim                  = hp.duration_predictor_dim
        kernel_size                 = hp.duration_predictor_kernel_size
        n_heads                     = hp.duration_predictor_n_heads
        dropout_ratio               = hp.duration_predictor_dropout_ratio
        n_layers                    = hp.duration_predictor_n_layers
        downsampled_dim             = hp.downsampled_dim

        # learnable downsampling
        conv_output_dim             = hp.downsampling_conv_dim        
        mkb_dim                     = hp.downsampling_mkb_dim
        trg_kernel                  = hp.downsampling_trg_kernel_size # phoneme
        src_kernel                  = hp.downsampling_src_kernel_size # mel
        max_seq_len                 = hp.downsampling_max_seq_len

        self.input_linear           = Linear(input_dim, hidden_dim)
        self.duration_predictor     = DurationPredictor(
            hidden_dim, kernel_size, n_heads, dropout_ratio, n_layers, downsampled_dim)
        self.learnable_downsampling = LearnableDownsampling(
            hidden_dim, conv_output_dim, kernel_size, dropout_ratio, mkb_dim, trg_kernel, src_kernel, downsampled_dim, max_seq_len)

    def forward(self, content_encoder_output, style_embedding, mask, rhythm_A, mel_len_A):
                
        # content_encoder_output: [B, T, D]
        # style_embedding: [B, D]

        # concat (content & style embedding)
        x = torch.cat([
            content_encoder_output, style_embedding.unsqueeze(1).repeat(1,content_encoder_output.size(1),1)], dim=-1) # [B, T, D(model+style)]
        x = self.input_linear(x) # [B, T, D]

        # rhythm-based duration predictor
        D, V, P = self.duration_predictor(x, mask)
        
        # learnable downsampling
        content_embedding, text_mask, downsampled_attn, alpha_A, scaled_D = self.learnable_downsampling(
            D, V, P, mask, rhythm_A, mel_len_A)

        outputs = {
            'content_embedding': content_embedding,     # [B, L, D(downsample)]
            'text_mask': text_mask,                     # [B, L]
            'downsampled_attn': downsampled_attn,       # [B, L, T]
            'alpha_A': alpha_A,                         # [B]
            'scaled_D': scaled_D                        # [B, T]
        }

        return outputs

class DurationPredictor(nn.Module):
    def __init__(self, hidden_dim, kernel_size, n_heads, dropout_ratio, n_layers, out_dim):
        super(DurationPredictor, self).__init__()

        self.convolution_stack = nn.ModuleList([
            LConvBlock(hidden_dim, kernel_size, n_heads, dropout=dropout_ratio)
            for _ in range(n_layers)])        

        # self.projection1 = nn.Linear(hidden_dim, 1) 
        # self.projection2 = nn.Conv1d(
        #     hidden_dim, out_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.projection1 = Linear(hidden_dim, 1) 
        self.projection2 = Conv1d(
            hidden_dim, out_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, mask):
        
        # x: [B, T or L, D]
        # mask: # [B, T or L]

        for i, conv_layer in enumerate(self.convolution_stack):
            V = conv_layer(x if i==0 else V, mask=mask) # [B, T or L, D]
        
        D = F.softplus(self.projection1(V)).squeeze(-1) # [B, T or L]    
        D = D.masked_fill(mask, 0)

        P = self.projection2(V.transpose(1,2)).transpose(1,2) # [B, T or L, D(sample)]

        return D, V, P # [B, T or L], [B, T or L, D], [B, T or L, D(sample)]

class LearnableDownsampling(nn.Module):
    def __init__(self, in_dim, conv_output_dim, kernel_size, dropout_ratio, mkb_dim, trg_kernel, src_kernel, downsampled_dim, max_seq_len):
        super(LearnableDownsampling, self).__init__()

        self.max_seq_len    = max_seq_len

        # attention
        self.conv_w         = ConvBlock(in_dim, conv_output_dim, kernel_size, dropout=dropout_ratio)
        self.MKB            = MultiConvBlocks(conv_output_dim+2, mkb_dim, mkb_dim, trg_kernel, src_kernel)
        self.linear_w       = Linear(mkb_dim, 1) 
        self.softmax_w      = nn.Softmax(dim=2)

        # downsampled representation
        self.layer_norm     = nn.LayerNorm(downsampled_dim)

    def forward(self, D, V, P, src_mask, rhythm_A, target_mel_len):

        B = D.shape[0]
        max_src_len = D.size(1)

        # duration controller        
        predicted_sum = D.sum(-1) # [B]
        alpha_A = (target_mel_len * rhythm_A) / predicted_sum # [B]

        # scaling
        scaled_D = alpha_A.unsqueeze(1) * D # [B, T] = [B, 1] * [B, T]
        scaled_V = alpha_A.unsqueeze(1).unsqueeze(1) * V # [B, T, D]
        scaled_P = alpha_A.unsqueeze(1).unsqueeze(1) * P # [B, T, downsampled_dim]

        # generate mask
        text_len_output = torch.round(scaled_D.sum(-1)).type(torch.LongTensor).view(-1).to(scaled_D.get_device()) # [B, T] -> [B]
        text_len_output = torch.clamp(text_len_output, max=self.max_seq_len, min=1)
        max_text_len = text_len_output.max().int().item() # scalar
        text_mask = get_mask_from_lengths(text_len_output, None) # [B, L]: [[F, F, F,..., T, T]] 

        # prepare attention mask
        src_mask_ = src_mask.float().unsqueeze(-1) # [B, src_len(T), 1]
        text_mask_ = text_mask.float().unsqueeze(-1) # [B, tgt_len(L), 1]
        attn_mask = torch.bmm(text_mask_, src_mask_.transpose(-2, -1)).bool() # [B, tgt_len(L), src_len(T)]

        # frame boundary grid
        e_k = torch.cumsum(scaled_D, dim=1) # [B, T]   
        s_k = e_k - scaled_D          
        e_k = e_k.unsqueeze(1).expand(B, max_text_len, -1) # [B, L, T]
        s_k = s_k.unsqueeze(1).expand(B, max_text_len, -1)
        t_arange = torch.arange(1, max_text_len+1).unsqueeze(0).unsqueeze(-1).expand(B, -1, max_src_len).to(scaled_D.get_device())
        S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(attn_mask, 0)

        # attention (W)
        W = self.MKB(S, E, self.conv_w(scaled_V)) # [B, L, T, mkb_dim]
        W = self.linear_w(W).squeeze(-1).masked_fill(attn_mask, -np.inf) # [B, L, T]  
        W = W.masked_fill(src_mask_.transpose(1,2).bool(), -np.inf)       
        W = self.softmax_w(W)

        # content embedding (C)
        content_embedding = torch.matmul(W, scaled_P) # [B, L, downsampled_dim]
        content_embedding = self.layer_norm(content_embedding)
        content_embedding = content_embedding.masked_fill(text_mask_.bool(), 0)

        return content_embedding, text_mask, W, alpha_A, scaled_D

class Upsampling(nn.Module):
    def __init__(self, hp):
        super(Upsampling, self).__init__()
        
        # input linear & duration predictor
        input_dim           = hp.downsampled_dim + hp.style_dim
        hidden_dim          = hp.duration_predictor_dim        
        kernel_size         = hp.duration_predictor_kernel_size
        n_heads             = hp.duration_predictor_n_heads
        dropout_ratio       = hp.duration_predictor_dropout_ratio
        n_layers            = hp.duration_predictor_n_layers
        upsampled_dim       = hp.upsampled_dim
        
        # learable upsampling
        conv_output_dim     = hp.upsampling_conv_dim
        mkb_dim             = hp.upsampling_mkb_dim   
        trg_kernel          = hp.upsampling_trg_kernel_size # [3,5,7,11] # mel
        src_kernel          = hp.upsampling_src_kernel_size # [1,1,1,1] # phoneme
        max_seq_len         = hp.upsampling_max_seq_len         

        self.input_linear = Linear(input_dim, hidden_dim)
        self.duration_predictor = DurationPredictor(
            hidden_dim, kernel_size, n_heads, dropout_ratio, n_layers, upsampled_dim)
        self.learned_upsampling = LearnableUpsampling(
            hidden_dim, conv_output_dim, kernel_size, dropout_ratio, mkb_dim, trg_kernel, src_kernel, upsampled_dim, max_seq_len)

    def forward(self, down_outputs, speaker_embedding_B, rhythm_B):

        # content_embedding: [B, L, D]

        # concat (content & style embedding)
        x = torch.cat([
            down_outputs['content_embedding'], 
            speaker_embedding_B.unsqueeze(1).repeat(1,down_outputs['content_embedding'].size(1),1)], dim=-1) # [B, L, D(downsampled+style)]
        x = self.input_linear(x) # [B, T, D]

        # rhythm-based duration predictor
        duration_output, V, T = self.duration_predictor(x, down_outputs['text_mask'])

        # learnable upsampling
        # upsampled_rep, upsampled_attn, upsampled_mel_mask, alpha_B, upsampled_mel_len = self.learned_upsampling(
        upsampled_rep, upsampled_attn, alpha_B, upsampled_mel_len, upsampled_mel_mask = self.learned_upsampling(
            duration_output, V, T, down_outputs['text_mask'], rhythm_B, down_outputs['scaled_D'])

        outputs = {
            'upsampled_rep': upsampled_rep,             # [B, T', D]
            'upsampled_attn': upsampled_attn,           # [B, T', L]
            'upsampled_mel_mask': upsampled_mel_mask,   # [B, T']
            'upsampled_mel_len': upsampled_mel_len,     # [B]
            'alpha_B': alpha_B,                         # [B]            
        }

        return outputs

class LearnableUpsampling(nn.Module):
    def __init__(self, in_dim, conv_output_dim, kernel_size, dropout_ratio, mkb_dim, trg_kernel, src_kernel, upsampled_dim, max_seq_len):
        super(LearnableUpsampling, self).__init__()

        # mask
        self.max_seq_len    = max_seq_len

        # attention (W)
        self.conv_w         = ConvBlock(in_dim, conv_output_dim, kernel_size, dropout=dropout_ratio)
        self.MKB            = MultiConvBlocks(conv_output_dim+2, mkb_dim, mkb_dim, trg_kernel, src_kernel)
        self.linear_w       = Linear(mkb_dim, 1) 
        self.softmax_w      = nn.Softmax(dim=2)

        # upsampled representation
        self.layer_norm     = nn.LayerNorm(upsampled_dim)

    def forward(self, D, V, T, src_mask, rhythm_B, scaled_duration_outputs):

        B = D.shape[0]
        max_src_len = D.size(1)

        # duration controller
        pred_phoneme_len = scaled_duration_outputs.sum(-1)
        predicted_sum = D.sum(-1) # [B]
        alpha_B = (pred_phoneme_len / rhythm_B).detach() / predicted_sum # [B], target은 고정시켜본다
        
        # scaling
        scaled_D = alpha_B.unsqueeze(1) * D # [B, L] = [B, 1] * [B, L]
        scaled_V = alpha_B.unsqueeze(1).unsqueeze(1) * V 
        scaled_T = alpha_B.unsqueeze(1).unsqueeze(1) * T

        # generate mask
        mel_len_output = torch.round(scaled_D.sum(-1)).type(torch.LongTensor).view(-1).to(scaled_D.get_device()) # [B] 
        mel_len_output = torch.clamp(mel_len_output, max=self.max_seq_len, min=1)
        max_mel_len = mel_len_output.max().int().item() # scalar
        mel_mask = get_mask_from_lengths(mel_len_output, None) # [B, T]: [[F, F, F,..., T, T]] 

        # prepare attention mask
        src_mask_ = src_mask.float().unsqueeze(-1) # [B, src_len(L), 1]
        mel_mask_ = mel_mask.float().unsqueeze(-1) # [B, tgt_len(T), 1]
        attn_mask = torch.bmm(mel_mask_, src_mask_.transpose(-2, -1)).bool() # [B, T(scaled_D), L(source)]

        # phoneme boundary grid
        e_k = torch.cumsum(scaled_D, dim=1) # [B, L]  
        s_k = e_k - scaled_D          
        e_k = e_k.unsqueeze(1).expand(B, max_mel_len, -1) # [B, T, L]
        s_k = s_k.unsqueeze(1).expand(B, max_mel_len, -1) 
        t_arange = torch.arange(1, max_mel_len+1).unsqueeze(0).unsqueeze(-1).expand(B, -1, max_src_len).to(scaled_D.get_device())
        S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(attn_mask, 0)

        # attention (W)
        W = self.MKB(S, E, self.conv_w(scaled_V)) # [B, T, L, mkb_dim]
        W = self.linear_w(W).squeeze(-1).masked_fill(attn_mask, -np.inf) # [B, T, L]
        W = W.masked_fill(src_mask_.transpose(1,2).bool(), -np.inf)  
        W = self.softmax_w(W)

        # Upsampled Representation
        upsampled_rep = torch.matmul(W, scaled_T) # [B, T, upsampled_dim]
        upsampled_rep = self.layer_norm(upsampled_rep)
        upsampled_rep = upsampled_rep.masked_fill(mel_mask_.bool(), 0)

        return upsampled_rep, W, alpha_B, mel_len_output, mel_mask
