import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def make_masks(input_lengths, output_lengths, src_max_len, trg_max_len):
    """
    Args:
        input_lengths (LongTensor or List): Batch of lengths (B,).
        output_lengths (LongTensor or List): Batch of lengths (B,).

    Examples:
        >>> input_lengths, output_lengths = [5, 2], [8, 5]
        >>> _make_mask(input_lengths, output_lengths)
        tensor([[[1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]],
                [[1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    in_masks = ~get_mask_from_lengths(input_lengths, src_max_len)
    out_masks = ~get_mask_from_lengths(output_lengths, trg_max_len)

    return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)

def make_guided_attention_masks(input_lengths, output_lengths, src_max_len, trg_max_len, sigma):

    n_batches = len(input_lengths)
    max_ilen = max(max(input_lengths), src_max_len)
    max_olen = max(max(output_lengths), trg_max_len)
    guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen)).to(input_lengths.get_device()) # [B, trg_len, src_len]

    for idx, (ilen, olen) in enumerate(zip(input_lengths, output_lengths)):
        mask = make_guided_attention_mask(ilen, olen, sigma) # [trg_len, src_len]
        guided_attn_masks[idx, :olen, :ilen] = mask # [B, trg_len, src_len]

    return guided_attn_masks

def make_guided_attention_mask(input_lengths, output_lengths, sigma):
    """Make guided attention mask.

    Examples:
        >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
        >>> guided_attn_mask.shape
        torch.Size([5, 5])
        >>> guided_attn_mask
        tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
        >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
        >>> guided_attn_mask.shape
        torch.Size([6, 3])
        >>> guided_attn_mask
        tensor([[0.0000, 0.2934, 0.7506],
                [0.0831, 0.0831, 0.5422],
                [0.2934, 0.0000, 0.2934],
                [0.5422, 0.0831, 0.0831],
                [0.7506, 0.2934, 0.0000],
                [0.8858, 0.5422, 0.0831]])

    """
    grid_x, grid_y = torch.meshgrid(torch.arange(output_lengths), torch.arange(input_lengths))
    grid_x, grid_y = grid_x.float().to(input_lengths.get_device()), grid_y.float().to(input_lengths.get_device()) 

    return 1.0 - torch.exp(-((grid_y / input_lengths - grid_x / output_lengths) ** 2) / (2 * (sigma ** 2)))

def get_mask_from_lengths(lengths, seq_len=None):

    # if max_len is None:
    max_len = torch.max(lengths).item() # temp
    if seq_len is not None:
        max_len = max_len if max_len > seq_len else seq_len

    ids = lengths.new_tensor(torch.arange(0, max_len)).to(lengths.get_device()) 
    mask = (lengths.unsqueeze(1) <= ids).to(lengths.get_device())

    return mask # [B, seq_len]: [[F, F, F, ..., T, T, T], ... ]

def get_accuracy(pred, target_id, seq_len=None):

    _, predicted = pred.max(-1)
    if seq_len is None:
        accuracy = (predicted == target_id).float().mean() * 100 
    else:
        # calcurate accuracy except padding
        total_sum = 0.0
        for pred_, target_, len_ in zip(predicted, target_id, seq_len):
            total_sum += (pred_[:len_] == target_[:len_]).float().sum()
        accuracy = total_sum / seq_len.sum() * 100 
    
    return accuracy

class GuidedAttentionLoss(nn.Module):
    def __init__(self, hp):
        super(GuidedAttentionLoss, self).__init__()

        self.sigma = hp.guided_attention_sigma

    def forward(self, source_len, target_len, predict_attns):
        
        _, trg_len, src_len = predict_attns.size()
        
        guided_attn_masks   = make_guided_attention_masks(source_len, target_len, src_len, trg_len, self.sigma).to(source_len.get_device()) # [B, trg_len, src_len]  
        masks               = make_masks(source_len, target_len, src_len, trg_len).to(source_len.get_device()) # [B, trg_len, src_len]              
        losses              = guided_attn_masks * predict_attns  
        losses              = losses.masked_select(masks).mean()

        return losses

class LossFunction(nn.Module):
    def __init__(self, hp):
        super(LossFunction, self).__init__()

        self.l1_loss                    = nn.L1Loss(reduction="mean")
        self.cross_entropy              = nn.CrossEntropyLoss(reduction="mean")
        self.guided_attn_loss           = GuidedAttentionLoss(hp)

        self.lmabda_att                 = hp.lmabda_att
        self.lambda_scale               = hp.lambda_scale
        self.lambda_adv                 = hp.lambda_adv

    def forward(self, outputs, labels, iteration):
        
        # parsing(outputs)        
        content_embedding               = outputs['content_embedding']
        content_embedding_recon         = outputs['content_embedding_recon']
        adv_content_speaker_logits      = outputs['adv_content_speaker_logits']

        style_embedding                 = outputs['style_embedding']
        style_embedding_recon           = outputs['style_embedding_recon']

        downsampled_attn                = outputs['downsampled_attn']  
        upsampled_attn                  = outputs['upsampled_attn']     

        alpha_A                         = outputs['alpha_A']
        alpha_B                         = outputs['alpha_B']

        mel_outputs                     = outputs['mel_outputs']
        mel_outputs_postnet             = outputs['mel_outputs_postnet']        

        # parsing(labels)
        mel_A, mel_len_A, spk_id_A, rhythm_A = labels

        # masking
        max_len                         = mel_outputs.size(1)
        mel_masks                       = ~get_mask_from_lengths(mel_len_A, mel_A.size(1)).unsqueeze(-1) # [B, T, 1]: [T, T, T, ..., F, F, F]
        mel_targets                     = mel_A[:,:max_len].masked_select(mel_masks[:, :max_len])
        mel_outputs                     = mel_outputs.masked_select(mel_masks[:, :max_len])
        mel_outputs_postnet             = mel_outputs_postnet.masked_select(mel_masks[:, :max_len])     
        
        # recon loss
        l1_loss                         = self.l1_loss(mel_outputs, mel_targets)
        l1_post_loss                    = self.l1_loss(mel_outputs_postnet, mel_targets)
        content_consistency_loss        = self.l1_loss(content_embedding_recon, content_embedding.detach())
        style_consistency_loss          = self.l1_loss(style_embedding_recon, style_embedding.detach())
        
        # duration scale loss
        scale_A_loss                    = self.lambda_scale * torch.abs(alpha_A-1.0).mean()
        scale_B_loss                    = self.lambda_scale * torch.abs(alpha_B-1.0).mean()

        # guided attention loss   
        phoneme_len                     = (mel_len_A * rhythm_A).int()
        guided_attn_loss_downsampled    = self.lmabda_att * self.guided_attn_loss(mel_len_A, phoneme_len, downsampled_attn) # [B, src_len, trg_len]
        guided_attn_loss_upsampled      = self.lmabda_att * self.guided_attn_loss(phoneme_len, mel_len_A, upsampled_attn) 
        
        # adversarial speaker loss
        adv_content_speaker_loss        = self.lambda_adv * self.cross_entropy(
            adv_content_speaker_logits.transpose(1,2), # [B, D, L]
            spk_id_A.unsqueeze(1).repeat(1,adv_content_speaker_logits.size(1))) # [B, L]
        adv_content_speaker_acc         = get_accuracy(
            adv_content_speaker_logits, spk_id_A.unsqueeze(1).repeat(1,adv_content_speaker_logits.size(1)))
        
        # total loss
        total_loss = l1_loss + l1_post_loss + \
            content_consistency_loss + style_consistency_loss + scale_A_loss + scale_B_loss + \
            guided_attn_loss_downsampled + guided_attn_loss_upsampled + adv_content_speaker_loss     

        results = {
            'total_loss': total_loss.item(),
            
            'l1_loss': l1_loss.item(),
            'l1_post_loss': l1_post_loss.item(),
            
            'content_consistency_loss': content_consistency_loss.item(),
            'style_consistency_loss': style_consistency_loss.item(),

            'scale_A_loss': scale_A_loss.item(),
            'scale_B_loss': scale_B_loss.item(),

            'guided_attn_loss_downsampled': guided_attn_loss_downsampled.item(),
            'guided_attn_loss_upsampled': guided_attn_loss_upsampled.item(),
            
            'adv_content_speaker_loss': adv_content_speaker_loss.item(),
            'adv_content_speaker_acc': adv_content_speaker_acc.item(),
        }
        
        print(f'Iter {iteration:<6d} total {total_loss.item():<6.3f} l1 {l1_loss.item():<6.3f} l1_post {l1_post_loss.item():<6.3f} content {content_consistency_loss.item():<6.3f} style {style_consistency_loss.item():<6.3f} scale_A {scale_A_loss.item():<6.3f} scale_B {scale_B_loss.item():<6.3f} guide_down {guided_attn_loss_downsampled.item():<6.3f} guide_up {guided_attn_loss_upsampled.item():<6.3f} adv_spk {adv_content_speaker_loss.item():<6.3f} adv_spk_acc {adv_content_speaker_acc.item():<6.3f}%')

        return total_loss, results
 