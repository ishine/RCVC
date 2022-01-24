import torch
import torch.nn as nn

from model.layers import Linear, Conv1d, Conv2d
from model.encoder_style import StyleEncoder
from model.encoder_content import ContentEncoder
from model.dynamic_information_bottleneck import Downsampling, Upsampling
from model.decoder import Decoder
from model.postnet import Postnet
from model.adversarial_speaker_classifier import ContentSpeakerClassifier, grad_reverse
from model.loss import get_mask_from_lengths

class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
                
        self.style_encoder                  = StyleEncoder(hp) 
        self.content_encoder                = ContentEncoder(hp)
        
        self.downsampling                   = Downsampling(hp)
        self.adv_content_speaker_classifier = ContentSpeakerClassifier(hp)  
        self.upsampling                     = Upsampling(hp)

        self.decoder                        = Decoder(hp)             
        self.mel_linear                     = Linear(hp.model_dim, hp.mel_dim)
        self.postnet                        = Postnet(hp)

    def forward(self, mel_A, mel_len_A, rhythm_A, isRecon=False):

        # style
        style_embedding_A = self.style_encoder(mel_A, mel_len_A, mel_A.size(1)) # [B, D(style)]  
        # content
        encoder_mask = get_mask_from_lengths(mel_len_A, mel_A.size(1)) # [B, T]: [[F, F, F,..., T, T]]         
        content_encoder_output = self.content_encoder(mel_A, style_embedding_A, encoder_mask) # [B, T, D(model)]

        ############################################################################################################################
        # dynamic information bottleneck
        down_outputs = self.downsampling(
            content_encoder_output, style_embedding_A, encoder_mask, rhythm_A, mel_len_A)

        if isRecon == True:
            return down_outputs['content_embedding'], style_embedding_A

        adv_content_speaker_logits = self.adv_content_speaker_classifier(
            grad_reverse(down_outputs['content_embedding'])) # [B, L, n_speakers]

        up_outputs = self.upsampling(
            down_outputs, style_embedding_A, rhythm_A)
        ############################################################################################################################

        # decoder
        decoder_output = self.decoder(
            up_outputs['upsampled_rep'], style_embedding_A, up_outputs['upsampled_mel_mask']) # [B, T', D]
        mel_outputs = self.mel_linear(decoder_output) # [B, T', 80]
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        outputs = {
            'style_embedding': style_embedding_A,
            'content_embedding': down_outputs['content_embedding'], 

            'downsampled_attn': down_outputs['downsampled_attn'],
            'upsampled_attn': up_outputs['upsampled_attn'],        

            'alpha_A': down_outputs['alpha_A'],
            'alpha_B': up_outputs['alpha_B'], 

            'upsampled_mel_len': up_outputs['upsampled_mel_len'],

            'mel_outputs': mel_outputs,
            'mel_outputs_postnet': mel_outputs_postnet,

            'adv_content_speaker_logits': adv_content_speaker_logits,
        }

        return outputs

    def inference(self, mel_A, mel_len_A, rhythm_A, mel_B, mel_len_B, rhythm_B):
    
        # style
        style_embedding_A = self.style_encoder(mel_A, mel_len_A, mel_A.size(1)) # [B, D(style)]   
        style_embedding_B = self.style_encoder(mel_B, mel_len_B, mel_A.size(1))  

        # content
        encoder_mask = get_mask_from_lengths(mel_len_A, mel_A.size(1)) # [B, T]: [[F, F, F,..., T, T]]         
        content_encoder_output = self.content_encoder(mel_A, style_embedding_A, encoder_mask) # [B, T, D(model)]

        # dynamic information bottleneck
        down_outputs = self.downsampling(
            content_encoder_output, style_embedding_A, encoder_mask, rhythm_A, mel_len_A)

        up_outputs = self.upsampling(
            down_outputs, style_embedding_B, rhythm_B)

        # decoder
        decoder_output = self.decoder(
            up_outputs['upsampled_rep'], style_embedding_B, up_outputs['upsampled_mel_mask']) # [B, T', D]
        mel_outputs = self.mel_linear(decoder_output) # [B, T', 80]
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        return mel_outputs_postnet