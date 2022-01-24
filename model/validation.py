import os
import torch
import torch.nn as nn
from model.loss import get_mask_from_lengths

def validation(model, criterion, writer, val_loader, iteration):

    model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        samples = 0     
        total_result = dict()
        for i, batch in enumerate(val_loader):
            samples += 1

            # parasing
            mel_A           = batch["mel"].cuda()           # [B, T, 80]
            mel_len_A       = batch["mel_len"].cuda()       # [B]
            speaker_id_A    = batch["speaker_id"].cuda()    # [B]
            rhythm_A        = batch["rhythm"].cuda()        # [B]  

            # run model
            outputs = model(
                mel_A, mel_len_A, rhythm_A) 
            
            # run model to consistency loss
            content_embedding_recon, style_embedding_recon = model(
                outputs['mel_outputs_postnet'], outputs['upsampled_mel_len'], rhythm_A, isRecon=True) 
            outputs['content_embedding_recon'] = content_embedding_recon
            outputs['style_embedding_recon'] = style_embedding_recon

            # loss
            labels = (mel_A, mel_len_A, speaker_id_A, rhythm_A)
            losses, result_dict = criterion(outputs, labels, iteration)

            # store result
            for key, value in result_dict.items():
                if key in total_result:
                    total_result[key] += value
                else:
                    total_result[key] = value
            
            # display result
            if i == 0:                       
                B, T, D = mel_A.size()
                mel_masks = ~get_mask_from_lengths(mel_len_A, T).unsqueeze(-1) # [B, T, 1]: [T, T, T, ..., F, F, F]

                mel_source_org      = mel_A.transpose(1,2)
                mel_output          = (outputs['mel_outputs_postnet'] * mel_masks).transpose(1,2) # [B, 80, seq_len(T)]
                downsampled_attn    = outputs['downsampled_attn']
                upsampled_attn      = outputs['upsampled_attn']

                for k in range(B):
                    writer.add_specs([
                        mel_source_org[k].detach().cpu().float(),
                        mel_output[k].detach().cpu().float()],
                        iteration, 'val_src_reconVC' + str(k))                        

                    # writer.add_alignments2(
                    #     downsampled_attn[k].detach().cpu().float(),                      
                    #     iteration, 'val_downsampled_alignment' + str(k), width=9, height=3, text_sequence=None, text=None)

                    # writer.add_alignments2(
                    #     upsampled_attn[k].detach().cpu().float(),                      
                    #     iteration, 'val_upsampled_alignment' + str(k), width=9, height=3, text_sequence=None, text=None)    

            if i == 20:                
                for key, value in total_result.items():     
                    writer.add_scalar('val_' + key, value/samples, iteration)

                break

        return True
