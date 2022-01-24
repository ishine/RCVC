import torch
from utils.utils import display_result_and_save_tensorboard

def train(model, optimizer, scheduler, criterion, writer, iteration, batch, scaler):

    # parsing
    mel_A           = batch["mel"].cuda()           # [B, T, 80]
    mel_len_A       = batch["mel_len"].cuda()       # [B]
    speaker_id_A    = batch["speaker_id"].cuda()    # [B]
    rhythm_A        = batch["rhythm"].cuda()        # [B]  

    with torch.cuda.amp.autocast(): 

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

        # update
        model.zero_grad()
        scaler.scale(losses).backward()
        scheduler.step(iteration)
        scaler.step(optimizer)        
        scaler.update()

    # display
    result_dict['lr'] = scheduler.get_learning_rate()
    display_result_and_save_tensorboard(writer, result_dict, iteration)

    return True

