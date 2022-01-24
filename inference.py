import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from tqdm import tqdm
import soundfile as sf

from model.model import Model
import model.hparams as hp
from utils.inference_utils import make_test_pairs, make_experiment_conversion_pair, pad_sequences, read_rhythm, load_data
from vocoder.inference_npy import main as run_vocoder

def run_inference(checkpoint_model, log_directory, org_save):
    
    # hyper parameter    
    experiment_dataset = ['evaluation_inter_list.txt', 'evaluation_intra_list.txt']
    data_root = os.path.join('data', hp.dataset, hp.dataset_path) # data/VCTK/VCTK_22K
    experiment_dataset = {
        'inter': os.path.join('data', hp.dataset, experiment_dataset[0]),
        'intra': os.path.join('data', hp.dataset, experiment_dataset[1])}
    checkpoint_model_path = os.path.join('outputs', checkpoint_model, 'checkpoint_150000')
    rhythm = read_rhythm(os.path.join('data', hp.dataset, hp.dataset_path, 'rhythm.txt')) 

    # load model
    model = Model(hp).cuda()
    model.load_state_dict(torch.load(checkpoint_model_path)['model'])
    model.eval()
    print("load checkpoint(main): ", checkpoint_model_path)

    with torch.no_grad():
        for experiment_name, dataset_path in experiment_dataset.items():
            print("\nstart test! --> ", experiment_name)

            # save directory
            save_dir = "generated/{}/{}".format(experiment_name, log_directory) 
            os.makedirs(save_dir, exist_ok=True)

            # load evaluation test
            all_test_pairs = make_test_pairs(data_root, dataset_path)

            for i, (src_spk, trg_spk) in enumerate(tqdm(all_test_pairs)):
                
                # load data            
                src_spk_name, src_data, audio_A, mel_A, mel_len_A, rhythm_A = load_data(src_spk, rhythm)
                trg_spk_name, trg_data, audio_B, mel_B, mel_len_B, rhythm_B = load_data(trg_spk, rhythm)

                mel_outputs_postnet_convert = model.inference(
                    mel_A, mel_len_A, rhythm_A, mel_B, mel_len_B, rhythm_B) 

                # save result
                source_mel = mel_A.squeeze(0).float().detach().cpu().numpy().T # [1, T, 80] -> [80, T]
                target_mel = mel_B.squeeze(0).float().detach().cpu().numpy().T               
                converted_mel = mel_outputs_postnet_convert.squeeze(0).float().detach().cpu().numpy().T
                path = "{}_to_{}.npy".format(src_data, trg_data) # p294_005_to_p334_005.npy

                # Inference result
                np.save(os.path.join(save_dir, path), converted_mel)

                if org_save == True:
                    # original ground-truths
                    os.makedirs(os.path.join(save_dir + 'GT_source_wav'), exist_ok=True)
                    os.makedirs(os.path.join(save_dir + 'GT_target_wav'), exist_ok=True)
                    sf.write(os.path.join(save_dir + 'GT_source_wav', path) + ".wav", audio_A, 22050, 'PCM_24')
                    sf.write(os.path.join(save_dir + 'GT_target_wav', path) + ".wav", audio_B, 22050, 'PCM_24')

                    # reconstructed ground-truths
                    os.makedirs(os.path.join(save_dir + 'GT_source'), exist_ok=True)
                    os.makedirs(os.path.join(save_dir + 'GT_target'), exist_ok=True)
                    np.save(os.path.join(save_dir + 'GT_source', path), source_mel)
                    np.save(os.path.join(save_dir + 'GT_target', path), target_mel)    
                break     
            
            if org_save == True:
                run_vocoder(hp, path=os.path.join(experiment_name, 'GT_source'))
                run_vocoder(hp, path=os.path.join(experiment_name, 'GT_target'))
            run_vocoder(hp, path=os.path.join(experiment_name, log_directory))

    print("complete inference!")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_model', default='RCVC_VCTK') # VCTK, NIKL
    parser.add_argument('--dataset', default='VCTK') # VCTK, NIKL
    parser.add_argument('--log_dir', default='RCVC_VCTK') # VCTK, NIKL
    parser.add_argument('--org_save', default=False) 
    a = parser.parse_args()

    hp.dataset = a.dataset
    checkpoint_model = a.checkpoint_model
    log_directory = a.log_dir
    org_save = a.org_save

    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    run_inference(checkpoint_model, log_directory, org_save)


