import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import glob
import numpy as np
import torch
import random

def make_test_pairs(data_root, test_list):

    all_test_pairs = []
    with open(test_list, 'rt') as f:
        all_line = f.readlines()
        for line in all_line:
            source, target = line.strip().split('|')
            all_test_pairs.append(
                (os.path.join(data_root, source), os.path.join(data_root, target))
            )

    return all_test_pairs

def generate_path(path, speaker):

    speaker_path_list = glob.glob(os.path.join(path, speaker, '*npz'))
    random_id = random.randint(0, len(speaker_path_list)-1)
    file_path = speaker_path_list[random_id] # 'data/VCTK/VCTK_22K/val/p226/p226_006.npz'

    return file_path 

def make_inference_sample_list(path, source_list, target_list):

    source_speaker = source_list[random.randint(0, len(source_list)-1)] # ex: 'p226'
    target_speaker = target_list[random.randint(0, len(target_list)-1)]

    source_line = generate_path(path, source_speaker)  
    target_line = generate_path(path, target_speaker)    
    
    conversion_pair = [source_line + "|" + target_line + "\n"]

    return conversion_pair

def make_experiment_conversion_pair(path, dataset, male, female, n_samples=1):

    """
    create conversion pair    
    """

    conversion_list = list()
    conversion_list += make_inference_sample_list(path, male, male)
    conversion_list += make_inference_sample_list(path, male, female)
    conversion_list += make_inference_sample_list(path, female, male)
    conversion_list += make_inference_sample_list(path, female, female)

    return conversion_list

def pad_sequences(mel, max_len):

    return np.pad(mel, ((0, max_len - mel.shape[0]), (0, 0)), 'constant')

def read_rhythm(path):
    
    rhythm = dict()
    with open(path, 'rt') as f:
        all_line = f.readlines()        
        for line in all_line:
            speaker_name, speaking_rate = line.strip().split(',')
            rhythm[speaker_name] = speaking_rate

    return rhythm
    
def load_data(path, rhythm):

    # path: 'data/VCTK/VCTK_22K/val/p298/p298_016.npz'
    spk_name = path.split('/')[-2]                  # 'p298'
    spk_data = path.split('/')[-1].split('.')[0]    # 'p298_016'

    npz = np.load(path)

    audio = npz['processed_audio'].reshape(1, -1)                       # [1, wav_len]
    melspec = torch.FloatTensor(npz['melspec']).unsqueeze(0).cuda()     # [1, T, 80]
    mel_len = torch.LongTensor([melspec.size(1)]).cuda()                # [1]             
    rhythm = float(rhythm[spk_name])                                    # [1]
    
    return spk_name, spk_data, audio, melspec, mel_len, rhythm