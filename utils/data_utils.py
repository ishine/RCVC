
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import glob
import pickle
import random

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, hp):
        
        # data
        self.rhythm             = self.read_rhythm(
            os.path.join('data', hp.dataset, hp.dataset_path, 'rhythm.txt'))
        self.speaker_ids        = hp.speaker_ids
        self.seq_len            = hp.seq_len
        self.cache_data         = dict()

        # dataset        
        self.npz_path = self.get_npz_path(data_dir)
        random.seed(random.randint(1, 10000))
        random.shuffle(self.npz_path)
        print("load dataset num: {}".format(len(self.npz_path)))

    def read_rhythm(self, path):
    
        rhythm = dict()
        with open(path, 'rt') as f:
            all_line = f.readlines()        
            for line in all_line:
                speaker_name, speaking_rate = line.strip().split(',')
                rhythm[speaker_name] = float(speaking_rate)

        return rhythm

    def get_npz_path(self, path):
        
        spk_path = glob.glob(os.path.join(path, '*'))

        npz_path = list()
        for spk in spk_path: # ['d:/datasets/VCTK/VCTK_22K_hifi_org_alignment/train\\p225', ... ,]

            npz_path += glob.glob(os.path.join(spk, r"*.npz")) 
            
        return npz_path          

    def get_sample(self, npz_path):

        if npz_path in self.cache_data.keys(): 
            return self.cache_data[npz_path]

        else:
            npz = np.load(npz_path, allow_pickle=True)

            melspec = torch.FloatTensor(npz['melspec'])         # [T, 80]
            speaker_name = str(npz['speaker_name'])             # ex: 'p225'
            speaker_id = self.speaker_ids[speaker_name]         # ex: 45
            rhythm = self.rhythm[speaker_name]                  # ex: 0.20705114307028244
            
            self.cache_data[npz_path] = (melspec, speaker_id, rhythm)

            return self.cache_data[npz_path]

    def __getitem__(self, index):

        return self.get_sample(self.npz_path[index])
    
    def __len__(self):
    
        return len(self.npz_path)

class DatasetCollate():
    def __init__(self, hp):

        self.seq_len = hp.seq_len

    def __call__(self, batch):

        # parsing        
        mels                = [b[0] for b in batch]
        spk_id              = [b[1] for b in batch]
        rhythms             = [b[2] for b in batch]

        # make batch
        T = max(m.size(0) for m in mels)      
        mel_seq, mel_len_seq, rhythm_seq = list(), list(), list()
        for i, (mel, rhythm) in enumerate(zip(mels, rhythms)):  

            frame_len = mel.shape[0]
            if frame_len < self.seq_len: 
                len_pad = self.seq_len - frame_len
                x = np.pad(mel, ((0, len_pad), (0, 0)), 'constant')
                mel_len_seq.append(frame_len)
            else:
                start = np.random.randint(frame_len - self.seq_len + 1)
                x = mel[start:start + self.seq_len]
                mel_len_seq.append(self.seq_len)

            mel_seq.append(x) # [T, 80]  
            rhythm_seq.append(rhythm)

        mel_seq = np.stack(mel_seq, axis=0)             # [T(seq_len), 80] * batch -> [B, T(seq_len), 80]
        rhythm_seq = np.stack(rhythm_seq, axis=0)

        out = {
            "mel": torch.FloatTensor(mel_seq),          # [B, T(seq_len), 80]
            "mel_len": torch.LongTensor(mel_len_seq),   # [B]
            "speaker_id": torch.LongTensor(spk_id),     # [B]
            "rhythm": torch.FloatTensor(rhythm_seq),    # [B]
        }

        return out

def prepare_dataloaders(hp, num_workers):
    
    training_data_path = os.path.join('data', hp.dataset, hp.dataset_path, 'train')
    validation_data_path = os.path.join('data', hp.dataset, hp.dataset_path, 'val')
    
    # Get data, data loaders and collate function ready
    trainset = Dataset(training_data_path, hp)    
    collate_fn = DatasetCollate(hp)
    
    train_loader = DataLoader(
        trainset, num_workers=num_workers, shuffle=True, pin_memory=True, sampler=None, 
        batch_size=hp.train_batch_size, drop_last=True, collate_fn=collate_fn)

    valset = Dataset(validation_data_path, hp)
    val_loader = DataLoader(
        valset, num_workers=1, shuffle=True, pin_memory=True, sampler=None, 
        batch_size=hp.val_batch_size, drop_last=True, collate_fn=collate_fn)

    return train_loader, val_loader
