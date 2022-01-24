
import os
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
np.seterr(all="ignore")

import librosa
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import torch

def trim_silence(wav, top_db, fft_size, hop_size):
    '''Trim leading and trailing silence
    Useful for M-AILABS datasets if we choose to trim the extra 0.5 silence at beginning and end.
    '''
    #Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per datasets.
    return librosa.effects.trim(wav, top_db=top_db, frame_length=fft_size, hop_length=hop_size)[0]

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_compression_torch(x, clip_val=1e-5, C=1):
    # an audio signal processing operation that reduces the volume of loud sounds or amplifies quiet sounds, 
    # thus reducing or compressing an audio signal's dynamic range.
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes, clip_val, C=1):
    output = dynamic_range_compression_torch(magnitudes, clip_val, C)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, eps, clip_val, center=False):

    global mel_basis, hann_window
    if fmax not in mel_basis: 
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax) # [80, T]
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device) # {'8000_cpu': tensor(80, T)}
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device) # {'cpu': tensor(1024)}

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect') # [1, 1, 8960]
    y = y.squeeze(1) # [1, 8960]

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True) # [1, T, 32, 2]
    spec = torch.sqrt(spec.pow(2).sum(-1) + eps) # [1, T, 32]
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec) # [1, 80, 32] # mel filter bank
    spec = spectral_normalize_torch(spec, clip_val) 

    return spec

def call_generate_melspec(audio_org, sampling_rate, MAX_WAV_VALUE, n_fft, num_mels, hop_size, win_size, fmin, fmax, eps, clip_val):

    audio = audio_org / MAX_WAV_VALUE 
    audio = normalize(audio) * 0.95  
    audio = trim_silence(audio, top_db=20, fft_size=win_size, hop_size=hop_size)  
    audio = torch.FloatTensor(audio).unsqueeze(0) # [wav length]

    mel = mel_spectrogram(audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, eps, clip_val, center=False) # [1, 80, T]

    return mel.squeeze(0).transpose(0,1), audio.squeeze(0) # [T, 80], [wav length]
