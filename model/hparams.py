

import random
import numpy as np

# directory setup
output_directory = 'outputs'  # followed argparser
dataset = "VCTK"  # VCTK, NIKL
log_directory = 'RCVC_VCTK'   # followed argparser

# dataset
if dataset == "VCTK":
    dataset_path = 'VCTK_22K'    
    male = ['p226', 'p227', 'p232', 'p237', 'p241', 'p243', 'p245', 'p246', 'p247', 'p251',
            'p252', 'p254', 'p255', 'p256', 'p258', 'p259', 'p260', 'p263', 'p270', 'p271', 
            'p272', 'p273', 'p274', 'p275', 'p278', 'p279', 'p281', 'p284', 'p285', 'p286',
            'p287', 'p292', 'p298', 'p302', 'p304', 'p311', 'p316', 'p326', 'p334', 'p345', 
            'p347', 'p360', 'p363', 'p364', 'p374', 'p376']  
    female = ['p225', 'p228', 'p229', 'p230', 'p231', 'p233', 'p234', 'p236', 'p238', 'p239',
              'p240', 'p244', 'p248', 'p249', 'p250', 'p253', 'p257', 'p261', 'p262', 'p264',
              'p265', 'p266', 'p267', 'p268', 'p269', 'p276', 'p277', 'p280', 'p282', 'p283', 
              'p288', 'p293', 'p294', 'p295', 'p297', 'p299', 'p300', 'p301', 'p303', 'p305', 
              'p306', 'p307', 'p308', 'p310', 'p312', 'p313', 'p314', 'p317', 'p318', 'p323', 
              'p329', 'p330', 'p333', 'p335', 'p336', 'p339', 'p340', 'p341', 'p343', 'p351', 
              'p361', 'p362']

elif dataset == "NIKL":
    dataset_path = 'NIKL_22K'
    male = ['mv01', 'mv02', 'mv03', 'mv04', 'mv05', 'mv06', 'mv07', 'mv08', 'mv09', 'mv10', 
            'mv11', 'mv12', 'mv13', 'mv15', 'mv16', 'mv17', 'mv19', 'mv20', 'mw01', 'mw02',
            'mw03', 'mw04', 'mw05', 'mw06', 'mw07'] 
    female = ['fv01', 'fv02', 'fv03', 'fv04', 'fv05', 'fv06', 'fv07', 'fv08', 'fv09', 'fv10', 
            'fv11', 'fv12', 'fv13', 'fv14', 'fv15', 'fv16', 'fv17', 'fv18', 'fv19', 'fv20',
            'fx01', 'fx02', 'fx03', 'fx04', 'fx05']

speakers = male + female
n_speakers = len(speakers)
speaker_ids = spk2idx = dict(zip(speakers, range(len(speakers))))

# training parameters
lr = 1e-3
train_batch_size = 24
val_batch_size = 2
iters_per_validation = 1000
iters_per_checkpoint = 150000
iters_per_online_inference = 1000
stop_iteration = 150001
seq_len = 192
seed = random.randint(1, 10000)

# multi-processing
dist_backend = "nccl"
dist_url = "tcp://localhost:54321"
world_size = 1
num_workers = 4

# audio parameters
sampling_rate = 22050
MAX_WAV_VALUE = 32768.0
n_fft = 1024
n_mels = 80
hop_size = 256
win_size = 1024
fmin = 0
fmax = 8000
eps = 1e-9
clip_val = 1e-5

# loss
guided_attention_sigma = 0.4
lmabda_att = 100.0
lambda_scale = 10.0
lambda_adv = 0.02

# model
mel_dim = 80
model_dim = 256
hidden_dim = 1024

# style encoder
style_hidden_dim = 256
style_dim = 1024 

# content encoder
prenet_dim = 256
prenet_dropout = 0.2
content_kernel_size = 3
encoder_attn_n_layers = 4
encoder_attn_n_heads = 2
encoder_ffn_dim = 1024
encoder_ffn_dropout = 0.1

# duration predictor
duration_predictor_dim = model_dim
duration_predictor_n_layers = 4
duration_predictor_n_heads = 4
duration_predictor_kernel_size = 3
duration_predictor_dropout_ratio = 0.1

# learnable downsampling
downsampling_trg_kernel_size = [1,1,1,1] # phoneme
downsampling_src_kernel_size = [3,5,7,11] # mel
downsampling_conv_dim = 8 
downsampling_mkb_dim = 16
downsampling_max_seq_len = 200
downsampled_dim = 6

# learnable upsampling
upsampling_trg_kernel_size = [3,5,7,11] # mel
upsampling_src_kernel_size = [1,1,1,1] # phoneme
upsampling_conv_dim = downsampling_conv_dim
upsampling_mkb_dim = downsampling_mkb_dim
upsampling_max_seq_len = 1000
upsampled_dim = duration_predictor_dim

# decoder
decoder_kernel_size = content_kernel_size
decoder_attn_n_layers = encoder_attn_n_layers
decoder_attn_n_heads = encoder_attn_n_heads
decoder_ffn_dim = encoder_ffn_dim
decoder_dropout = encoder_ffn_dropout 

# postnet
postnet_in_dim = mel_dim
postnet_hidden_dim = 256
postnet_n_layers = 5
postnet_kernel_size = 5

# adversarial speaker classifier
adv_speaker_classifier_dim = duration_predictor_dim
adv_speaker_classifier_dropout_ratio = 0.5
adv_speaker_classifier_kernel_size = 3
