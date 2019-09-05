import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
n_mels = 80  # dimension of feature
window_size = 25  # window size for FFT (ms)
stride = 10  # window stride for FFT (ms)
hidden_size = 512
embedding_dim = 512
cmvn = True  # apply CMVN on feature
LFR_m = 1  # change to 4 if use LFR
LFR_n = 3

# Reference encoder
ref_enc_filters = [32, 32, 64, 64, 128, 128]

# Style token layer
token_num = 1000
token_emb_size = 512
num_heads = 8

# Training parameters
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 50  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
sample_rate = 16000  # vox1

# Data parameters
num_train = 148642
num_valid = 1000

num_classes = 1251

DATA_DIR = 'data'
vox1_folder = 'data/vox1'
wav_folder = os.path.join(vox1_folder, 'wav')
data_file = 'data/vox1.pickle'
