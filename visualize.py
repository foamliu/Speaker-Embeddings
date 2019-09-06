import pickle

import torch

import config as hp
from models.embedder import GST
from utils import extract_feature, build_LFR_features

if __name__ == '__main__':
    checkpoint = 'speaker-embeddings.pt'
    print('loading model: {}...'.format(checkpoint))
    model = GST()
    model.load_state_dict(torch.load(checkpoint))
    model = model.to(hp.device)
    model.eval()

    with open(hp.data_file, 'rb') as file:
        data = pickle.load(file)
    samples = data['valid']

    for sample in samples:
        wave = sample['audiopath']
        label = sample['label']
        feature = extract_feature(input_file=wave, feature='fbank', dim=hp.n_mels, cmvn=True)
        feature = build_LFR_features(feature, m=hp.LFR_m, n=hp.LFR_n)
        padded_input = torch.unsqueeze(torch.from_numpy(feature), dim=0)
        padded_input = padded_input.to(hp.device)
        feature = model(padded_input)
        print(feature)
