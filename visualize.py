import pickle

import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

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

    embeddings = np.zeros((1000, 512), dtype=np.float)
    dots = []
    labels = []
    with torch.no_grad():
        for i in tqdm(range(len(samples))):
            sample = samples[i]
            wave = sample['audiopath']
            label = sample['label']
            feature = extract_feature(input_file=wave, feature='fbank', dim=hp.n_mels, cmvn=True)
            feature = build_LFR_features(feature, m=hp.LFR_m, n=hp.LFR_n)
            padded_input = torch.unsqueeze(torch.from_numpy(feature), dim=0)
            padded_input = padded_input.to(hp.device)
            feature = model(padded_input)[0][0]
            feature = feature.cpu().numpy()
            embeddings[i] = np.linalg.norm(feature)
            labels.append(label)

    print('t-SNE: fitting transform...')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(embeddings)

    x = []
    y = []
    for i in range(1000):
        x.append(two_d_embeddings[i][0])
        y.append(two_d_embeddings[i][1])
    plt.scatter(x, y, c=labels, alpha=0.5)
    plt.show()
