import pickle
from matplotlib import pylab
import matplotlib
import torch.nn.functional as F
import random
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
from config import wav_folder
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

    dirs = [d for d in os.listdir(wav_folder) if d.startswith('id')]
    dirs = random.sample(dirs, 10)

    samples = []

    for id in dirs:
        label = id
        folder = os.path.join(wav_folder, id)
        sub_folders = [s for s in os.listdir(folder)]
        for sub in sub_folders:
            sub_folder = os.path.join(folder, sub)
            files = [f for f in os.listdir(sub_folder) if f.endswith('.wav')]
            for f in files:
                audiopath = os.path.join(sub_folder, f)
                samples.append({'audiopath': audiopath, 'label': label})

    num_samples = len(samples)
    print('num_samples: ' + str(num_samples))

    embeddings = np.zeros((num_samples, 512), dtype=np.float)
    dots = []
    labels = []
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            sample = samples[i]
            wave = sample['audiopath']
            label = sample['label']
            mel = extract_feature(input_file=wave, feature='fbank', dim=hp.n_mels, cmvn=True)
            mel = build_LFR_features(mel, m=hp.LFR_m, n=hp.LFR_n)
            mel = torch.unsqueeze(torch.from_numpy(mel), dim=0)
            mel = mel.to(hp.device)
            feature = model(mel)[0]
            feature = feature.cpu().numpy()
            feature = feature / np.linalg.norm(feature)
            embeddings[i] = feature
            labels.append(label)
    print(labels)

    print('t-SNE: fitting transform...')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(embeddings)

    # x = []
    # y = []
    # for i in range(1000):
    #     x.append(two_d_embeddings[i][0])
    #     y.append(two_d_embeddings[i][1])
    # plt.scatter(x, y, c=labels)

    pylab.figure(figsize=(15, 15))
    for i, label in enumerate(labels):
        x, y = two_d_embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')

    # plt.annotate(labels, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
