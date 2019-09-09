import random

import matplotlib
import matplotlib as mpl
from matplotlib import pylab

matplotlib.use('tkagg')
# import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
from config import wav_folder
import config as hp
from models.embedder import GST
from utils import extract_feature, build_LFR_features


def build_dict(id):
    global id_to_label
    if not id in id_to_label:
        next_index = len(id_to_label)
        id_to_label[id] = next_index
    return id_to_label[id]


def get_cmap():
    # x = np.random.rand(20)  # define the data
    # y = np.random.rand(20)  # define the data
    # tag = np.random.randint(0, 20, 20)
    # tag[10:12] = 0  # make sure there are some 0 values to show up as grey

    cmap = pylab.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, 10, 11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


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
    id_to_label = {}

    for id in dirs:
        label = build_dict(id)
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
    # print(labels)

    print('t-SNE: fitting transform...')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(embeddings)

    cmap, norm = get_cmap()

    pylab.figure(figsize=(15, 15))
    # for i, label in enumerate(labels):
    #     x, y = two_d_embeddings[i, :]
    #     print(x)
    #     print(y)
    #     print(label)
    pylab.scatter(two_d_embeddings[:, 0], two_d_embeddings[:, 1], c=labels)
        # pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.show()
