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
from config import test_wav_folder
import config as hp
from models.embedder import GST
from utils import extract_feature


def build_dict(id):
    global id_to_label
    if not id in id_to_label:
        next_index = len(id_to_label)
        id_to_label[id] = next_index
    return id_to_label[id]


def get_cmap():
    cmap = pylab.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, 40, 41)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def get_annotations(two_d, ids):
    speakers = {}
    for i, id in enumerate(ids):
        x, y = two_d[i][0], two_d[i][1]
        if id in speakers:
            speakers[id]['x'].append(x)
            speakers[id]['y'].append(y)
        else:
            speakers[id] = {'x': [], 'y': []}

    xs = []
    ys = []
    ids = []
    for id in speakers.keys():
        xs.append(np.median(speakers[id]['x']))
        ys.append(np.median(speakers[id]['y']))
        ids.append(id)

    # print('xs: ' + str(xs))
    # print('ys: ' + str(ys))
    # print('ids: ' + str(ids))

    return xs, ys, ids


if __name__ == '__main__':
    checkpoint = 'speaker-embeddings.pt'
    print('loading model: {}...'.format(checkpoint))
    model = GST()
    model.load_state_dict(torch.load(checkpoint))
    model = model.to(hp.device)
    model.eval()

    dirs = [d for d in os.listdir(test_wav_folder) if d.startswith('id')]
    # dirs = random.sample(dirs, 20)

    samples = []
    id_to_label = {}

    for id in dirs:
        label = build_dict(id)
        folder = os.path.join(test_wav_folder, id)
        sub_folders = [s for s in os.listdir(folder)]
        for sub in sub_folders:
            sub_folder = os.path.join(folder, sub)
            files = [f for f in os.listdir(sub_folder) if f.endswith('.wav')]
            for f in files:
                audiopath = os.path.join(sub_folder, f)
                samples.append({'audiopath': audiopath, 'label': label, 'id': id})

    num_samples = len(samples)
    print('num_samples: ' + str(num_samples))

    embeddings = np.zeros((num_samples, 512), dtype=np.float)
    dots = []
    labels = []
    ids = []
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            sample = samples[i]
            wave = sample['audiopath']
            mel = extract_feature(input_file=wave, feature='fbank', dim=hp.n_mels, cmvn=True)
            # mel = build_LFR_features(mel, m=hp.LFR_m, n=hp.LFR_n)
            mel = torch.unsqueeze(torch.from_numpy(mel), dim=0)
            mel = mel.to(hp.device)
            feature = model(mel)[0]
            feature = feature.cpu().numpy()
            feature = feature / np.linalg.norm(feature)
            embeddings[i] = feature
            labels.append(sample['label'])
            ids.append(sample['id'])
    # print(labels)

    print('t-SNE: fitting transform...')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(embeddings)

    cmap, norm = get_cmap()

    xs, ys, ids = get_annotations(two_d_embeddings, ids)

    pylab.figure(figsize=(15, 15))
    pylab.scatter(two_d_embeddings[:, 0], two_d_embeddings[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.8,
                  edgecolors='none', s=10)
    for i, id in enumerate(ids):
        pylab.annotate(id, xy=(xs[i], ys[i]), xytext=(0, 0), textcoords='offset points',
                       ha='center', va='center')
    pylab.show()
