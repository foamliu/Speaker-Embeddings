import pickle

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import config as hp
from utils import extract_feature


def pad_collate(batch):
    max_input_len = float('-inf')

    for elem in batch:
        feature, label = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]

    for i, elem in enumerate(batch):
        feature, label = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        padded_input[:input_length, :] = feature

        batch[i] = (padded_input, input_length, label)

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


class VoxCeleb1Dataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        with open(hp.data_file, 'rb') as file:
            data = pickle.load(file)

        self.samples = data[split]
        print('loading {} {} samples...'.format(len(self.samples), split))

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = sample['audiopath']
        label = sample['label']

        feature = extract_feature(input_file=wave, feature='fbank', dim=hp.n_mels, cmvn=True)

        return feature, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import torch
    from utils import parse_args
    from tqdm import tqdm

    args = parse_args()
    train_dataset = VoxCeleb1Dataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers,
                                               collate_fn=pad_collate)

    max_len = 0

    for data in tqdm(train_loader):
        feature = data[0]
        # print(feature.shape)
        if feature.shape[1] > max_len:
            max_len = feature.shape[1]

    print('max_len: ' + str(max_len))
