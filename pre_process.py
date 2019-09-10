import os
import pickle
import random

from config import dev_wav_folder, data_file


def get_data():
    print('getting data...')

    dirs = [d for d in os.listdir(dev_wav_folder) if d.startswith('id')]

    samples = []

    for id in dirs:
        label = build_dict(id)
        folder = os.path.join(dev_wav_folder, id)
        sub_folders = [s for s in os.listdir(folder)]
        for sub in sub_folders:
            sub_folder = os.path.join(folder, sub)
            files = [f for f in os.listdir(sub_folder) if f.endswith('.wav')]
            for f in files:
                audiopath = os.path.join(sub_folder, f)
                samples.append({'audiopath': audiopath, 'label': label})

    print('num_files: {}'.format(len(samples)))
    return samples


def build_dict(id):
    global id_to_label
    if not id in id_to_label:
        next_index = len(id_to_label)
        id_to_label[id] = next_index
    return id_to_label[id]


if __name__ == "__main__":
    id_to_label = {}

    samples = get_data()

    data = dict()
    data['id_to_label'] = id_to_label

    num_valid = 1000
    valid = random.sample(samples, num_valid)
    train = [s for s in samples if s not in valid]

    data['train'] = train
    data['valid'] = valid

    with open(data_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_valid: ' + str(len(data['valid'])))
    print('num_classes: ' + str(len(id_to_label)))

    print(samples[:10])
