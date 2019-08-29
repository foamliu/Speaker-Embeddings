import os
import pickle
import random

from config import wav_folder, data_file


def get_data():
    print('getting data...')

    dirs = [d for d in os.listdir(wav_folder) if d.startswith('id')]

    classes = []
    samples = []

    for id in dirs:
        classes.append(id)
        folder = os.path.join(wav_folder, id)
        sub_folders = [s for s in os.listdir(folder)]
        for sub in sub_folders:
            sub_folder = os.path.join(folder, sub)
            files = [f for f in os.listdir(sub_folder) if f.endswith('.wav')]
            for f in files:
                audiopath = os.path.join(sub_folder, f)
                build_dict(audiopath)
                label = audiopath_to_label[audiopath]
                samples.append({'audiopath': audiopath, 'label': label})

    print('num_files: {}'.format(len(samples)))
    return samples, classes


def build_dict(audiopath):
    global audiopath_to_label
    if not audiopath in audiopath_to_label:
        next_index = len(audiopath_to_label)
        audiopath_to_label[audiopath] = next_index


if __name__ == "__main__":
    audiopath_to_label = {}

    samples, classes = get_data()

    data = dict()
    data['audiopath_to_label'] = audiopath_to_label

    num_valid = 1000
    valid = random.sample(samples, num_valid)
    train = [s for s in samples if s not in valid]

    data['train'] = train
    data['valid'] = valid

    with open(data_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_valid: ' + str(len(data['valid'])))
    print('num_classes: ' + str(len(classes)))

    print(samples[:10])
