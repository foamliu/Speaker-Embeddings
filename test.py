import math
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

import config as hp
from models.embedder import GST
from utils import extract_feature

test_file = 'data/test_pairs.txt'
angles_file = 'data/angles.txt'


def gen_test_pairs():
    num_tests = 6000

    num_same = int(num_tests / 2)
    num_not_same = num_tests - num_same

    out_lines = []

    for _ in tqdm(range(num_same)):
        dirs = [d for d in os.listdir(hp.test_wav_folder) if os.path.isdir(os.path.join(hp.test_wav_folder, d))]
        folder = random.choice(dirs)
        folder = os.path.join(hp.test_wav_folder, folder)
        file_list = []
        for root, dir, files in os.walk(folder):
            files = [os.path.join(root, file) for file in files if file.endswith('.wav')]
            file_list += files
        pair = random.sample(file_list, 2)
        out_lines.append('{} {} {}\n'.format(pair[0], pair[1], 1))

    for _ in tqdm(range(num_not_same)):
        dirs = [d for d in os.listdir(hp.test_wav_folder) if os.path.isdir(os.path.join(hp.test_wav_folder, d))]
        folders = [os.path.join(hp.test_wav_folder, folder) for folder in random.sample(dirs, 2)]
        file_list_0 = []
        for root, dir, files in os.walk(folders[0]):
            files = [os.path.join(root, file) for file in files if file.endswith('.wav')]
            file_list_0 += files
        file_0 = random.choice(file_list_0)
        file_list_1 = []
        for root, dir, files in os.walk(folders[1]):
            files = [os.path.join(root, file) for file in files if file.endswith('.wav')]
            file_list_1 += files
        file_1 = random.choice(file_list_1)
        out_lines.append('{} {} {}\n'.format(file_0, file_1, 0))

    with open(test_file, 'w') as file:
        file.writelines(out_lines)


def evaluate(model):
    with open(test_file, 'r') as file:
        lines = file.readlines()

    angles = []

    start = time.time()
    with torch.no_grad():
        for line in tqdm(lines):
            tokens = line.split()
            file0 = tokens[0]
            mel0 = extract_feature(input_file=file0, feature='fbank', dim=hp.n_mels, cmvn=True)
            mel0 = torch.unsqueeze(torch.from_numpy(mel0), dim=0)
            mel0 = mel0.to(hp.device)
            output = model(mel0)[0]
            feature0 = output.cpu().numpy()

            file1 = tokens[1]
            mel1 = extract_feature(input_file=file1, feature='fbank', dim=hp.n_mels, cmvn=True)
            mel1 = torch.unsqueeze(torch.from_numpy(mel1), dim=0)
            mel1 = mel1.to(hp.device)
            output = model(mel1)[0]
            feature1 = output.cpu().numpy()

            x0 = feature0 / np.linalg.norm(feature0)
            x1 = feature1 / np.linalg.norm(feature1)
            cosine = np.dot(x0, x1)
            theta = math.acos(cosine)
            theta = theta * 180 / math.pi
            is_same = tokens[2]
            angles.append('{} {}\n'.format(theta, is_same))

    elapsed_time = time.time() - start
    print('elapsed time(sec) per image: {}'.format(elapsed_time / (6000 * 2)))

    with open(angles_file, 'w') as file:
        file.writelines(angles)


def get_threshold():
    with open(angles_file, 'r') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'angle': angle, 'type': type})

    min_error = 6000
    min_threshold = 0

    for d in data:
        threshold = d['angle']
        type1 = len([s for s in data if s['angle'] <= threshold and s['type'] == 0])
        type2 = len([s for s in data if s['angle'] > threshold and s['type'] == 1])
        num_errors = type1 + type2
        if num_errors < min_error:
            min_error = num_errors
            min_threshold = threshold

    # print(min_error, min_threshold)
    return min_threshold


def accuracy(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    wrong = 0
    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            if angle > threshold:
                wrong += 1
        else:
            if angle <= threshold:
                wrong += 1

    accuracy = 1 - wrong / 6000
    return accuracy


if __name__ == "__main__":
    checkpoint = 'speaker-embeddings.pt'
    print('loading model: {}...'.format(checkpoint))
    model = GST()
    model.load_state_dict(torch.load(checkpoint))
    model = model.to(hp.device)
    model.eval()

    print('Evaluating {}...'.format(angles_file))
    evaluate(model)

    print('Calculating threshold...')
    thres = get_threshold()

    print('Calculating accuracy...')
    acc = accuracy(thres)
    print('Accuracy: {}%, threshold: {}'.format(acc * 100, thres))
