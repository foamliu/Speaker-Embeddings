import os
import random

from tqdm import tqdm

import config as hp

if __name__ == "__main__":
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

    with open('data/test_pairs.txt', 'w') as file:
        file.writelines(out_lines)
