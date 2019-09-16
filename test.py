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
            files = [os.path.join(root, file) for file in files]
            file_list += files
        pair = random.sample(file_list, 2)

    with open('data/test_pairs.txt', 'w') as file:
        file.writelines(out_lines)

    #     while len([f for f in os.listdir(os.path.join(IMG_DIR, folder)) if
    #                f.endswith('.jpg') and not f.endswith('0.jpg')]) < 1:
    #         folder = random.choice(dirs)
    #
    #     files = [f for f in os.listdir(os.path.join(IMG_DIR, folder)) if f.endswith('.jpg') and not f.endswith('0.jpg')]
    #     file_1 = random.choice(files)
    #     file_0 = os.path.join(folder, '0.jpg').replace('\\', '/')
    #     file_1 = os.path.join(folder, file_1).replace('\\', '/')
    #     out_lines.append('{} {} {}\n'.format(file_0, file_1, 1))
    #     exclude_list.add(file_0)
    #     exclude_list.add(file_1)
    #
    # for _ in tqdm(range(num_not_same)):
    #     dirs = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
    #     folders = random.sample(dirs, 2)
    #     while len([f for f in os.listdir(os.path.join(IMG_DIR, folders[0])) if
    #                f.endswith('.jpg') and not f.endswith('0.jpg')]) < 1 or len(
    #         [f for f in os.listdir(os.path.join(IMG_DIR, folders[1])) if
    #          f.endswith('.jpg') and not f.endswith('0.jpg')]) < 1:
    #         folders = random.sample(dirs, 2)
    #
    #     file_0 = folders[0] + '/' + '0.jpg'
    #     file_1 = pick_one_file(folders[1])
    #     out_lines.append('{} {} {}\n'.format(file_0, file_1, 0))
    #     exclude_list.add(os.path.join(file_0))
    #     exclude_list.add(os.path.join(file_1))
    #
    # with open('data/test_pairs.txt', 'w') as file:
    #     file.writelines(out_lines)
    #
    # print(exclude_list)
    #
    # samples = get_data()
    # filtered = []
    # for item in samples:
    #     if item['img'] not in exclude_list:
    #         filtered.append(item)
    #
    # print(len(filtered))
    # print(filtered[:10])
    #
    # with open(pickle_file, 'wb') as file:
    #     pickle.dump(filtered, file)
