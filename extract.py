import zipfile


def extract(filename, folder):
    print('Extracting {}...'.format(filename))
    tar = zipfile.ZipFile(filename, 'r')
    tar.extractall('data/' + folder)
    tar.close()


if __name__ == "__main__":
    extract('data/vox1_dev_wav.zip', 'train')
    extract('data/vox1_test_wav.zip', 'valid')
