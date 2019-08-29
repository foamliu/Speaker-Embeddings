import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = zipfile.ZipFile(filename, 'r')
    tar.extractall('data/vox1')
    tar.close()


if __name__ == "__main__":
    extract('data/vox1_dev_wav.zip')
    extract('data/vox1_test_wav.zip')
