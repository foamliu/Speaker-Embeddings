import zipfile


def extract(filename, split):
    print('Extracting {}...'.format(filename))
    tar = zipfile.ZipFile(filename, 'r')
    tar.extractall('data/vox1/' + split)
    tar.close()


if __name__ == "__main__":
    extract('data/vox1_dev_wav.zip', 'dev')
    extract('data/vox1_test_wav.zip', 'test')
