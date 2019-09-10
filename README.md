# Speech Embeddings

## Introduction

This is a PyTorch implementation of a self-attentive speaker embedding.

## Dataset

VoxCeleb1 contains over 100,000 utterances for 1,251 celebrities, extracted from videos uploaded to YouTube. 

| |dev|test|
|---|---|---|
|# of speakers|1,211|40|
|# of utterances|148,642|4,874|

Download following files into "data" folder:
- vox1_dev_wav_partaa  
- vox1_dev_wav_partab  
- vox1_dev_wav_partac  
- vox1_dev_wav_partad  
- vox1_test_wav.zip

Then concatenate the files using the command:
```bash
$ cat vox1_dev* > vox1_dev_wav.zip
```

## Dependency

- Python 3.5.2
- PyTorch 1.0.0

## Usage
### Data Pre-processing
Extract data_aishell.tgz:
```bash
$ python extract.py
```


Split dev set to train and valid samples:
```bash
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo
Visualize speaker embeddings from test set:
```bash
$ python visualize.py
```
