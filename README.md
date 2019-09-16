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
Extract vox1_dev_wav.zip & vox1_test_wav.zip:
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

### Performance
Model|Margin-s|Margin-m|Test(%)|Inference speed|
|---|---|---|---|---|
|1|10.0|0.0|84.67%|18.18 ms|



### Demo
Visualize speaker embeddings from test set:
```bash
$ python visualize.py
```

#### Embeddings
![image](https://github.com/foamliu/Speaker-Embeddings/raw/master/images/embeddings.png)


#### Theta j Distribution
![image](https://github.com/foamliu/Speaker-Embeddings/raw/master/images/theta_dist.png)