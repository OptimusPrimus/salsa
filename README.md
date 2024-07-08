# SALSA :hot_pepper: :tomato: :dancer:
## A Shared Audio–Language Embedding Space for Audio Retrieval

DISCLAIMER: Work in progress

This repository contains the code to our submission to the DCASE Workshop & Challenge 2024.


## Demo!


## Test our pre-trained model

A checkpoint of the model is available here: 
https://cloud.cp.jku.at/index.php/s/ZZkWXQ7f3aXRXYW

```
CUDA_VISIBLE_DEVICES=2 python -m experiments.ex_dcase24 cmd_test_on_clothov2 with \
data_loader.batch_size_eval=32 \
audio_features.segment_length=10 \
audio_features.model=passt \
sentence_features.model=roberta-large \
load_parameters=volcanic-planet-149
```


## Train your own classifier


### Setting up the environment
Create environment:
- `conda env create -f environment.yml`
- `CFLAGS='-O3 -march=native' pip install https://github.com/f0k/minimp3py/archive/master.zip`

### Download the datasets

The default directory for datasets is `~/shared`. Change this by setting `directories.data_dir` to the folder that contains the datasets.

Download links for the data sets:
- ClothoV2 https://zenodo.org/records/4743815
- AudioCaps (captions only) https://github.com/cdjkim/audiocaps
- WavCaps https://huggingface.co/datasets/cvssp/WavCaps
- WavCaps Excluded List: https://dcase.community/documents/challenge2024/dcase2024_task6_excluded_freesound_ids.csv


### Training

stage 1
```
CUDA_VISIBLE_DEVICES=0 python -m ... seed=144272510 

```

create embeddings
```
CUDA_VISIBLE_DEVICES=0 python -m ... seed=144272510 
```

stage 2
```
CUDA_VISIBLE_DEVICES=0 python -m ... seed=144272510
```
