# SALSA :hot_pepper: :tomato: :dancer:
## A Shared Audio–Language Embedding Space for Audio Retrieval

DISCLAIMER: Work in progress :construction_worker: :nut_and_bolt:

This repository contains the code to our submission to the DCASE Workshop & Challenge 2024.

## Demo!
(TODO)

## Setting up the environment
(TODO)

Create environment:
- `conda env create -f environment.yml`
- `CFLAGS='-O3 -march=native' pip install https://github.com/f0k/minimp3py/archive/master.zip`

Download the Datasets & ... 
- (TODO) 

## Test our pre-trained model on the ClothoV2 benachmark

A checkpoint of the model is available here: 
https://cloud.cp.jku.at/index.php/s/ZZkWXQ7f3aXRXYW

Ensemble the checkpoint with this command
```
cat passt_roberta.*.ckpt > passt_roberta.ckpt
```

And then, use this command to test on the ClothoV2 benchmark
```
CUDA_VISIBLE_DEVICES=0 python -m experiments.ex_dcase24 cmd_test_on_clothov2 with \
data_loader.batch_size_eval=32 \
audio_features.segment_length=10 \
audio_features.model=passt \
sentence_features.model=roberta-large \
load_model=passt_roberta.ckpt
```


## Train your own classifier
(TODO)


### Download the datasets
(TODO)

The default directory for datasets is `~/shared`. Change this by setting `directories.data_dir` to the folder that contains the datasets.

Download links for the data sets:
- ClothoV2 https://zenodo.org/records/4743815
- AudioCaps (captions only) https://github.com/cdjkim/audiocaps
- WavCaps https://huggingface.co/datasets/cvssp/WavCaps
- WavCaps Excluded List: https://dcase.community/documents/challenge2024/dcase2024_task6_excluded_freesound_ids.csv


### Training
(TODO)