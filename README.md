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

Download the ClothoV2
- run `scripts/download_clothov2.sh` and copy resulting folder (called `clothov2`) to desired location
- (TODO) same for AudioCaps and WavCaps

## Test our pre-trained model on the ClothoV2 benchmark

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