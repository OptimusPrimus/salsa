# SALSA :hot_pepper: :tomato: :dancer: - a Shared Audio–Language embedding Space for Audio retrieval

- state of the art

This repository contains the code to our submission to the DCASE Workshop & Challenge 2024.


## Demo!


## Use our pre-trained model



## Train your own classifier


### Setting up the environment
Create environment:
- `conda env create -f environment.yml`
- `pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.24#egg=hear21passt' `
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
CUDA_VISIBLE_DEVICES=0 python -m ... seed= 

```

create embeddings
```
CUDA_VISIBLE_DEVICES=0 python -m ... seed= 
```

stage 2
```
CUDA_VISIBLE_DEVICES=0 python -m ... seed= 
```
