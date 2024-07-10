# SALSA :hot_pepper:
## A Shared Audio–Language Embedding Space for Retrieval



DISCLAIMER: Work in progress :construction_worker: :nut_and_bolt:

This repository contains the code of our submissions [1, 2] to the DCASE Workshop & Challenge ([task 8](https://dcase.community/challenge2024/task-language-based-audio-retrieval) [3]) 2024.

Our system (illustrated in the figure) transforms audio and descriptions into the shared audio–caption embedding space via the audio and description embedding models $\phi_\mathrm{a}$ and $\phi_\mathrm{c}$, respectively. 
In stage 1, we assume that audio $a_i$ and caption $c_j$ do not match if $i \neq j$ and train the model with contrastive loss $L_{\textrm{sup}}$.
Stage 2 uses predictions ensembled from several Stage 1 models (bottom left) to estimate the correspondence between $a_i$ and $c_j$; those estimates then serve as prediction targets instead of the ground truth from stage 1. 
Stage 2 model parameters are initialized with stage 1 parameters, and the corresponding loss is denoted as $L_{\mathrm{dist}}$.

<p align="center">
<img src="figure.png" alt="system illustration" width="400"/>
</p>


## Example
Query: A large gathering of people are talking loudly with each other.
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/344952/simple/small/" width="375" height="30"></iframe>
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/57595/simple/small/" width="375" height="30"></iframe>
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/352819/simple/small/" width="375" height="30"></iframe>
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/88530/simple/small/" width="375" height="30"></iframe>
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/158513/simple/small/" width="375" height="30"></iframe>
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/362353/simple/small/" width="375" height="30"></iframe>
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/179777/simple/small/" width="375" height="30"></iframe>
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/261472/simple/small/" width="375" height="30"></iframe>
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/260062/simple/small/" width="375" height="30"></iframe>
<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/326111/simple/small/" width="375" height="30"></iframe>

## Setting up the environment

Create environment:
- `conda env create -f environment.yml`
- `CFLAGS='-O3 -march=native' pip install https://github.com/f0k/minimp3py/archive/master.zip`

Download ClothoV2 [4]:
- run `scripts/download_clothov2.sh`
- this downloads the dataset into a folder called `clothov2`

## Test our pre-trained model on the ClothoV2 benchmark

A checkpoint of the model is available here: 
https://cloud.cp.jku.at/index.php/s/ZZkWXQ7f3aXRXYW

Download and ensemble the checkpoint with this command
- run `scripts/download_checkpoint.sh`

And then, use this command to test on the ClothoV2 benchmark
```
CUDA_VISIBLE_DEVICES=0 python -m experiments.ex_dcase24 cmd_test_on_clothov2 with \
data_loader.batch_size_eval=32 \
audio_features.segment_length=10 \
audio_features.model=passt \
sentence_features.model=roberta-large \
load_model=passt_roberta.ckpt \
directories.data_dir=.
```

### References
- [1] P. Primus, F. Schmid, and G. Widmer, “Estimated Audio--Caption Correspondences improve Language-Based Audio Retrieval”, under review
- [2] P. Primus, and G. Widmer, “A Knowledge Distillation Approach to Improving Language-Based Audio Retrieval Models,” DCASE2024 Challenge, Tech. Rep., June 2024
- [3] H. Xie, S. Lipping, and T. Virtanen, "Language-Based Audio Retrieval Task in DCASE 2022 Challenge", in Proc. of the Detection and Classification of Acoustic Scenes and Events Workshop, DCASE, Nancy, France, 2022,
- [4] K. Drossos, S. Lipping, and T. Virtanen, “Clotho: an Audio Captioning Dataset,” in Proc. of the IEEE Int. Conf. Acoustic., Speech and Signal Process., ICASSP, Barcelona, Spain, 2020
