#!/usr/bin/bash

# get audioset metadata
wget https://cloud.cp.jku.at/index.php/s/mm8woXSxLokJowc/download/audioset.zip
unzip audioset.zip
rm audioset.zip

# get audiocaps metadata
git clone https://github.com/cdjkim/audiocaps.git audiocaps

# TODO: ask for files -> mail to paul.primus[AT]jku.at
mkdir tmp
cd tmp

wget https://cloud.cp.jku.at/index.php/s/xxxxxx/download/AudioCaps_val_32000.hdf
wget https://cloud.cp.jku.at/index.php/s/xxxxxx/download/AudioCaps_test_32000.hdf
wget https://cloud.cp.jku.at/index.php/s/xxxxxx/download/AudioCaps_train_32000.hdf
