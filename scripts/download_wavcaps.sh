#!/usr/bin/bash

python -c """
from huggingface_hub import snapshot_download

snapshot_download(repo_id='cvssp/WavCaps', repo_type='dataset', local_dir='./wavcaps')
"""
cd wavcaps

zip -F Zip_files/AudioSet_SL/AudioSet_SL.zip --out AS.zip
zip -F Zip_files/BBC_Sound_Effects/BBC_Sound_Effects.zip --out BBC.zip
zip -F Zip_files/FreeSound/FreeSound.zip --out FreeSound.zip
zip -F Zip_files/SoundBible/SoundBible.zip --out SoundBible.zip

rm -r Zip_files
unzip AS.zip
unzip BBC.zip
unzip FreeSound.zip
unzip SoundBible.zip

rm -r *.zip
mkdir audio
mv mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac audio/AudioSet_SL
mv mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/FreeSound_flac audio/FreeSound
mv mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/BBC_Sound_Effects_flac audio/BBC_Sound_Effects
mv mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/SoundBible_flac audio/SoundBible

rm -r mnt


