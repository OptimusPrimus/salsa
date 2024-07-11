#!/usr/bin/bash

# create folder
mkdir clotho_v2

cd clotho_v2

# download
wget -nc https://zenodo.org/records/4743815/files/clotho_audio_development.7z
wget -nc https://zenodo.org/records/4743815/files/clotho_audio_evaluation.7z
wget -nc https://zenodo.org/records/4743815/files/clotho_audio_validation.7z
wget -nc https://zenodo.org/records/4743815/files/clotho_captions_development.csv
wget -nc https://zenodo.org/records/4743815/files/clotho_captions_evaluation.csv
wget -nc https://zenodo.org/records/4743815/files/clotho_captions_validation.csv
wget -nc https://zenodo.org/records/4743815/files/clotho_metadata_development.csv
wget -nc https://zenodo.org/records/4743815/files/clotho_metadata_evaluation.csv
wget -nc https://zenodo.org/records/4743815/files/clotho_metadata_validation.csv
wget -nc https://zenodo.org/records/4743815/files/LICENSE

# unzip
for f in *.7z; 7z x $f

rm *.7z