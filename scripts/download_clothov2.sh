#!/usr/bin/bash

# create folder
mkdir clotho_v2

cd clotho_v2

# download
wget -nc https://zenodo.org/records/4783391/files/clotho_audio_development.7z
wget -nc https://zenodo.org/records/4783391/files/clotho_audio_evaluation.7z
wget -nc https://zenodo.org/records/4783391/files/clotho_audio_validation.7z
wget -nc https://zenodo.org/records/4783391/files/clotho_captions_development.csv
wget -nc https://zenodo.org/records/4783391/files/clotho_captions_evaluation.csv
wget -nc https://zenodo.org/records/4783391/files/clotho_captions_validation.csv
wget -nc https://zenodo.org/records/4783391/files/clotho_metadata_development.csv
wget -nc https://zenodo.org/records/4783391/files/clotho_metadata_evaluation.csv
wget -nc https://zenodo.org/records/4783391/files/clotho_metadata_validation.csv
wget -nc https://zenodo.org/records/4783391/files/LICENSE

# unzip
for f in clotho_audio_development.7z clotho_audio_evaluation.7z clotho_audio_validation.7z
do
  7z x $f
done

rm clotho_audio_development.7z clotho_audio_evaluation.7z clotho_audio_validation.7z

cd ..