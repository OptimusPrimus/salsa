#!/usr/bin/bash

# create folder

python -c """
from huggingface_hub import snapshot_download

snapshot_download(repo_id='cvssp/WavCaps', repo_type='dataset')
"""
