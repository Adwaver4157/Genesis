#!/bin/bash

cd /root

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh 

cd /workspace/terrain-generator && conda env create -f environment.yaml
pip install -e .

cd /workspace