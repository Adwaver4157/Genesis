#!/bin/bash

cd /root

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh 

cd /workspace/terrain-generator && conda env create -f environment.yaml
conda activate wfc
pip install -e .
apt-get update && apt-get install -y --no-install-recommends blender
cd /workspace