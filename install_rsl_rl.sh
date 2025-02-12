#!/bin/bash

# Install rsl_rl.
cd /root/
git clone https://github.com/Adwaver4157/rsl_rl.git
cd rsl_rl && git checkout feature/add_vis_encoder && pip install -e .

# Install tensorboard.
pip install tensorboard

cd /workspace