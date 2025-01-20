#!/bin/bash

# Install rsl_rl.
cd /root/
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .

# Install tensorboard.
pip install tensorboard

cd /workspace