#!/bin/bash

: '
chmod +x 00_setup.sh
./00_setup.sh
'

set -e
deactivate || true  # Ignore errors
rm -rf drone-ai-env
python3.11 -m venv drone-ai-env
source drone-ai-env/bin/activate

pip list
pip install --upgrade pip 
pip install gymnasium==0.28.1
pip install gym
pip install pybullet
pip install stable-baselines3
pip install 'shimmy>=2.0'
pip install tensorboard

echo "Setup complete! Virtual environment 'drone-ai-env' is ready."

