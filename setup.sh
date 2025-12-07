#!/bin/bash

set -e

echo "Setting up fine-tuning environment..."

mkdir -p logs
mkdir -p mlruns
mkdir -p cache

echo "Creating directories in /mnt/data..."
mkdir -p /mnt/data/models
mkdir -p /mnt/data/models/checkpoints

echo "Creating dataset directory in /home..."
mkdir -p /home/data

chmod +x download_model.py
chmod +x download_dataset.py
chmod +x train.py
chmod +x monitor_gpu.py
chmod +x slurm_*.sh

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml with your model and dataset settings"
echo "2. Submit model download job: sbatch slurm_download_model.sh"
echo "3. Submit dataset download job: sbatch slurm_download_dataset.sh"
echo "4. Submit training job: sbatch slurm_train.sh"

