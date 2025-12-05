#!/bin/bash
# Setup script for fine-tuning environment

set -e

echo "Setting up fine-tuning environment..."

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p datasets
mkdir -p checkpoints
mkdir -p mlruns
mkdir -p cache

# Make scripts executable
echo "Making scripts executable..."
chmod +x download_model.py
chmod +x download_dataset.py
chmod +x train.py
chmod +x monitor_gpu.py
chmod +x slurm_*.sh

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml with your model and dataset settings"
echo "2. Submit model download job: sbatch slurm_download_model.sh"
echo "3. Submit dataset download job: sbatch slurm_download_dataset.sh"
echo "4. Submit training job: sbatch slurm_train.sh"

