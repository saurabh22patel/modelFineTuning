#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --output=logs/download_model_%j.out
#SBATCH --error=logs/download_model_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Create logs directory
mkdir -p logs

# Load necessary modules (adjust based on your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment if using one
# source venv/bin/activate

# Install dependencies if needed
# pip install -r requirements.txt

# Download model
python download_model.py \
    --model_name "${MODEL_NAME:-meta-llama/Llama-2-7b-hf}" \
    --output_dir "${MODEL_OUTPUT_DIR:-./models}" \
    --cache_dir "${HF_CACHE_DIR:-./cache}"

echo "Model download completed!"

