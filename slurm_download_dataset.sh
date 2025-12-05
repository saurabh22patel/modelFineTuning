#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --output=logs/download_dataset_%j.out
#SBATCH --error=logs/download_dataset_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Create logs directory
mkdir -p logs

# Load necessary modules (adjust based on your cluster)
# module load python/3.10

# Activate virtual environment if using one
# source venv/bin/activate

# Download dataset
python download_dataset.py \
    --dataset_name "${DATASET_NAME:-wikitext}" \
    --output_dir "${DATASET_OUTPUT_DIR:-./datasets}" \
    --tokenizer_path "${TOKENIZER_PATH:-./models}" \
    --max_length "${MAX_LENGTH:-2048}" \
    --split "${SPLIT:-train}"

echo "Dataset download completed!"

