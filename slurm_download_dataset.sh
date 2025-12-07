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

# Load configuration from config.yaml (sets VENV_PATH if configured)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/load_config.sh" ]; then
    source "$SCRIPT_DIR/load_config.sh" "${CONFIG_PATH:-$SCRIPT_DIR/config.yaml}"
fi

# Load necessary modules (adjust based on your cluster)
# module load python/3.10

# Activate virtual environment
# Priority: 1) VENV_PATH from config.yaml, 2) VENV_PATH env var, 3) project-local venv, 4) ~/llmtune, 5) fail
if [ -n "$VENV_PATH" ]; then
    # Use venv path from config.yaml or environment variable
    if [ -f "$VENV_PATH" ]; then
        source "$VENV_PATH"
        echo "Activated virtual environment from config: $VENV_PATH"
    else
        echo "Error: VENV_PATH specified but not found: $VENV_PATH"
        echo "Please check your config.yaml environment.venv_path setting"
        exit 1
    fi
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    # Use project-local venv (relative to script directory)
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "Activated project-local virtual environment"
elif [ -f "$HOME/llmtune/bin/activate" ]; then
    # Fallback to ~/llmtune venv (common on SLURM clusters)
    source "$HOME/llmtune/bin/activate"
    echo "Activated fallback virtual environment: ~/llmtune"
else
    echo "Error: No virtual environment found. Please either:"
    echo "  1. Set environment.venv_path in config.yaml"
    echo "  2. Set VENV_PATH environment variable before submitting job"
    echo "  3. Create a project-local venv: python3 -m venv venv"
    echo "  4. Ensure ~/llmtune/bin/activate exists"
    exit 1
fi

# Download dataset with pre-tokenization (HF_TOKEN can be set as environment variable or in config.yaml)
python download_dataset.py \
    --dataset_name "${DATASET_NAME:-wikitext}" \
    --output_dir "${DATASET_OUTPUT_DIR:-/root/data}" \
    --tokenizer_path "${TOKENIZER_PATH:-/mnt/data/models}" \
    --model_name "${MODEL_NAME:-meta-llama/Llama-3.1-70B-Instruct}" \
    --max_length "${MAX_LENGTH:-2048}" \
    --split "${SPLIT:-train}" \
    --hf_token "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"

echo "Dataset download completed!"

