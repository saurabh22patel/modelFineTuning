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

# Load configuration from config.yaml (sets VENV_PATH if configured)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/load_config.sh" ]; then
    source "$SCRIPT_DIR/load_config.sh" "${CONFIG_PATH:-$SCRIPT_DIR/config.yaml}"
fi

# Load necessary modules (adjust based on your cluster)
# module load python/3.10
# module load cuda/11.8

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

# Install dependencies if needed
# pip install -r requirements.txt

# Download model (HF_TOKEN can be set as environment variable or in config.yaml)
python download_model.py \
    --model_name "${MODEL_NAME:-meta-llama/Llama-3.1-70B-Instruct}" \
    --output_dir "${MODEL_OUTPUT_DIR:-/mnt/data/models}" \
    --cache_dir "${HF_CACHE_DIR:-./cache}" \
    --hf_token "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"

echo "Model download completed!"

