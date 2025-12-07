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

mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/load_config.sh" ]; then
    source "$SCRIPT_DIR/load_config.sh" "${CONFIG_PATH:-$SCRIPT_DIR/config.yaml}"
fi

if [ -n "$VENV_PATH" ]; then
    if [ -f "$VENV_PATH" ]; then
        source "$VENV_PATH"
    else
        echo "Error: VENV_PATH specified but not found: $VENV_PATH"
        exit 1
    fi
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -f "$HOME/llmtune/bin/activate" ]; then
    source "$HOME/llmtune/bin/activate"
else
    echo "Error: No virtual environment found"
    exit 1
fi
python download_model.py \
    --model_name "${MODEL_NAME:-meta-llama/Llama-3.1-70B-Instruct}" \
    --output_dir "${MODEL_OUTPUT_DIR:-/mnt/data/models}" \
    --cache_dir "${HF_CACHE_DIR:-./cache}" \
    --hf_token "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"

echo "Model download completed!"

