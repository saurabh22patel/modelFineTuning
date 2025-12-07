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
python download_dataset.py \
    --dataset_name "${DATASET_NAME:-wikitext}" \
    --output_dir "${DATASET_OUTPUT_DIR:-/home/data}" \
    --tokenizer_path "${TOKENIZER_PATH:-/mnt/data/models}" \
    --model_name "${MODEL_NAME:-meta-llama/Llama-3.1-70B-Instruct}" \
    --max_length "${MAX_LENGTH:-2048}" \
    --split "${SPLIT:-train}" \
    --hf_token "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"

echo "Dataset download completed!"

