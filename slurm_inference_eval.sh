#!/bin/bash
#SBATCH --job-name=inference_eval
#SBATCH --output=logs/inference_eval_%j.out
#SBATCH --error=logs/inference_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment
source ~/llmtune/bin/activate 2>/dev/null || source ./venv/bin/activate 2>/dev/null || true

# Set environment variables
export HF_TOKEN=${HF_TOKEN:-""}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-""}
export MLFLOW_USERNAME=${MLFLOW_USERNAME:-""}
export MLFLOW_PASSWORD=${MLFLOW_PASSWORD:-""}

# Default arguments
CONFIG=${CONFIG:-"config.yaml"}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-""}
FINE_TUNED_MODEL_PATH=${FINE_TUNED_MODEL_PATH:-""}
NUM_PROMPTS=${NUM_PROMPTS:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
CALCULATE_PPL=${CALCULATE_PPL:-"false"}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-""}
RUN_NAME=${RUN_NAME:-""}

# Build command
CMD="python inference_eval.py --config $CONFIG"

if [ ! -z "$BASE_MODEL_PATH" ]; then
    CMD="$CMD --base_model_path $BASE_MODEL_PATH"
fi

if [ ! -z "$FINE_TUNED_MODEL_PATH" ]; then
    CMD="$CMD --fine_tuned_model_path $FINE_TUNED_MODEL_PATH"
fi

if [ ! -z "$NUM_PROMPTS" ]; then
    CMD="$CMD --num_prompts $NUM_PROMPTS"
fi

if [ ! -z "$MAX_NEW_TOKENS" ]; then
    CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
fi

if [ "$CALCULATE_PPL" = "true" ]; then
    CMD="$CMD --calculate_perplexity"
fi

if [ ! -z "$EVAL_DATASET_PATH" ]; then
    CMD="$CMD --eval_dataset_path $EVAL_DATASET_PATH"
fi

if [ ! -z "$RUN_NAME" ]; then
    CMD="$CMD --run_name $RUN_NAME"
fi

echo "=========================================="
echo "Running Inference Evaluation"
echo "=========================================="
echo "Config: $CONFIG"
echo "Base Model: $BASE_MODEL_PATH"
echo "Fine-tuned Model: $FINE_TUNED_MODEL_PATH"
echo "Number of Prompts: $NUM_PROMPTS"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Calculate Perplexity: $CALCULATE_PPL"
echo "=========================================="
echo "Command: $CMD"
echo "=========================================="

# Run the evaluation
$CMD

echo "=========================================="
echo "Inference evaluation completed!"
echo "=========================================="

