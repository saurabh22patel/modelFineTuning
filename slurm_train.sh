#!/bin/bash
#SBATCH --job-name=llm_finetune
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --partition=main
#SBATCH --exclusive

set -e

# Configuration
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_PATH="/root/llmtune/venv/bin/activate"
CONFIG_FILE="config.yaml"
MASTER_PORT=29500

# Optional: Resume from checkpoint
# Usage: sbatch slurm_train.sh [checkpoint_path|latest]
# Example: sbatch slurm_train.sh latest
# Example: sbatch slurm_train.sh /path/to/checkpoint-1000
RESUME_CHECKPOINT="${1:-}"

# Setup
mkdir -p logs
cd "$PROJECT_DIR"
source "$VENV_PATH"

# Distributed training setup
export TORCH_DISTRIBUTED_BACKEND=nccl
export PYTHONUNBUFFERED=1

# NCCL timeout settings to prevent watchdog timeouts
# Increase heartbeat timeout to handle long-running operations (default is 480s)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# Optional: Uncomment the line below to disable NCCL monitoring if timeout issues persist
# export TORCH_NCCL_ENABLE_MONITORING=0

# Get master node IP
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=$(getent hosts $MASTER_NODE 2>/dev/null | awk '{print $1}' | head -n 1)
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=$MASTER_NODE

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Resolve config path
[ "${CONFIG_FILE#/}" = "$CONFIG_FILE" ] && CONFIG_FILE="$PROJECT_DIR/$CONFIG_FILE"

# Verify files
[ ! -f "$PROJECT_DIR/train.py" ] && echo "ERROR: train.py not found" && exit 1
[ ! -f "$CONFIG_FILE" ] && echo "ERROR: Config file not found: $CONFIG_FILE" && exit 1

# Build training command
TRAIN_CMD="python -u $PROJECT_DIR/train.py --config $CONFIG_FILE"
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

# Launch training
srun --ntasks=$SLURM_NTASKS $TRAIN_CMD
