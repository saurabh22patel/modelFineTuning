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
#SBATCH --partition=gpu
#SBATCH --exclusive

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

export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_TREE_THRESHOLD=0
export TORCH_DISTRIBUTED_BACKEND=nccl
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES=($NODELIST)
MASTER_NODE=${NODES[0]}
MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)

export MASTER_ADDR=$MASTER_ADDR
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODEID"
echo "Master Node: $MASTER_NODE"
echo "Master Addr: $MASTER_ADDR"
echo "World Size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Local Rank: $LOCAL_RANK"
echo "========================================="

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
srun python train.py \
    --config "${CONFIG_PATH:-$SCRIPT_DIR/config.yaml}" \
    --local_rank $SLURM_LOCALID

echo "Training completed!"

