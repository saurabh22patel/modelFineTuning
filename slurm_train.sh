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

# This script runs distributed training on 2 nodes with 8 GPUs each (16 GPUs total)

# Create logs directory
mkdir -p logs

# Load necessary modules (adjust based on your cluster)
# module load python/3.10
# module load cuda/11.8
# module load nccl

# Activate virtual environment if using one
# source venv/bin/activate

# Set environment variables for distributed training
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

# NCCL configuration (adjust based on your cluster network)
# For InfiniBand:
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_HCA=mlx5

# For Ethernet (if InfiniBand not available):
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# Optimize for H100 GPUs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_TREE_THRESHOLD=0

# Set PyTorch distributed backend
export TORCH_DISTRIBUTED_BACKEND=nccl

# PyTorch optimizations for H100
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Get node list
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

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run training
srun python train.py \
    --config "${CONFIG_PATH:-config.yaml}" \
    --local_rank $SLURM_LOCALID

echo "Training completed!"

