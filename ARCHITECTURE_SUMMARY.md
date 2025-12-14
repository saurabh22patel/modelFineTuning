# Training Architecture Summary

## Quick Overview

This system implements **distributed fine-tuning of Large Language Models** using **PyTorch FSDP (Fully Sharded Data Parallel)** across multiple nodes and GPUs.

## Core Architecture

### Training Paradigm
- **Framework**: PyTorch FSDP (Fully Sharded Data Parallel)
- **Hardware**: 2 nodes × 8 H100 GPUs = 16 GPUs total
- **Communication**: NCCL backend with InfiniBand for inter-node, NVLink for intra-node
- **Precision**: bfloat16 (BF16) mixed precision

### Key Components

1. **Model Sharding (FSDP)**
   - Model parameters, gradients, and optimizer states sharded across all GPUs
   - Each GPU holds 1/16th of the model
   - Parameters gathered only when needed (forward/backward pass)
   - **Memory Benefit**: 7B model needs ~0.875GB per GPU (vs ~14GB without FSDP)

2. **Data Pipeline**
   - Pre-tokenized datasets for faster loading
   - DistributedSampler shards data across GPUs
   - Multi-worker DataLoader with prefetching
   - Persistent workers to avoid restart overhead

3. **Training Loop**
   - HuggingFace Trainer with CustomTrainer extension
   - GPU utilization monitoring
   - MLflow integration for experiment tracking
   - Optimized checkpoint saving for FSDP

4. **Memory Management**
   - Gradient checkpointing (trade compute for memory)
   - Periodic CUDA cache clearing
   - FSDP sharding reduces per-GPU memory by ~16x
   - Mixed precision (BF16) reduces memory by 50%

## Training Flow

```
1. Initialization
   ├── Setup distributed (NCCL, process groups)
   ├── Load model (CPU → FSDP wrap → GPU)
   ├── Load dataset (pre-tokenized)
   └── Initialize trainer

2. Training Loop
   ├── Forward pass (FSDP gathers parameters)
   ├── Loss computation
   ├── Backward pass (FSDP shards gradients)
   ├── Gradient synchronization (all-reduce)
   ├── Optimizer step
   └── Logging (MLflow)

3. Checkpointing (periodic)
   ├── Barrier synchronization
   ├── State dict gathering (rank 0 only)
   └── Save checkpoint (safetensors)

4. Finalization
   ├── Final checkpoint save
   ├── Model consolidation
   └── Cleanup
```

## Key Design Decisions

### Why FSDP?
- **Memory Efficiency**: Enables training large models on limited GPU memory
- **Scalability**: Works seamlessly across multiple nodes
- **Performance**: Maintains high GPU utilization with proper configuration

### Why CPU-First Model Loading?
- Prevents OOM during model initialization
- Allows FSDP to handle GPU placement
- More reliable for large models

### Why Pre-Tokenized Datasets?
- Faster training startup
- Consistent tokenization
- Reduces CPU overhead during training

### Why CustomTrainer?
- GPU utilization monitoring
- Optimized checkpoint saving for FSDP
- Better error handling
- MLflow integration

## Performance Characteristics

### Memory Usage (7B model, BF16)
- **Without FSDP**: ~58-60GB per GPU
- **With FSDP**: ~5.5-7GB per GPU
- **Savings**: ~90% reduction per GPU

### Communication Patterns
- **Intra-node**: NVLink (high bandwidth, low latency)
- **Inter-node**: InfiniBand (high bandwidth, low latency)
- **Collective Ops**: All-reduce (gradients), All-gather (parameters)

### GPU Utilization Target
- **Target**: 80%+ utilization
- **Optimization**: Batch size, gradient accumulation, DataLoader workers

## Configuration Highlights

### Key Settings in `config.yaml`
```yaml
fsdp:
  sharding_strategy: "FULL_SHARD"  # Maximum memory efficiency
  cpu_offload: false               # Keep on GPU for NVLink performance
  mixed_precision: true           # BF16 for memory and speed

dataset:
  batch_size_per_device: 10       # Per-GPU batch size
  gradient_accumulation_steps: 3  # Effective batch = 10 × 3 × 16 = 480
  max_length: 2048                # Sequence length

performance:
  dataloader_num_workers: 8       # Parallel data loading
  prefetch_factor: 16             # Prefetch batches
  persistent_workers: true        # Avoid worker restart overhead
```

## File Structure

```
train.py              # Main training script with FSDP
config.yaml           # Training configuration
download_model.py     # Model download utility
download_dataset.py   # Dataset preprocessing
inference_eval.py     # Model evaluation
slurm_train.sh        # SLURM job script
utils.py              # Utility functions
```

## Quick Commands

```bash
# Training
sbatch slurm_train.sh
sbatch slurm_train.sh latest  # Resume from checkpoint

# Monitoring
watch -n 1 nvidia-smi
tail -f logs/train_*.out

# MLflow
mlflow ui --backend-store-uri file:./mlruns

# Evaluation
python inference_eval.py --config config.yaml --fine_tuned_model_path ./checkpoints/final_model
```

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Low GPU utilization | Increase batch size or gradient accumulation |
| Out of memory | Enable gradient checkpointing, reduce batch size |
| Checkpoint hangs | Check dataset size, increase timeout |
| Communication errors | Verify NCCL config, check network connectivity |

## Key Metrics to Monitor

- **GPU Utilization**: Target 80%+
- **Training Speed**: Samples/second
- **Memory Usage**: Should stay below 90%
- **Loss**: Should decrease over time
- **Communication Overhead**: Should be minimal (< 10%)

## Extension Points

- DeepSpeed ZeRO integration
- LoRA/QLoRA for parameter-efficient fine-tuning
- Multi-task training support
- Automatic hyperparameter tuning
- Model quantization support

