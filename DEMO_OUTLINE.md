# LLM Fine-Tuning System - Overview & Demo Outline

## 1. Training Architecture Overview

### 1.1 High-Level Architecture
- **Distributed Training Framework**: PyTorch FSDP (Fully Sharded Data Parallel)
- **Hardware Setup**: Multi-node, multi-GPU (2 nodes × 8 H100 GPUs = 16 GPUs total)
- **Communication Backend**: NCCL (NVIDIA Collective Communications Library)
- **Training Paradigm**: Data Parallel with Model Sharding

### 1.2 Core Components

#### **Model Architecture**
- **Base Models**: HuggingFace Transformers (e.g., Llama-3.2-1B-Instruct)
- **Model Loading**: CPU-first loading, then FSDP wrapping for GPU distribution
- **Precision**: bfloat16 (BF16) mixed precision training
- **Optimizations**:
  - Gradient Checkpointing (memory efficiency)
  - Flash Attention 2 (optional, for memory and speed)
  - FSDP Sharding Strategy: FULL_SHARD (model parameters, gradients, optimizer states)

#### **Data Pipeline**
- **Dataset Format**: Pre-tokenized HuggingFace datasets
- **Data Loading**: 
  - DistributedSampler for data sharding across GPUs
  - Multi-worker DataLoader with prefetching
  - Persistent workers to avoid restart overhead
- **Tokenization**: Pre-tokenized or on-the-fly with max_length truncation
- **Batch Processing**: 
  - Per-device batch size × gradient accumulation × world_size = effective batch size

#### **Training Loop**
- **Framework**: HuggingFace Trainer with CustomTrainer extension
- **Optimizer**: AdamW (fused or 8-bit variants)
- **Learning Rate**: Cosine scheduler with warmup
- **Loss Function**: Causal Language Modeling (next token prediction)

### 1.3 FSDP Architecture Details

#### **Sharding Strategy**
```
FULL_SHARD: Model parameters, gradients, and optimizer states are sharded across all GPUs
- Each GPU holds a fraction of the model
- Parameters are gathered only when needed (forward/backward pass)
- Reduces memory footprint per GPU significantly
```

#### **Auto-Wrap Policy**
- **Transformer Layer Wrapping**: Automatically wraps transformer layers
- **Size-Based Fallback**: Wraps modules with >10M parameters if transformer detection fails

#### **State Dict Management**
- **Training**: SHARDED_STATE_DICT (memory efficient)
- **Checkpointing**: FULL_STATE_DICT (gathered on rank 0 only)
- **Checkpoint Optimization**: 
  - Skips checkpoints for very small datasets (< 500 samples)
  - Uses safetensors format for faster I/O
  - Saves only model weights (not optimizer states) for speed

### 1.4 Memory Management

#### **GPU Memory Optimizations**
- **Gradient Checkpointing**: Trade compute for memory
- **CPU Offload**: Optional (disabled by default for NVLink performance)
- **Mixed Precision**: BF16 reduces memory by 50% vs FP32
- **FSDP Sharding**: Distributes model across all GPUs
- **Cache Clearing**: Periodic CUDA cache clearing (configurable frequency)

#### **Memory Allocation Strategy**
- Max split size: 512MB chunks
- Dynamic memory allocation
- Memory monitoring and warnings at 90%+ usage

### 1.5 Communication Architecture

#### **NCCL Configuration**
- **Backend**: NCCL for multi-GPU communication
- **InfiniBand**: Enabled for inter-node communication
- **P2P Communication**: Enabled for intra-node GPU communication
- **NVLink**: Utilized for high-speed GPU-to-GPU transfers

#### **Synchronization Points**
- **Barriers**: Used before/after critical operations (checkpointing, state dict gathering)
- **All-Reduce**: For gradient synchronization
- **All-Gather**: For parameter gathering during forward/backward passes

## 2. System Components

### 2.1 Core Scripts

#### **train.py** (Main Training Script)
- **Purpose**: Distributed fine-tuning with FSDP
- **Key Features**:
  - Multi-node multi-GPU setup
  - FSDP model wrapping
  - Custom trainer with GPU utilization monitoring
  - MLflow integration for experiment tracking
  - Graceful shutdown handling
  - Checkpoint management (save/resume)
  - Error handling and recovery

#### **download_model.py**
- **Purpose**: Download models from HuggingFace Hub
- **Features**:
  - Handles gated models (token authentication)
  - Saves model and tokenizer locally
  - Supports cache directory configuration

#### **download_dataset.py**
- **Purpose**: Download and preprocess datasets
- **Features**:
  - Converts conversational formats (OpenHermes) to instruction-following format
  - Pre-tokenizes datasets for faster training
  - Handles various dataset formats
  - Saves processed dataset to disk

#### **inference_eval.py**
- **Purpose**: Evaluate base vs fine-tuned models
- **Features**:
  - Side-by-side model comparison
  - Comprehensive metrics (perplexity, generation speed, response length)
  - MLflow logging of evaluation results
  - Test prompt generation

### 2.2 Configuration System

#### **config.yaml**
- **Structure**: YAML-based configuration
- **Sections**:
  - `dataset`: Dataset paths, batch sizes, sequence lengths
  - `model`: Model paths, optimization flags
  - `training`: Hyperparameters (LR, epochs, warmup, etc.)
  - `fsdp`: FSDP-specific settings
  - `performance`: DataLoader and memory optimizations
  - `mlflow`: Experiment tracking configuration

### 2.3 SLURM Integration

#### **slurm_train.sh**
- **Purpose**: SLURM job submission script
- **Features**:
  - Multi-node job configuration
  - Environment variable setup
  - NCCL configuration
  - Master node discovery
  - Checkpoint resume support

## 3. Training Workflow

### 3.1 Pre-Training Setup

1. **Model Download**
   ```bash
   python download_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --output_dir "./models"
   ```

2. **Dataset Download & Preprocessing**
   ```bash
   python download_dataset.py --dataset_name "teknium/OpenHermes-2.5" --output_dir "./datasets" --model_name "meta-llama/Llama-3.2-1B-Instruct"
   ```

3. **Configuration**
   - Edit `config.yaml` with paths and hyperparameters
   - Set MLflow tracking URI
   - Configure FSDP and performance settings

### 3.2 Training Execution

1. **SLURM Job Submission**
   ```bash
   sbatch slurm_train.sh
   # Or with checkpoint resume:
   sbatch slurm_train.sh latest
   ```

2. **Training Process Flow**:
   ```
   Initialization
   ├── Distributed setup (NCCL, process groups)
   ├── Model loading (CPU → FSDP wrapping)
   ├── Dataset loading (pre-tokenized or tokenize on-the-fly)
   ├── Trainer initialization
   └── MLflow run start
   
   Training Loop
   ├── Forward pass (FSDP gathers parameters)
   ├── Loss computation
   ├── Backward pass (FSDP shards gradients)
   ├── Gradient synchronization (all-reduce)
   ├── Optimizer step
   ├── Learning rate update
   └── Logging (metrics to MLflow)
   
   Checkpointing (periodic)
   ├── Barrier synchronization
   ├── State dict gathering (rank 0 only)
   ├── Save checkpoint (safetensors format)
   └── Barrier synchronization
   
   Finalization
   ├── Final checkpoint save
   ├── Model consolidation (if needed)
   ├── MLflow logging
   └── Cleanup
   ```

### 3.3 Post-Training

1. **Model Evaluation**
   ```bash
   python inference_eval.py --config config.yaml --fine_tuned_model_path ./checkpoints/final_model
   ```

2. **Results Review**
   - MLflow UI: `mlflow ui --backend-store-uri file:./mlruns`
   - Compare metrics: loss curves, GPU utilization, generation quality

## 4. Demo Structure

### 4.1 Demo Flow (30-45 minutes)

#### **Part 1: Architecture Overview (10 min)**
1. **System Overview** (3 min)
   - Hardware setup (2 nodes × 8 H100 GPUs)
   - Distributed training paradigm
   - FSDP sharding strategy visualization

2. **Key Components** (4 min)
   - Model architecture and loading
   - Data pipeline and preprocessing
   - Training loop with FSDP
   - Memory management strategies

3. **Configuration Deep Dive** (3 min)
   - Walk through `config.yaml`
   - Explain key hyperparameters
   - FSDP and performance settings

#### **Part 2: Code Walkthrough (15 min)**
1. **train.py Structure** (8 min)
   - Distributed setup (`setup_distributed()`)
   - Model loading and FSDP wrapping
   - CustomTrainer class
   - Checkpoint saving logic
   - Error handling and cleanup

2. **Data Pipeline** (4 min)
   - Dataset loading (`prepare_dataset()`)
   - Tokenization process
   - DistributedSampler usage
   - DataLoader optimization

3. **Training Loop** (3 min)
   - CustomTrainer training step
   - GPU utilization monitoring
   - MLflow integration
   - Checkpoint management

#### **Part 3: Live Demo (15 min)**
1. **Setup Verification** (3 min)
   - Check GPU availability
   - Verify distributed setup
   - Test NCCL communication

2. **Configuration Review** (2 min)
   - Show current config.yaml
   - Explain key settings
   - Adjust if needed for demo

3. **Training Execution** (8 min)
   - Submit SLURM job (or run locally if single GPU)
   - Monitor GPU utilization
   - Show MLflow metrics in real-time
   - Demonstrate checkpoint saving

4. **Results Analysis** (2 min)
   - Show training metrics
   - Display GPU utilization stats
   - Review checkpoint structure

#### **Part 4: Q&A and Best Practices (5 min)**
1. **Common Issues and Solutions**
   - Low GPU utilization → increase batch size
   - Out of memory → enable gradient checkpointing
   - Checkpoint hangs → adjust checkpoint settings

2. **Performance Tuning Tips**
   - Batch size optimization
   - Gradient accumulation tuning
   - DataLoader worker configuration

3. **Scaling Considerations**
   - Adding more nodes
   - Different model sizes
   - Memory vs speed tradeoffs

### 4.2 Demo Script Sections

#### **Section 1: Introduction**
```markdown
# Welcome to LLM Fine-Tuning System Demo

## What we'll cover:
1. Training architecture with FSDP
2. Code structure and key components
3. Live training demonstration
4. Results and metrics analysis

## System Overview:
- 2 nodes × 8 H100 GPUs (16 GPUs total)
- PyTorch FSDP for distributed training
- MLflow for experiment tracking
- SLURM for job management
```

#### **Section 2: Architecture Deep Dive**
```markdown
## FSDP Architecture

### How FSDP Works:
1. Model is sharded across all GPUs
2. Each GPU holds a fraction of parameters
3. Parameters gathered only when needed
4. Gradients sharded and synchronized via all-reduce

### Memory Benefits:
- Without FSDP: 7B model needs ~28GB per GPU (FP32) or ~14GB (BF16)
- With FSDP FULL_SHARD: 7B model needs ~2GB per GPU (BF16)
- Enables training larger models on limited GPU memory
```

#### **Section 3: Code Walkthrough**
```markdown
## Key Functions in train.py

### setup_distributed()
- Initializes NCCL process group
- Handles SLURM environment variables
- Sets up CUDA device assignment
- Verifies communication between ranks

### Model Loading
- Loads model on CPU first (low_cpu_mem_usage)
- Wraps with FSDP using auto_wrap_policy
- Enables gradient checkpointing if configured
- Moves to GPU via FSDP device placement

### CustomTrainer
- Extends HuggingFace Trainer
- Monitors GPU utilization
- Optimizes checkpoint saving for FSDP
- Handles graceful shutdown
```

#### **Section 4: Live Demo Commands**
```bash
# 1. Check setup
python check_setup.py

# 2. Review config
cat config.yaml | grep -A 5 "dataset:"
cat config.yaml | grep -A 5 "fsdp:"

# 3. Submit training job
sbatch slurm_train.sh

# 4. Monitor job
squeue -u $USER
tail -f logs/train_*.out

# 5. Check GPU utilization (in another terminal)
watch -n 1 nvidia-smi

# 6. View MLflow metrics
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
```

#### **Section 5: Results Analysis**
```markdown
## Metrics to Review

### Training Metrics:
- Loss curve (should decrease over time)
- Learning rate schedule (warmup → cosine decay)
- GPU utilization (target: 80%+)
- Training speed (samples/second)

### System Metrics:
- Memory usage per GPU
- Communication overhead
- DataLoader efficiency

### Model Quality:
- Generation examples (before/after)
- Perplexity scores
- Response quality improvements
```

## 5. Key Talking Points

### 5.1 Why FSDP?
- **Memory Efficiency**: Enables training large models on limited GPU memory
- **Scalability**: Works across multiple nodes seamlessly
- **Performance**: Maintains high GPU utilization with proper configuration
- **Flexibility**: Supports various sharding strategies

### 5.2 Design Decisions

#### **CPU-First Model Loading**
- Prevents OOM during model initialization
- Allows FSDP to handle GPU placement
- More reliable for large models

#### **Pre-Tokenized Datasets**
- Faster training startup
- Consistent tokenization
- Reduces CPU overhead during training

#### **CustomTrainer**
- GPU utilization monitoring
- Optimized checkpoint saving
- Better error handling
- MLflow integration

#### **Safetensors Checkpoint Format**
- Faster I/O than PyTorch format
- More secure (no arbitrary code execution)
- Smaller file sizes

### 5.3 Performance Optimizations

1. **Batch Size Tuning**
   - Start small, increase until memory limit
   - Use gradient accumulation for larger effective batches

2. **DataLoader Optimization**
   - Multiple workers (4-8 typically optimal)
   - Persistent workers (avoid restart overhead)
   - Prefetch factor (2-4 for balance)

3. **Memory Management**
   - Gradient checkpointing (trade compute for memory)
   - Periodic cache clearing
   - FSDP CPU offload (if needed, but slower)

4. **Communication Optimization**
   - NVLink for intra-node
   - InfiniBand for inter-node
   - Limit all-gathers to reduce memory spikes

## 6. Troubleshooting Guide

### Common Issues

1. **Low GPU Utilization**
   - Increase batch size or gradient accumulation
   - Increase DataLoader workers
   - Check for data loading bottlenecks

2. **Out of Memory**
   - Enable gradient checkpointing
   - Reduce batch size
   - Enable FSDP CPU offload (slower)
   - Reduce sequence length

3. **Checkpoint Hangs**
   - Check dataset size (skips for < 500 samples)
   - Increase checkpoint timeout
   - Disable CPU offload during checkpointing

4. **Distributed Training Failures**
   - Verify NCCL configuration
   - Check network connectivity
   - Ensure all nodes can access shared storage
   - Verify SLURM environment variables

## 7. Extension Points

### Future Enhancements
- DeepSpeed ZeRO integration
- LoRA/QLoRA for parameter-efficient fine-tuning
- Multi-task training support
- Automatic hyperparameter tuning
- Model quantization support

### Customization Options
- Different model architectures
- Custom datasets and formats
- Alternative optimizers
- Custom loss functions
- Evaluation metrics

---

## Quick Reference

### Key Files
- `train.py`: Main training script
- `config.yaml`: Configuration file
- `slurm_train.sh`: SLURM job script
- `download_model.py`: Model download utility
- `download_dataset.py`: Dataset preprocessing utility
- `inference_eval.py`: Model evaluation script

### Key Commands
```bash
# Training
sbatch slurm_train.sh
sbatch slurm_train.sh latest  # Resume from latest checkpoint

# Monitoring
watch -n 1 nvidia-smi
tail -f logs/train_*.out

# MLflow
mlflow ui --backend-store-uri file:./mlruns

# Evaluation
python inference_eval.py --config config.yaml --fine_tuned_model_path ./checkpoints/final_model
```

### Key Metrics
- **GPU Utilization**: Target 80%+
- **Training Speed**: Samples/second
- **Memory Usage**: Should stay below 90%
- **Loss**: Should decrease over time
- **Checkpoint Frequency**: Every N steps (configurable)

