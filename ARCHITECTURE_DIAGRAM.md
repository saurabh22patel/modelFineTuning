# Training Architecture Diagrams

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      SLURM Cluster                               │
│                                                                  │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │   Node 1 (8 H100s)   │      │   Node 2 (8 H100s)   │        │
│  │                      │      │                      │        │
│  │  GPU0  GPU1  GPU2... │      │  GPU0  GPU1  GPU2... │        │
│  │   │     │     │      │      │   │     │     │      │        │
│  └───┼─────┼─────┼──────┘      └───┼─────┼─────┼──────┘        │
│      │     │     │                  │     │     │                │
│      └─────┼─────┼──────────────────┼─────┼─────┘                │
│            │     │                  │     │                      │
│            └─────┴──────────────────┴─────┘                      │
│                    InfiniBand Network                            │
└─────────────────────────────────────────────────────────────────┘
```

## 2. FSDP Sharding Strategy

### Model Distribution Across GPUs

```
Without FSDP (DDP):
┌─────────────────────────────────────────┐
│  GPU 0: Full Model (7B params)        │  ~14GB (BF16)
│  GPU 1: Full Model (7B params)        │  ~14GB (BF16)
│  GPU 2: Full Model (7B params)        │  ~14GB (BF16)
│  ...                                   │
│  GPU 15: Full Model (7B params)        │  ~14GB (BF16)
└─────────────────────────────────────────┘
Total Memory: 224GB

With FSDP FULL_SHARD:
┌─────────────────────────────────────────┐
│  GPU 0: 1/16 of Model                  │  ~0.875GB
│  GPU 1: 1/16 of Model                  │  ~0.875GB
│  GPU 2: 1/16 of Model                  │  ~0.875GB
│  ...                                   │
│  GPU 15: 1/16 of Model                 │  ~0.875GB
└─────────────────────────────────────────┘
Total Memory: 14GB (distributed)
```

### Parameter Gathering During Forward Pass

```
Step 1: Forward Pass (Gather Parameters)
┌─────────┐  ┌─────────┐  ┌─────────┐
│ GPU 0   │  │ GPU 1   │  │ GPU 2   │
│ Param A │  │ Param B │  │ Param C │
└────┬────┘  └────┬────┘  └────┬────┘
     │           │           │
     └───────────┼───────────┘
                 │
         All-Gather Operation
                 │
         ┌───────▼───────┐
         │  Full Model   │
         │  (Temporary)  │
         └───────┬───────┘
                 │
         Forward Pass
                 │
         └───────▼───────┐
         │  Activations  │
         └───────────────┘

Step 2: Backward Pass (Shard Gradients)
┌─────────┐  ┌─────────┐  ┌─────────┐
│ GPU 0   │  │ GPU 1   │  │ GPU 2   │
│ Grad A  │  │ Grad B  │  │ Grad C  │
└────┬────┘  └────┬────┘  └────┬────┘
     │           │           │
     └───────────┼───────────┘
                 │
         All-Reduce Operation
                 │
         └───────▼───────┐
         │ Synchronized  │
         │   Gradients   │
         └───────────────┘
```

## 3. Training Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataset (Pre-tokenized)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Sample 0 │  │ Sample 1 │  │ Sample 2 │  │ Sample 3 │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ DistributedSampler
                          │ (shards data across GPUs)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   ┌────────┐        ┌────────┐        ┌────────┐
   │ GPU 0  │        │ GPU 1  │        │ GPU 2  │
   │ Batch  │        │ Batch  │        │ Batch  │
   │ 0,3,6..│        │ 1,4,7..│        │ 2,5,8..│
   └───┬────┘        └───┬────┘        └───┬────┘
       │                 │                 │
       │  Forward Pass   │                 │
       │  (FSDP Gather)  │                 │
       │                 │                 │
       ▼                 ▼                 ▼
   ┌────────┐        ┌────────┐        ┌────────┐
   │ Loss   │        │ Loss   │        │ Loss   │
   └───┬────┘        └───┬────┘        └───┬────┘
       │                 │                 │
       │  Backward Pass  │                 │
       │  (FSDP Shard)   │                 │
       │                 │                 │
       └─────────────────┼─────────────────┘
                        │
                 All-Reduce Gradients
                        │
                        ▼
                 ┌──────────────┐
                 │ Optimizer    │
                 │ Step (Local) │
                 └──────────────┘
```

## 4. Training Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Initialization                  │
├─────────────────────────────────────────────────────────────┤
│  1. Setup Distributed (NCCL, process groups)                │
│  2. Load Model (CPU → FSDP wrap → GPU)                     │
│  3. Load Dataset (pre-tokenized or tokenize)               │
│  4. Initialize Trainer (CustomTrainer)                     │
│  5. Setup MLflow tracking                                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  For each epoch:                                             │
│    For each batch:                                           │
│      ┌─────────────────────────────────────┐                │
│      │ 1. Load batch (DataLoader)          │                │
│      │ 2. Forward pass                      │                │
│      │    - FSDP gathers parameters        │                │
│      │    - Compute logits                 │                │
│      │    - Calculate loss                 │                │
│      │ 3. Backward pass                     │                │
│      │    - Compute gradients              │                │
│      │    - FSDP shards gradients          │                │
│      │    - All-reduce gradients           │                │
│      │ 4. Optimizer step                   │                │
│      │    - Update parameters (local)       │                │
│      │ 5. Learning rate update              │                │
│      │ 6. Log metrics (MLflow)              │                │
│      │ 7. Monitor GPU utilization           │                │
│      └─────────────────────────────────────┘                │
│                                                               │
│      Every N steps:                                          │
│        ┌─────────────────────────────────┐                 │
│        │ Checkpoint Save                  │                 │
│        │ - Barrier sync                   │                 │
│        │ - Gather state dict (rank 0)    │                 │
│        │ - Save to disk (safetensors)    │                 │
│        │ - Barrier sync                   │                 │
│        └─────────────────────────────────┘                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Training Finalization                     │
├─────────────────────────────────────────────────────────────┤
│  1. Save final checkpoint                                    │
│  2. Consolidate model (if needed)                            │
│  3. Log final metrics to MLflow                              │
│  4. Cleanup distributed resources                           │
│  5. End MLflow run                                           │
└─────────────────────────────────────────────────────────────┘
```

## 5. Memory Management Strategy

```
┌─────────────────────────────────────────────────────────────┐
│              GPU Memory Breakdown (Per GPU)                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Without FSDP (7B model, BF16):                             │
│  ┌─────────────────────────────────────────┐                │
│  │ Model Parameters:     ~14GB            │                │
│  │ Gradients:            ~14GB              │                │
│  │ Optimizer States:     ~28GB (AdamW)     │                │
│  │ Activations:          ~2-4GB             │                │
│  │ ─────────────────────────────────────── │                │
│  │ Total:                ~58-60GB           │                │
│  └─────────────────────────────────────────┘                │
│                                                               │
│  With FSDP FULL_SHARD (7B model, BF16, 16 GPUs):            │
│  ┌─────────────────────────────────────────┐                │
│  │ Model Parameters:     ~0.875GB (1/16)   │                │
│  │ Gradients:            ~0.875GB (1/16)   │                │
│  │ Optimizer States:     ~1.75GB (1/16)    │                │
│  │ Activations:          ~2-4GB             │                │
│  │ ─────────────────────────────────────── │                │
│  │ Total:                ~5.5-7GB           │                │
│  └─────────────────────────────────────────┘                │
│                                                               │
│  Additional Optimizations:                                   │
│  • Gradient Checkpointing: Reduces activation memory by 50% │
│  • Mixed Precision (BF16): Reduces memory by 50%           │
│  • Periodic Cache Clearing: Prevents fragmentation          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 6. Communication Patterns

```
┌─────────────────────────────────────────────────────────────┐
│              NCCL Communication Operations                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Intra-Node (Same Node, 8 GPUs):                            │
│  ┌─────────────────────────────────────────┐                │
│  │ GPU0 ──NVLink── GPU1                     │                │
│  │  │              │                        │                │
│  │  │              │                        │                │
│  │ GPU2 ──NVLink── GPU3                    │                │
│  │  │              │                        │                │
│  │  └──────────────┴──────────────┐         │                │
│  │                               │         │                │
│  │ GPU4 ──NVLink── GPU5          │         │                │
│  │  │              │             │         │                │
│  │ GPU6 ──NVLink── GPU7          │         │                │
│  └─────────────────────────────────────────┘                │
│  Communication: NVLink (high bandwidth, low latency)        │
│                                                               │
│  Inter-Node (Different Nodes):                               │
│  ┌─────────────────────────────────────────┐                │
│  │ Node 1 (8 GPUs)                        │                │
│  │      │                                  │                │
│  │      │ InfiniBand                       │                │
│  │      │                                  │                │
│  │ Node 2 (8 GPUs)                        │                │
│  └─────────────────────────────────────────┘                │
│  Communication: InfiniBand (high bandwidth, low latency)    │
│                                                               │
│  Collective Operations:                                      │
│  • All-Reduce: Gradient synchronization                     │
│  • All-Gather: Parameter gathering for forward pass         │
│  • Reduce-Scatter: Gradient sharding                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 7. Checkpoint Saving Process

```
┌─────────────────────────────────────────────────────────────┐
│              FSDP Checkpoint Saving Flow                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Step 1: Synchronization                                     │
│  ┌─────────────────────────────────────────┐                │
│  │ All ranks: Barrier()                     │                │
│  │ Wait for all ranks to reach checkpoint   │                │
│  └─────────────────────────────────────────┘                │
│                          │                                   │
│                          ▼                                   │
│  Step 2: State Dict Gathering                               │
│  ┌─────────────────────────────────────────┐                │
│  │ with FSDP.state_dict_type(              │                │
│  │     FULL_STATE_DICT,                     │                │
│  │     rank0_only=True                      │                │
│  │ ):                                       │                │
│  │   state_dict = model.state_dict()       │                │
│  │   # All ranks participate in all-gather   │                │
│  │   # Only rank 0 gets full state dict     │                │
│  └─────────────────────────────────────────┘                │
│                          │                                   │
│                          ▼                                   │
│  Step 3: Save (Rank 0 Only)                                 │
│  ┌─────────────────────────────────────────┐                │
│  │ if rank == 0:                           │                │
│  │   save_file(state_dict,                 │                │
│  │            "model.safetensors")         │                │
│  │   tokenizer.save_pretrained(...)        │                │
│  │   trainer_state.save(...)               │                │
│  └─────────────────────────────────────────┘                │
│                          │                                   │
│                          ▼                                   │
│  Step 4: Final Synchronization                              │
│  ┌─────────────────────────────────────────┐                │
│  │ All ranks: Barrier()                     │                │
│  │ Resume training                          │                │
│  └─────────────────────────────────────────┘                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 8. Component Interaction Diagram

```
┌──────────────┐
│  config.yaml │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                      train.py                                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Distributed  │───▶│   Model     │───▶│   Dataset    │  │
│  │   Setup      │    │   Loading    │    │   Loading    │  │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘  │
│                             │                    │          │
│                             ▼                    ▼          │
│                      ┌──────────────┐    ┌──────────────┐  │
│                      │   FSDP       │    │ Distributed  │  │
│                      │   Wrapping   │    │   Sampler    │  │
│                      └──────┬───────┘    └──────┬───────┘  │
│                             │                    │          │
│                             └────────┬───────────┘          │
│                                      ▼                       │
│                             ┌──────────────┐                │
│                             │  CustomTrainer│                │
│                             │  (HuggingFace)│                │
│                             └──────┬───────┘                │
│                                    │                         │
│                    ┌───────────────┼───────────────┐        │
│                    │               │               │        │
│                    ▼               ▼               ▼        │
│            ┌──────────────┐ ┌──────────────┐ ┌──────────┐ │
│            │   Training   │ │  Checkpoint  │ │  MLflow  │ │
│            │     Loop     │ │   Saving     │ │ Tracking │ │
│            └──────────────┘ └──────────────┘ └──────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
       │
       │
       ▼
┌──────────────┐
│  Checkpoints  │
│  (safetensors)│
└──────────────┘
```

## 9. Data Flow Through System

```
Dataset (Disk)
    │
    │ load_from_disk()
    ▼
Pre-tokenized Dataset
    │
    │ DistributedSampler
    │ (shards across GPUs)
    ▼
DataLoader (per GPU)
    │
    │ Batch loading
    │ (multi-worker, prefetch)
    ▼
Training Batch
    │
    │ Forward pass
    │ (FSDP gathers params)
    ▼
Model Output (Logits)
    │
    │ Loss computation
    ▼
Loss Value
    │
    │ Backward pass
    │ (FSDP shards grads)
    ▼
Gradients (Sharded)
    │
    │ All-reduce
    │ (synchronize)
    ▼
Synchronized Gradients
    │
    │ Optimizer step
    ▼
Updated Parameters (Sharded)
    │
    │ Next iteration
    ▼
[Loop continues...]
```

## 10. Error Handling and Recovery

```
┌─────────────────────────────────────────────────────────────┐
│              Error Handling Flow                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Training Step                                               │
│      │                                                       │
│      ├─▶ OOM Error                                          │
│      │   └─▶ Clear cache                                   │
│      │       └─▶ Log warning                               │
│      │           └─▶ Suggest: reduce batch size            │
│      │                                                       │
│      ├─▶ Checkpoint Hang                                   │
│      │   └─▶ Timeout detection                             │
│      │       └─▶ Skip checkpoint                           │
│      │           └─▶ Log warning                           │
│      │                                                       │
│      ├─▶ Communication Error                                │
│      │   └─▶ Retry with backoff                            │
│      │       └─▶ If fails: graceful shutdown              │
│      │                                                       │
│      └─▶ Signal (SIGTERM/SIGINT)                            │
│          └─▶ Set shutdown flag                              │
│              └─▶ Finish current step                        │
│                  └─▶ Save checkpoint                        │
│                      └─▶ Cleanup resources                 │
│                          └─▶ Exit gracefully               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

