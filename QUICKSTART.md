# Quick Start Guide

## Prerequisites

- Access to Nebius soperator cluster with SLURM
- Python 3.10+
- CUDA 11.8+ (for H100 GPUs)
- Access to shared storage for models/datasets

## Step-by-Step Setup

### 1. Initial Setup

```bash
# Clone or navigate to the project directory
cd /path/to/modelFineTuning

# Run setup script
./setup.sh
```

### 2. Configure Your Training

Edit `config.yaml`:

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # Your model
  path: "./models"

dataset:
  name: "wikitext"  # Your dataset
  path: "./datasets/dataset"
  batch_size_per_device: 4
  gradient_accumulation_steps: 8
```

### 3. Download Model

```bash
# Option 1: Submit SLURM job
sbatch slurm_download_model.sh

# Option 2: Run directly (if on login node)
python download_model.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --output_dir "./models"
```

### 4. Download Dataset

```bash
# Option 1: Submit SLURM job
sbatch slurm_download_dataset.sh

# Option 2: Run directly
python download_dataset.py \
    --dataset_name "wikitext" \
    --output_dir "./datasets" \
    --tokenizer_path "./models"
```

### 5. Start Training

```bash
# Submit training job
sbatch slurm_train.sh

# Monitor job
squeue -u $USER

# Check logs
tail -f logs/train_*.out
```

### 6. Monitor GPU Utilization

In a separate terminal (or after job starts):

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use the monitoring script
python monitor_gpu.py --interval 5 --output gpu_stats.json
```

### 7. View MLflow Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5000

# Open in browser: http://localhost:5000
```

## Tuning for 80%+ GPU Utilization

If GPU utilization is below 80%:

1. **Increase Batch Size**: Edit `config.yaml`
   ```yaml
   dataset:
     batch_size_per_device: 6  # Increase from 4
   ```

2. **Increase Gradient Accumulation**:
   ```yaml
   dataset:
     gradient_accumulation_steps: 12  # Increase from 8
   ```

3. **Optimize Data Loading**:
   ```yaml
   performance:
     dataloader_num_workers: 8  # Increase from 4
   ```

4. **Check for Bottlenecks**:
   - Monitor CPU usage
   - Check data loading pipeline
   - Verify network bandwidth between nodes

## Common Issues

### Issue: Out of Memory

**Solution**: Reduce batch size or enable gradient checkpointing
```yaml
model:
  gradient_checkpointing: true
dataset:
  batch_size_per_device: 2  # Reduce
```

### Issue: Low GPU Utilization

**Solution**: Increase batch size and data loading workers
```yaml
dataset:
  batch_size_per_device: 6
  gradient_accumulation_steps: 12
performance:
  dataloader_num_workers: 8
```

### Issue: Distributed Training Fails

**Solution**: Check NCCL configuration in `slurm_train.sh`
- Verify network interface (ib0, eth0, etc.)
- Check NCCL_IB_DISABLE setting
- Ensure nodes can communicate

## Expected Results

After successful training, you should see:

1. **Checkpoints** in `./checkpoints/`
2. **Final model** in `./checkpoints/final_model/`
3. **MLflow metrics** showing:
   - Training loss decreasing
   - GPU utilization > 80%
   - Before/after model performance comparison

## Next Steps

- Compare model performance in MLflow UI
- Export fine-tuned model for inference
- Adjust hyperparameters based on results
- Scale to more nodes/GPUs if needed

