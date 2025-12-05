# LLM Fine-Tuning on Nebius Soperator Cluster

This repository contains scripts and code for fine-tuning Large Language Models (LLMs) on a Nebius soperator cluster with 2 worker nodes, each with 8 H100 GPUs (16 GPUs total).

## Features

- **Distributed Training**: Multi-node multi-GPU training using PyTorch FSDP (Fully Sharded Data Parallel)
- **High GPU Utilization**: Optimized for 80%+ GPU utilization
- **MLflow Integration**: Track training metrics, model performance, and compare before/after fine-tuning
- **SLURM Integration**: Ready-to-use SLURM job scripts
- **H100 Optimizations**: Specific optimizations for H100 GPUs

## Project Structure

```
.
├── requirements.txt              # Python dependencies
├── config.yaml                  # Training configuration
├── download_model.py            # Script to download models
├── download_dataset.py          # Script to download datasets
├── train.py                     # Main training script with FSDP
├── monitor_gpu.py               # GPU monitoring utility
├── utils.py                     # Utility functions
├── slurm_download_model.sh      # SLURM job for model download
├── slurm_download_dataset.sh    # SLURM job for dataset download
├── slurm_train.sh               # SLURM job for training
└── README.md                    # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Training

Edit `config.yaml` to set:
- Model name and path
- Dataset name and path
- Training hyperparameters
- FSDP settings
- MLflow tracking URI

### 3. Download Model

Submit the model download job:

```bash
sbatch slurm_download_model.sh
```

Or run manually:

```bash
python download_model.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --output_dir "./models"
```

### 4. Download Dataset

Submit the dataset download job:

```bash
sbatch slurm_download_dataset.sh
```

Or run manually:

```bash
python download_dataset.py \
    --dataset_name "wikitext" \
    --output_dir "./datasets" \
    --tokenizer_path "./models"
```

## Running Training

### Submit SLURM Job

```bash
sbatch slurm_train.sh
```

### Monitor Training

In a separate terminal, monitor GPU utilization:

```bash
python monitor_gpu.py --interval 5 --output gpu_stats.json
```

Or use `watch` with `nvidia-smi`:

```bash
watch -n 1 nvidia-smi
```

### Check MLflow

Start MLflow UI to view training metrics:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open http://localhost:5000 in your browser.

## Configuration

### Key Configuration Parameters

**Model Settings:**
- `model.name`: HuggingFace model identifier
- `model.path`: Local path to downloaded model
- `model.gradient_checkpointing`: Enable to save memory

**Dataset Settings:**
- `dataset.batch_size_per_device`: Batch size per GPU
- `dataset.gradient_accumulation_steps`: Gradient accumulation steps
- Effective batch size = `batch_size_per_device × num_gpus × gradient_accumulation_steps`

**FSDP Settings:**
- `fsdp.sharding_strategy`: "FULL_SHARD" for maximum memory efficiency
- `fsdp.mixed_precision`: Use bfloat16 for H100

**Performance Tuning:**
- Adjust `batch_size_per_device` and `gradient_accumulation_steps` to maximize GPU utilization
- Increase `dataloader_num_workers` if CPU is not fully utilized
- Enable `gradient_checkpointing` if running out of memory

## GPU Utilization Optimization

To achieve 80%+ GPU utilization:

1. **Batch Size**: Start with `batch_size_per_device=4` and increase until memory is fully utilized
2. **Gradient Accumulation**: Use gradient accumulation to increase effective batch size without using more memory
3. **Data Loading**: Set `dataloader_num_workers=4-8` to keep GPUs fed with data
4. **Mixed Precision**: Use bfloat16 (already enabled for H100)
5. **FSDP**: Use FULL_SHARD for large models to distribute memory across GPUs

## MLflow Integration

MLflow tracks:
- Training metrics (loss, learning rate, GPU utilization)
- Model performance before and after fine-tuning
- System metrics (CPU, memory, GPU usage)
- Model artifacts and checkpoints
- Configuration parameters

### Viewing Results

1. Start MLflow UI: `mlflow ui --backend-store-uri file:./mlruns`
2. Compare runs to see performance improvements
3. View before/after generation samples
4. Monitor GPU utilization trends

## Troubleshooting

### Low GPU Utilization

1. Increase batch size or gradient accumulation steps
2. Increase `dataloader_num_workers`
3. Check data loading pipeline for bottlenecks
4. Verify FSDP is working correctly (check logs)

### Out of Memory

1. Enable gradient checkpointing
2. Reduce batch size
3. Use CPU offloading in FSDP (slower but uses less GPU memory)
4. Reduce sequence length

### Distributed Training Issues

1. Verify NCCL is properly configured
2. Check network connectivity between nodes
3. Ensure all nodes can access shared storage
4. Check SLURM environment variables

## Customization

### Using Different Models

1. Update `config.yaml` with your model name
2. Download the model using `download_model.py`
3. Ensure the model supports FSDP (most HuggingFace models do)

### Using Custom Datasets

1. Prepare your dataset in HuggingFace format or JSON/JSONL
2. Update `config.yaml` with dataset path
3. Ensure dataset has a text column or update `train.py` accordingly

### Adjusting for Different GPU Counts

Update `slurm_train.sh`:
- `--nodes`: Number of nodes
- `--ntasks-per-node`: GPUs per node
- `--gres=gpu:X`: Number of GPUs per node

## Performance Benchmarks

Expected performance on 2 nodes × 8 H100 GPUs:
- **Model**: Llama-2-7B
- **Batch Size**: 4 per GPU
- **Gradient Accumulation**: 8 steps
- **Effective Batch Size**: 512
- **GPU Utilization**: 80-90%
- **Training Speed**: ~2-3 tokens/second/GPU (varies by model and sequence length)

## License

This project is provided as-is for fine-tuning LLMs on HPC clusters.

## Support

For issues or questions:
1. Check SLURM job logs in `logs/` directory
2. Review MLflow metrics for training insights
3. Monitor GPU utilization with `monitor_gpu.py`

