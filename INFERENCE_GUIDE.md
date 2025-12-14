# Inference Guide: Running Inference on Original and Fine-Tuned Models

This guide explains how to run inference on both the original (base) model and your fine-tuned model using the `inference_eval.py` script.

## Quick Start

### Option 1: Run Both Models (Recommended)

The script automatically evaluates both models if paths are configured correctly:

```bash
python inference_eval.py --config config.yaml
```

This will:
- Load the base model from `config.yaml` → `model.path`
- Load the fine-tuned model from `config.yaml` → `training.output_dir/final_model`
- Compare their performance side-by-side
- Log all results to MLflow

### Option 2: Specify Custom Paths

If your models are in different locations:

```bash
python inference_eval.py \
    --config config.yaml \
    --base_model_path /path/to/original/model \
    --fine_tuned_model_path /path/to/fine-tuned/model/checkpoint
```

### Option 3: Using SLURM

Submit a SLURM job:

```bash
sbatch slurm_inference_eval.sh
```

Or with custom environment variables:

```bash
export BASE_MODEL_PATH="/mnt/data/models/1b"
export FINE_TUNED_MODEL_PATH="/mnt/data/models/checkpoints/final_model"
export NUM_PROMPTS=50
export MAX_NEW_TOKENS=512
sbatch slurm_inference_eval.sh
```

## Understanding Model Paths

### Base Model Path
- **Config location**: `config.yaml` → `model.path`
- **Default example**: `/mnt/data/models/1b`
- **Can be**: Local path or HuggingFace model ID (e.g., `meta-llama/Llama-3.2-1B-Instruct`)

### Fine-Tuned Model Path
- **Config location**: `config.yaml` → `training.output_dir` + `/final_model`
- **Default example**: `/mnt/data/models/checkpoints/final_model`
- **Note**: After training, you may need to consolidate checkpoints first (see `consolidate_checkpoint.py`)

## Common Use Cases

### 1. Evaluate Both Models with Default Settings

```bash
python inference_eval.py --config config.yaml
```

### 2. Evaluate with More Prompts and Longer Responses

```bash
python inference_eval.py \
    --config config.yaml \
    --num_prompts 50 \
    --max_new_tokens 512
```

### 3. Include Perplexity Calculation

```bash
python inference_eval.py \
    --config config.yaml \
    --calculate_perplexity \
    --eval_dataset_path /path/to/eval/dataset
```

### 4. Custom MLflow Run Name

```bash
python inference_eval.py \
    --config config.yaml \
    --run_name "experiment_v2_comparison"
```

### 5. Evaluate Only Base Model

If you want to evaluate only the base model, you can point the fine-tuned path to a non-existent location:

```bash
python inference_eval.py \
    --config config.yaml \
    --fine_tuned_model_path /nonexistent/path
```

The script will skip fine-tuned evaluation and only evaluate the base model.

## Output and Results

### Console Output
The script prints:
- Progress for each prompt
- Aggregate metrics for each model
- Comparison metrics between models
- Summary statistics

### MLflow Logging
All results are logged to MLflow:
- **Metrics**: Response lengths, generation speed, tokens per second, perplexity
- **Artifacts**: 
  - `evaluation/test_prompts.txt` - All test prompts used
  - `evaluation/base_model_generations.txt` - Base model responses
  - `evaluation/fine_tuned_model_generations.txt` - Fine-tuned model responses

### Viewing Results in MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open http://localhost:5000 in your browser to:
- Compare metrics side-by-side
- View generated text samples
- Track performance improvements

## Metrics Explained

### Per-Model Metrics
- **avg_response_length**: Average character length of responses
- **avg_tokens**: Average number of tokens generated
- **avg_generation_time**: Average time to generate responses
- **avg_tokens_per_second**: Generation speed
- **perplexity**: Model's uncertainty (lower is better, if calculated)

### Comparison Metrics
- **avg_response_length_diff**: Difference in response lengths
- **avg_response_length_pct_change**: Percentage change in response length
- **avg_speed_diff**: Difference in generation speed
- **perplexity_improvement**: Improvement in perplexity (positive = better)

## Troubleshooting

### Fine-Tuned Model Not Found

If you see:
```
Warning: Fine-tuned model path does not exist: /path/to/model
Skipping fine-tuned model evaluation.
```

**Solutions:**
1. Check that training completed successfully
2. Verify the checkpoint path in your config
3. **If using FSDP checkpoints**, you need to consolidate them first:
   ```bash
   python consolidate_checkpoint.py \
       /path/to/checkpoint-XXXX \
       /path/to/final_model \
       --config config.yaml
   ```
   Then use the consolidated model for inference:
   ```bash
   python inference_eval.py \
       --config config.yaml \
       --fine_tuned_model_path /path/to/final_model
   ```
4. Manually specify the correct path:
   ```bash
   --fine_tuned_model_path /correct/path/to/model
   ```

### Finding Your Checkpoints After Training

After FSDP training, checkpoints are saved in the `output_dir` from your config. To find them:

```bash
# List all checkpoints
ls -la /mnt/data/models/checkpoints/

# Find the latest checkpoint
ls -t /mnt/data/models/checkpoints/checkpoint-* | head -1
```

Checkpoints are typically named `checkpoint-XXXX` where XXXX is the step number. The latest checkpoint is usually the one with the highest step number.

### Consolidating FSDP Checkpoints

FSDP saves checkpoints in a sharded format. To use them for inference, you must consolidate them:

```bash
# Example: Consolidate checkpoint-2000
python consolidate_checkpoint.py \
    /mnt/data/models/checkpoints/checkpoint-2000 \
    /mnt/data/models/checkpoints/final_model \
    --config config.yaml
```

This creates a standard HuggingFace model at `/mnt/data/models/checkpoints/final_model` that can be used for inference.

### Out of Memory

If you run out of GPU memory:
- Reduce `--max_new_tokens` (e.g., 128 instead of 256)
- Reduce `--num_prompts` (e.g., 10 instead of 20)
- Ensure only one model is loaded at a time (the script does this automatically)

### HuggingFace Authentication

If using a gated model, ensure your token is set:
```bash
export HF_TOKEN="your_token_here"
# Or set in config.yaml under model.hf_token
```

## Example Workflow

1. **After training completes**, consolidate your checkpoint:
   ```bash
   python consolidate_checkpoint.py \
       --checkpoint_dir /mnt/data/models/checkpoints/checkpoint-2000 \
       --output_dir /mnt/data/models/checkpoints/final_model
   ```

2. **Run inference evaluation**:
   ```bash
   python inference_eval.py \
       --config config.yaml \
       --fine_tuned_model_path /mnt/data/models/checkpoints/final_model \
       --num_prompts 30 \
       --max_new_tokens 256
   ```

3. **View results in MLflow**:
   ```bash
   mlflow ui --backend-store-uri file:./mlruns
   ```

## Advanced Options

### Custom Test Prompts

The script uses predefined prompts by default. To use prompts from your dataset:

```bash
python inference_eval.py \
    --config config.yaml \
    --eval_dataset_path /path/to/your/dataset
```

The script will attempt to sample prompts from the dataset.

### Temperature and Sampling

Currently, the script uses:
- `temperature=0.7`
- `top_p=0.9`
- `do_sample=True`

To modify these, edit `inference_eval.py` in the `generate_text()` function.

## Next Steps

After running inference:
1. Review the generated text samples in MLflow
2. Compare metrics to see improvements
3. Adjust training hyperparameters if needed
4. Run additional evaluations with different prompts

