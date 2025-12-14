# Quick Start: Running Inference After Training

## Your Situation

Your training completed successfully! However, you saw:
- A checkpoint saving warning (now fixed in the code)
- "WARNING: No checkpoints found!" message
- Final model save was skipped (normal for FSDP)

## Step 1: Find Your Checkpoints

Checkpoints should be in your `output_dir` from config.yaml. Check:

```bash
# Check your output directory (from config.yaml)
ls -la /mnt/data/models/checkpoints/

# Or if using a different path, check:
ls -la <your_output_dir_from_config>
```

Look for directories named `checkpoint-XXXX` (e.g., `checkpoint-2000`, `checkpoint-4000`).

## Step 2: Consolidate a Checkpoint for Inference

FSDP saves checkpoints in a sharded format. You need to consolidate one for inference:

```bash
# Replace checkpoint-XXXX with your actual checkpoint number
python consolidate_checkpoint.py \
    /mnt/data/models/checkpoints/checkpoint-XXXX \
    /mnt/data/models/checkpoints/final_model \
    --config config.yaml
```

**Example:**
```bash
python consolidate_checkpoint.py \
    /mnt/data/models/checkpoints/checkpoint-2000 \
    /mnt/data/models/checkpoints/final_model \
    --config config.yaml
```

This creates a standard HuggingFace model at `/mnt/data/models/checkpoints/final_model`.

## Step 3: Run Inference

Now you can run inference on both models:

```bash
python inference_eval.py \
    --config config.yaml \
    --fine_tuned_model_path /mnt/data/models/checkpoints/final_model
```

Or if your paths are different:

```bash
python inference_eval.py \
    --config config.yaml \
    --base_model_path /mnt/data/models/1b \
    --fine_tuned_model_path /mnt/data/models/checkpoints/final_model \
    --num_prompts 30 \
    --max_new_tokens 256
```

## Step 4: View Results

View your results in MLflow:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open http://localhost:5000 in your browser.

## Troubleshooting

### "No checkpoints found" but training completed

This can happen if:
1. Checkpoints are in a different location
2. The checkpoint directory structure is different

**Solution:** Manually find your checkpoints:
```bash
# Search for checkpoint directories
find /mnt/data/models/checkpoints -type d -name "checkpoint-*" 2>/dev/null

# Or check the training output directory
ls -la <your_output_dir>
```

### Checkpoint consolidation fails

If consolidation fails, check:
1. The checkpoint path is correct
2. You have enough disk space
3. The checkpoint files are complete (not corrupted)

**Check checkpoint contents:**
```bash
ls -la /mnt/data/models/checkpoints/checkpoint-XXXX/
```

You should see files like:
- `model.safetensors` or `pytorch_model.bin`
- `config.json`
- `trainer_state.json`

### Using SLURM for Inference

If you're on a SLURM cluster:

```bash
# Set environment variables
export BASE_MODEL_PATH="/mnt/data/models/1b"
export FINE_TUNED_MODEL_PATH="/mnt/data/models/checkpoints/final_model"
export NUM_PROMPTS=30
export MAX_NEW_TOKENS=256

# Submit job
sbatch slurm_inference_eval.sh
```

## Quick Command Reference

```bash
# 1. Find latest checkpoint
ls -t /mnt/data/models/checkpoints/checkpoint-* | head -1

# 2. Consolidate checkpoint (replace XXXX with actual step number)
python consolidate_checkpoint.py \
    /mnt/data/models/checkpoints/checkpoint-XXXX \
    /mnt/data/models/checkpoints/final_model \
    --config config.yaml

# 3. Run inference
python inference_eval.py --config config.yaml

# 4. View results
mlflow ui --backend-store-uri file:./mlruns
```

