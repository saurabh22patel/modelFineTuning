# Flash Attention Installation Guide

## Problem
SSH sessions terminate when building flash-attention wheels because the build process takes 15-30 minutes and SSH connections can timeout.

## Solution: Use SLURM Job (Recommended)

The best way to install flash-attention on a SLURM cluster is to submit it as a job that runs independently of your SSH session:

```bash
sbatch slurm_install_flash_attn.sh
```

This will:
- Run in the background (survives SSH disconnections)
- Allocate resources (CPU, memory) for the build
- Save output to `logs/install_flash_attn_<job_id>.out`
- Complete even if you disconnect

### Monitor the job:
```bash
# Check job status
squeue -u $USER

# Watch the output
tail -f logs/install_flash_attn_*.out

# Check if it completed successfully
squeue -u $USER  # Should be empty when done
```

## Alternative Methods

### Option 2: Use nohup
If you want to run it interactively but survive disconnections:

```bash
source /root/llmtune/venv/bin/activate
nohup ./install_flash_attn.sh > logs/flash_attn_install.log 2>&1 &
```

Then check progress:
```bash
tail -f logs/flash_attn_install.log
```

### Option 3: Use screen/tmux
```bash
# Start a screen session
screen -S flash_attn

# Run the installation
source /root/llmtune/venv/bin/activate
./install_flash_attn.sh

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r flash_attn
```

## Verify Installation

After installation completes, verify it works:

```bash
source /root/llmtune/venv/bin/activate
python -c "import flash_attn; print('Flash attention installed!')"
```

## GPU Utilization Optimization

Once flash-attention is installed, you can increase GPU utilization by:

1. **Increase batch size** (if memory allows):
   - Current: `batch_size_per_device: 8` using ~40GB
   - Try: `batch_size_per_device: 16` or `20` to use more memory and increase utilization

2. **Reduce gradient accumulation**:
   - Current: `gradient_accumulation_steps: 8`
   - Try: `gradient_accumulation_steps: 4` to reduce idle time

3. **Increase data loading workers**:
   - Current: `dataloader_num_workers: 4`
   - Try: `dataloader_num_workers: 8` to keep data pipeline ahead

These changes should boost GPU utilization from 50% to 90%+.

