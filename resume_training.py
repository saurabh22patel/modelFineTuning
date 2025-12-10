#!/usr/bin/env python3
"""
Helper script to resume training from the latest checkpoint.
This script automatically finds the latest checkpoint and resumes training.
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory in the output directory."""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        checkpoint_path = os.path.join(output_dir, item)
        if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
            try:
                step_num = int(item.split("-")[1])
                # Check if it's a valid checkpoint (has trainer_state.json)
                state_file = os.path.join(checkpoint_path, "trainer_state.json")
                if os.path.exists(state_file):
                    checkpoints.append((step_num, checkpoint_path))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        return None
    
    # Sort by step number and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Resume training from the latest checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint path to resume from (default: latest checkpoint)"
    )
    parser.add_argument(
        "--use-slurm",
        action="store_true",
        help="Submit SLURM job using slurm_train.sh (default: run directly)"
    )
    parser.add_argument(
        "--slurm-script",
        type=str,
        default="slurm_train.sh",
        help="Path to SLURM training script (default: slurm_train.sh)"
    )
    
    args = parser.parse_args()
    
    # Load config to get output directory
    try:
        config = load_config(args.config)
        output_dir = config["training"]["output_dir"]
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
            sys.exit(1)
        print(f"Using specified checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = find_latest_checkpoint(output_dir)
        if not checkpoint_path:
            print(f"No checkpoint found in {output_dir}")
            print("Starting training from scratch...")
            checkpoint_path = None
        else:
            print(f"Found latest checkpoint: {checkpoint_path}")
    
    # Determine the training script to use
    if args.use_slurm:
        script_path = args.slurm_script
        if not os.path.exists(script_path):
            print(f"Error: {script_path} not found")
            sys.exit(1)
        
        # Build sbatch command
        sbatch_cmd = ["sbatch"]
        
        # Determine checkpoint argument for slurm_train.sh
        if checkpoint_path:
            # Use "latest" if it's the latest checkpoint, otherwise use full path
            checkpoint_arg = checkpoint_path
            # Check if this is actually the latest
            latest_checkpoint = find_latest_checkpoint(output_dir)
            if latest_checkpoint == checkpoint_path:
                checkpoint_arg = "latest"
            sbatch_cmd.extend([script_path, checkpoint_arg])
            print(f"\nSubmitting SLURM job to resume from checkpoint: {checkpoint_path}")
        else:
            sbatch_cmd.append(script_path)
            print(f"\nSubmitting SLURM job to start training from scratch")
        
        print(f"Command: {' '.join(sbatch_cmd)}")
        print()
        
        # Submit the job
        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip())
            print("\n✓ SLURM job submitted successfully!")
            print("\nMonitor your job with:")
            print("  squeue -u $USER")
            print("\nCheck job output with:")
            job_id = result.stdout.strip().split()[-1] if result.stdout else "JOBID"
            print(f"  tail -f logs/train_{job_id}.out")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed to submit SLURM job:")
            print(f"  {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            print("\n✗ Error: 'sbatch' command not found.")
            print("  Are you on a system with SLURM?")
            print("  If not, use without --use-slurm flag to run directly.")
            sys.exit(1)
    else:
        # Run training script directly
        script_path = "train.py"
        if not os.path.exists(script_path):
            print(f"Error: {script_path} not found")
            sys.exit(1)
        
        # Build command
        cmd = ["python", script_path, "--config", args.config]
        if checkpoint_path:
            cmd.extend(["--resume_from_checkpoint", checkpoint_path])
        else:
            print("No checkpoint found, starting from scratch...")
        
        print(f"\nResuming training with command:")
        print(" ".join(cmd))
        print()
        
        # Execute
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"\nTraining failed with error code {e.returncode}")
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()

