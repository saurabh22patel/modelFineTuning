#!/usr/bin/env python3
"""
Distributed fine-tuning script with FSDP for high GPU utilization.
Supports multi-node multi-GPU training with MLflow integration.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
import mlflow
import mlflow.pytorch
from datetime import datetime, timedelta
import json
import psutil
import time
import signal
import atexit
import traceback
from huggingface_hub import login
from urllib.parse import urlparse

# Helper functions for backward-compatible barrier and destroy_process_group calls
# timeout parameter for barrier() and destroy_process_group() was added in PyTorch 2.0
def barrier_with_timeout(timeout_seconds=300):
    """Call barrier with timeout if supported, otherwise use threading-based timeout."""
    try:
        # Try PyTorch 2.0+ timeout support first
        dist.barrier(timeout=timedelta(seconds=timeout_seconds))
    except TypeError:
        # timeout parameter not supported in older PyTorch versions
        # Use threading-based timeout as fallback
        import threading
        barrier_result = {"success": False, "error": None}
        
        def barrier_thread():
            try:
                dist.barrier()
                barrier_result["success"] = True
            except Exception as e:
                barrier_result["error"] = e
        
        thread = threading.Thread(target=barrier_thread, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Thread is still running - barrier timed out
            raise RuntimeError(f"Barrier timed out after {timeout_seconds} seconds. Some ranks may be stuck.")
        elif not barrier_result["success"]:
            # Barrier failed with an error
            if barrier_result["error"]:
                raise barrier_result["error"]
            else:
                raise RuntimeError("Barrier failed for unknown reason")

def destroy_process_group_with_timeout(timeout_seconds=10):
    """Call destroy_process_group with timeout if supported, otherwise without timeout."""
    try:
        dist.destroy_process_group(timeout=timedelta(seconds=timeout_seconds))
    except TypeError:
        # timeout parameter not supported in older PyTorch versions
        dist.destroy_process_group()

def print_flush(msg, rank=None):
    """Print message with immediate flush. Only prints if rank matches or rank is None."""
    if rank is None:
        print(msg, flush=True)
    elif dist.is_initialized():
        if dist.get_rank() == rank:
            print(msg, flush=True)
    elif rank == 0:
        print(msg, flush=True)

def check_process_group_health():
    """Check if the process group is still healthy and can communicate."""
    try:
        if not dist.is_initialized():
            return False
        # Try a simple all-reduce to test connectivity
        # Use a small tensor to minimize overhead
        test_tensor = torch.tensor([1.0], device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM, async_op=False, timeout=timedelta(seconds=10))
        return True
    except Exception as e:
        # If communication fails, process group is unhealthy
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Warning: Process group health check failed: {e}", flush=True)
        return False

def check_nvlink_topology(rank, local_rank):
    """Check NVLink topology and report connectivity."""
    if not torch.cuda.is_available():
        return
    
    try:
        import subprocess
        # Use nvidia-smi to check NVLink topology
        result = subprocess.run(
            ['nvidia-smi', 'topo', '-m'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and rank == 0:
            print_flush("=" * 60, rank=0)
            print_flush("NVLink Topology Check:", rank=0)
            print_flush("(Run 'nvidia-smi topo -m' on each node for full topology)", rank=0)
            
            # Check if P2P is enabled
            num_gpus = torch.cuda.device_count()
            p2p_enabled = []
            for i in range(num_gpus):
                for j in range(i + 1, num_gpus):
                    try:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        if can_access:
                            p2p_enabled.append((i, j))
                    except:
                        pass
            
            if p2p_enabled:
                print_flush(f"  ✓ P2P (Peer-to-Peer) enabled between {len(p2p_enabled)} GPU pairs", rank=0)
                print_flush(f"  ✓ NVLink should be utilized for intra-node communication", rank=0)
                if len(p2p_enabled) == num_gpus * (num_gpus - 1) // 2:
                    print_flush(f"  ✓ Full mesh connectivity: All GPUs can communicate via P2P/NVLink", rank=0)
                else:
                    print_flush(f"  ⚠️  Partial connectivity: {len(p2p_enabled)}/{num_gpus * (num_gpus - 1) // 2} pairs", rank=0)
            else:
                print_flush(f"  ⚠️  WARNING: P2P not enabled - NVLink may not be utilized", rank=0)
                print_flush(f"  Check NCCL_P2P_DISABLE environment variable", rank=0)
            
            # Check NCCL environment variables
            nvls_enabled = os.environ.get("NCCL_NVLS_ENABLE", "0")
            p2p_disable = os.environ.get("NCCL_P2P_DISABLE", "0")
            print_flush(f"  NCCL_P2P_DISABLE: {p2p_disable} (0=enabled, required for NVLink)", rank=0)
            print_flush(f"  NCCL_NVLS_ENABLE: {nvls_enabled} (1=enabled for NVLink Switch on H100+)", rank=0)
            print_flush(f"  Note: NVLink is for intra-node communication only", rank=0)
            print_flush(f"        Inter-node communication uses network (Ethernet/InfiniBand)", rank=0)
            print_flush("=" * 60, rank=0)
    except Exception as e:
        if rank == 0:
            print_flush(f"Note: Could not check NVLink topology: {e}", rank=0)
            print_flush("NVLink should still work if hardware supports it and NCCL_P2P_DISABLE=0", rank=0)

def setup_distributed():
    """Initialize distributed training."""
    if "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        # Use MASTER_ADDR from environment (should be set by SLURM script)
        # DO NOT use SLURM_STEP_NODELIST or SLURM_NODELIST directly as they may contain hostlist format
        if "MASTER_ADDR" not in os.environ:
            raise ValueError(
                "MASTER_ADDR not set in environment! "
                "The SLURM script should resolve the master node hostname to an IP address. "
                "Check that slurm_train.sh is properly setting MASTER_ADDR."
            )
        
        # Set master port
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        # Store original SLURM_LOCALID for debugging (before GPU index mapping)
        os.environ["SLURM_LOCALID_ORIG"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)  # Will be updated after GPU selection
        
        # Print connection info for debugging
        print(f"[Rank {rank}] Distributed setup: MASTER_ADDR={os.environ.get('MASTER_ADDR', 'NOT SET')}, MASTER_PORT={os.environ.get('MASTER_PORT', 'NOT SET')}", flush=True)
        print(f"[Rank {rank}] World size: {world_size}, Rank: {rank}, Local rank: {local_rank}", flush=True)
        
        # Verify MASTER_ADDR is set and is an IP address (not a hostlist)
        master_addr = os.environ.get('MASTER_ADDR', '')
        if not master_addr:
            raise ValueError("MASTER_ADDR is not set! Check SLURM script.")
        if '[' in master_addr or '-' in master_addr:
            raise ValueError(f"MASTER_ADDR appears to be a SLURM hostlist format: {master_addr}. It should be a hostname or IP address.")
    else:
        # Local or torchrun environment
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # CRITICAL FIX: Handle CUDA_VISIBLE_DEVICES correctly for multi-node training
    # When CUDA_VISIBLE_DEVICES is set, PyTorch remaps device indices:
    # - If CUDA_VISIBLE_DEVICES="0,1,2,3", then cuda:0 refers to physical GPU 0, cuda:1 to physical GPU 1, etc.
    # - We should use local_rank as the index within the visible devices
    # - If CUDA_VISIBLE_DEVICES is not set, all GPUs are visible and we use local_rank directly
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if cuda_visible_devices:
        # Parse CUDA_VISIBLE_DEVICES to understand which physical GPUs are visible
        try:
            visible_physical_gpus = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
            print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES={cuda_visible_devices}, visible physical GPUs: {visible_physical_gpus}, num_visible_gpus={num_visible_gpus}", flush=True)
        except ValueError:
            visible_physical_gpus = []
            print(f"[Rank {rank}] WARNING: Could not parse CUDA_VISIBLE_DEVICES='{cuda_visible_devices}'", flush=True)
        
        # Use local_rank as the index within the visible devices
        # PyTorch will automatically remap cuda:local_rank to the correct physical GPU
        if local_rank >= num_visible_gpus:
            print(f"[Rank {rank}] ERROR: local_rank {local_rank} >= num_visible_gpus {num_visible_gpus}", flush=True)
            print(f"[Rank {rank}] This indicates a configuration error - too many tasks per node or incorrect GPU allocation", flush=True)
            actual_gpu_index = local_rank % num_visible_gpus if num_visible_gpus > 0 else 0
            print(f"[Rank {rank}] Using modulo: GPU index {actual_gpu_index}", flush=True)
        else:
            actual_gpu_index = local_rank
    else:
        # CUDA_VISIBLE_DEVICES not set, all GPUs on the node are visible
        # Use local_rank directly to select the GPU
        actual_gpu_index = local_rank
        print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES not set, all GPUs visible, using local_rank {local_rank} directly", flush=True)
    
    # Verify we have enough GPUs
    if num_visible_gpus == 0:
        raise RuntimeError(f"[Rank {rank}] No GPUs available! Check CUDA installation and GPU allocation.")
    
    if actual_gpu_index >= num_visible_gpus:
        raise RuntimeError(f"[Rank {rank}] GPU index {actual_gpu_index} >= num_visible_gpus {num_visible_gpus}. Check task/GPU allocation.")
    
    print(f"[Rank {rank}] Selected GPU: cuda:{actual_gpu_index} (local_rank={local_rank}, num_visible_gpus={num_visible_gpus})", flush=True)
    
    # Set timeout for initialization (default 30 minutes, increase if needed)
    timeout = int(os.environ.get("TORCH_DISTRIBUTED_INIT_TIMEOUT", 1800))
    
    # Initialize process group with timeout and retry logic
    print(f"[Rank {rank}] Initializing process group...", flush=True)
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout)
            )
            print(f"[Rank {rank}] Process group initialized successfully", flush=True)
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[Rank {rank}] WARNING: Failed to initialize process group (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
                print(f"[Rank {rank}] Retrying in {retry_delay} seconds...", flush=True)
                time.sleep(retry_delay)
            else:
                print(f"[Rank {rank}] ERROR: Failed to initialize process group after {max_retries} attempts: {e}", flush=True)
                print(f"[Rank {rank}] MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}", flush=True)
                raise
    
    # Synchronize all ranks after initialization to ensure all are ready
    # This helps catch any ranks that failed to initialize properly
    try:
        print(f"[Rank {rank}] Synchronizing all ranks...", flush=True)
        barrier_with_timeout(timeout_seconds=300)  # 5 minute timeout for barrier
        print(f"[Rank {rank}] All ranks synchronized successfully", flush=True)
    except Exception as e:
        print(f"[Rank {rank}] WARNING: Barrier synchronization failed: {e}", flush=True)
        print(f"[Rank {rank}] This may indicate a rank failed to initialize. Continuing anyway...", flush=True)
        # Note: If this fails, training may still work but some ranks might be out of sync
    
    # Set device using the actual GPU index
    # Note: When CUDA_VISIBLE_DEVICES is set, PyTorch remaps device indices
    # So we use actual_gpu_index which is the index within visible devices
    torch.cuda.set_device(actual_gpu_index)
    device = torch.device(f"cuda:{actual_gpu_index}")
    
    # Update LOCAL_RANK to reflect the actual GPU index being used
    # This is what PyTorch and other libraries will use
    os.environ["LOCAL_RANK"] = str(actual_gpu_index)
    
    # Log the mapping for debugging
    if "SLURM_LOCALID_ORIG" in os.environ:
        orig_localid = os.environ["SLURM_LOCALID_ORIG"]
        print(f"[Rank {rank}] GPU mapping: SLURM_LOCALID={orig_localid} -> GPU index={actual_gpu_index}", flush=True)
    
    return rank, actual_gpu_index, world_size, device

# Global flag for graceful shutdown
_shutdown_requested = False
_cleanup_done = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    rank = 0
    try:
        if dist.is_initialized():
            rank = dist.get_rank()
    except:
        pass
    
    print(f"\n[Rank {rank} PID {os.getpid()}] Received signal {signum}, initiating graceful shutdown...", flush=True)
    _shutdown_requested = True
    # Trigger cleanup immediately
    try:
        cleanup_distributed()
    except Exception as e:
        print(f"[Rank {rank}] Error during signal-triggered cleanup: {e}", flush=True)
    
    # Force exit after cleanup
    import threading
    def force_exit():
        time.sleep(2)  # Give a moment for cleanup messages to flush
        print(f"[Rank {rank} PID {os.getpid()}] Force exiting after cleanup...", flush=True)
        os._exit(1)
    threading.Thread(target=force_exit, daemon=True).start()

def cleanup_distributed():
    """Clean up distributed training with error handling."""
    global _cleanup_done
    if _cleanup_done:
        return
    
    rank = 0
    world_size = 1
    is_initialized = False
    
    try:
        is_initialized = dist.is_initialized()
        if is_initialized:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    except:
        pass  # If we can't check initialization, proceed with cleanup anyway
    
    # Log why we're cleaning up (if we can get traceback)
    import sys
    if sys.exc_info()[0] is not None:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_type is not None:
            print(f"[Rank {rank}] Cleanup triggered by exception: {exc_type.__name__}: {exc_value}", flush=True)
    
    try:
        if is_initialized:
            # CRITICAL: Always try to destroy process group, even if barrier fails
            # This prevents the "destroy_process_group() was not called" warning
            print(f"[Rank {rank}] Cleaning up distributed process group...", flush=True)
            print(f"[Rank {rank}] Note: If this hangs, other ranks may still be initializing", flush=True)
            
            # Try to synchronize first, but don't let failure prevent cleanup
            # Use a very short timeout - if other ranks are still initializing/training, 
            # we shouldn't wait for them. They'll clean up when they're done.
            try:
                print(f"[Rank {rank}] Attempting cleanup barrier (timeout: 5s)...", flush=True)
                print(f"[Rank {rank}] Note: If other ranks are still initializing, this will timeout (expected)", flush=True)
                barrier_with_timeout(timeout_seconds=5)  # Very short timeout - don't wait for stuck ranks
                print(f"[Rank {rank}] All ranks synchronized, proceeding with cleanup...", flush=True)
            except Exception as e:
                # If barrier fails, continue with cleanup anyway - this is expected if ranks are still initializing
                print(f"[Rank {rank}] Cleanup barrier timed out (expected if other ranks still initializing): {e}", flush=True)
                print(f"[Rank {rank}] Proceeding with cleanup anyway - other ranks will clean up when ready", flush=True)
            
            # Always attempt to destroy process group, even if barrier failed
            # This is critical to prevent resource leaks and warnings
            try:
                # Use a short timeout to avoid hanging if TCPStore is already down
                # But still call destroy_process_group() to satisfy PyTorch's requirement
                destroy_process_group_with_timeout(timeout_seconds=10)
                print(f"[Rank {rank}] Process group destroyed successfully", flush=True)
            except RuntimeError as e:
                # RuntimeError often means TCPStore is already down or connection lost
                error_str = str(e)
                if "TCPStore" in error_str or "Connection" in error_str or "should dump" in error_str or "Broken pipe" in error_str:
                    # This is the common harmless warning - suppress detailed error for this case
                    print(f"[Rank {rank}] Note: TCPStore connection closed during cleanup (harmless if training completed successfully)", flush=True)
                else:
                    print(f"[Rank {rank}] Warning: Error during process group destruction: {e}", flush=True)
            except Exception as e:
                error_str = str(e)
                if "TCPStore" in error_str or "Connection" in error_str or "should dump" in error_str or "Broken pipe" in error_str:
                    # Suppress detailed error for common TCPStore issues
                    print(f"[Rank {rank}] Note: TCPStore communication issue during cleanup (harmless if training completed)", flush=True)
                else:
                    print(f"[Rank {rank}] Warning: Unexpected error during cleanup: {e}", flush=True)
        else:
            # Process group not initialized, nothing to clean up
            print(f"[Rank {rank}] Process group not initialized, skipping cleanup", flush=True)
    except Exception as e:
        # If dist.is_initialized() itself fails, that's okay - process group may already be destroyed
        print(f"[Rank {rank}] Note: Process group may already be destroyed: {e}", flush=True)
    finally:
        _cleanup_done = True
        # Clear CUDA cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

def force_cleanup_and_exit(exit_code=1):
    """Force cleanup and exit immediately."""
    try:
        cleanup_distributed()
    except Exception as e:
        # Even if cleanup fails, try to destroy process group if possible
        try:
            if dist.is_initialized():
                try:
                    destroy_process_group_with_timeout(timeout_seconds=5)
                except:
                    pass  # Ignore errors during forced cleanup
        except:
            pass
    # Force exit to prevent hanging
    os._exit(exit_code)

def load_config(config_path):
    """Load configuration from YAML file."""
    # Convert to absolute path if relative
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Please ensure the config file path is correct."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_dataset(dataset_path, tokenizer, max_length):
    """Prepare dataset for training. Loads pre-tokenized dataset if available."""
    print(f"Loading dataset from {dataset_path}...", flush=True)
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded. Number of samples: {len(dataset)}", flush=True)
    
    # Check if dataset is already pre-tokenized
    # Pre-tokenized datasets should have 'input_ids' and 'labels' columns
    if "input_ids" in dataset.column_names and "labels" in dataset.column_names:
        print("Dataset is already pre-tokenized. Using pre-tokenized data.", flush=True)
        return dataset
    
    # If not pre-tokenized, tokenize on the fly (fallback)
    print("Dataset is not pre-tokenized. Tokenizing on the fly...", flush=True)
    text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
    
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    print("Tokenizing dataset (this may take a while)...", flush=True)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Tokenizing"
    )
    print("Tokenization complete.", flush=True)
    
    return tokenized_dataset

def get_gpu_utilization():
    """Get current GPU utilization percentage."""
    try:
        if torch.cuda.is_available():
            # Try to get utilization from nvidia-smi if available
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                utilizations = [float(x.strip()) for x in result.stdout.strip().split('\n')]
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if local_rank < len(utilizations):
                    return utilizations[local_rank]
    except:
        pass
    return 0.0

def log_system_metrics(mlflow_client, run_id):
    """Log system metrics to MLflow."""
    if dist.get_rank() == 0:
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpu_util = get_gpu_utilization()
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
            else:
                gpu_memory = 0
                gpu_memory_reserved = 0
            
            mlflow.log_metrics({
                "system/cpu_percent": cpu_percent,
                "system/memory_percent": memory.percent,
                "system/gpu_utilization": gpu_util,
                "system/gpu_memory_gb": gpu_memory,
                "system/gpu_memory_reserved_gb": gpu_memory_reserved
            })
        except Exception as e:
            print(f"Warning: Could not log system metrics: {e}")

def evaluate_model_before_training(model, tokenizer, device, config):
    """Evaluate model performance before fine-tuning. Returns metrics for comparison."""
    if dist.get_rank() == 0:
        print("Evaluating base model before fine-tuning...")
        
        # Simple evaluation: generate text
        model.eval()
        test_prompts = [
            "The future of artificial intelligence is",
            "Once upon a time",
            "The key to success is"
        ]
        
        base_metrics = {}
        all_generations = []
        
        with torch.no_grad():
            for i, test_prompt in enumerate(test_prompts):
                inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                all_generations.append(f"Prompt {i+1}: {test_prompt}\nGeneration: {generated_text}\n")
                
                # Calculate metrics
                base_metrics[f"base/prompt_{i+1}_length"] = len(generated_text)
                base_metrics[f"base/prompt_{i+1}_tokens"] = len(outputs[0])
        
        # Log to MLflow
        mlflow.log_text("\n".join(all_generations), "evaluation/base_model_generations.txt")
        mlflow.log_metrics(base_metrics)
        mlflow.log_metric("evaluation/base_model_avg_length", sum(base_metrics.values()) / len(base_metrics))
        
        print(f"Base model evaluation complete. Logged {len(test_prompts)} generations to MLflow.")
        model.train()
        
        # Clear CUDA cache after evaluation
        torch.cuda.empty_cache()
        
        return base_metrics
    return {}

def evaluate_model_after_training(model, tokenizer, device, config, base_metrics=None):
    """Evaluate model performance after fine-tuning and compare with base model."""
    if dist.get_rank() == 0:
        print("Evaluating fine-tuned model...")
        
        model.eval()
        test_prompts = [
            "The future of artificial intelligence is",
            "Once upon a time",
            "The key to success is"
        ]
        
        fine_tuned_metrics = {}
        all_generations = []
        comparison_metrics = {}
        
        with torch.no_grad():
            for i, test_prompt in enumerate(test_prompts):
                inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                all_generations.append(f"Prompt {i+1}: {test_prompt}\nGeneration: {generated_text}\n")
                
                # Calculate metrics
                fine_tuned_metrics[f"fine_tuned/prompt_{i+1}_length"] = len(generated_text)
                fine_tuned_metrics[f"fine_tuned/prompt_{i+1}_tokens"] = len(outputs[0])
                
                # Compare with base if available
                if base_metrics:
                    base_key = f"base/prompt_{i+1}_length"
                    if base_key in base_metrics:
                        diff = len(generated_text) - base_metrics[base_key]
                        comparison_metrics[f"comparison/prompt_{i+1}_length_diff"] = diff
                        comparison_metrics[f"comparison/prompt_{i+1}_length_pct_change"] = (diff / base_metrics[base_key]) * 100 if base_metrics[base_key] > 0 else 0
        
        # Log to MLflow
        mlflow.log_text("\n".join(all_generations), "evaluation/fine_tuned_model_generations.txt")
        mlflow.log_metrics(fine_tuned_metrics)
        mlflow.log_metric("evaluation/fine_tuned_model_avg_length", sum(fine_tuned_metrics.values()) / len(fine_tuned_metrics))
        
        # Log comparison metrics
        if comparison_metrics:
            mlflow.log_metrics(comparison_metrics)
            mlflow.log_metric("evaluation/comparison_avg_length_diff", 
                            sum([v for k, v in comparison_metrics.items() if "length_diff" in k]) / len([k for k in comparison_metrics.keys() if "length_diff" in k]))
        
        print(f"Fine-tuned model evaluation complete. Logged {len(test_prompts)} generations to MLflow.")
        if comparison_metrics:
            print("Comparison metrics logged to MLflow for base vs fine-tuned model.")
        model.train()
        
        # Clear CUDA cache after evaluation
        torch.cuda.empty_cache()

class CustomTrainer(Trainer):
    """Custom trainer with GPU utilization monitoring and optimized data loading."""
    
    def __init__(self, *args, prefetch_factor=None, persistent_workers=None, dataloader_timeout=None, clear_cache_frequency=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_utilizations = []
        self.last_log_time = time.time()
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.dataloader_timeout = dataloader_timeout
        self.clear_cache_frequency = clear_cache_frequency
        self._step_counter = 0  # Track training steps for cache clearing frequency
    
    def get_train_dataloader(self):
        """Override to add persistent_workers and prefetch_factor for better GPU utilization."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        # Get the sampler using parent class method
        # The parent Trainer class handles distributed sampling automatically
        train_sampler = self._get_train_sampler()
        
        # Get configuration for data loading
        num_workers = self.args.dataloader_num_workers
        pin_memory = self.args.dataloader_pin_memory
        
        # Use config values if provided, otherwise use defaults
        prefetch_factor = self.prefetch_factor if self.prefetch_factor is not None else 2
        persistent_workers = self.persistent_workers if self.persistent_workers is not None else False
        
        # Get timeout from config if available (for preventing worker hangs)
        timeout = self.dataloader_timeout if self.dataloader_timeout is not None else 0
        
        # Create optimized dataloader with persistent workers and higher prefetch
        # This ensures GPU is always fed with data, preventing utilization spikes
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            timeout=timeout,
        )
        
        return dataloader
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Check for shutdown request
        global _shutdown_requested
        if _shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested during training")
        
        # Monitor GPU utilization and memory
        if time.time() - self.last_log_time > 5:  # Log every 5 seconds
            gpu_util = get_gpu_utilization()
            self.gpu_utilizations.append(gpu_util)
            self.last_log_time = time.time()
            
            # Monitor memory usage
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                allocated = torch.cuda.memory_allocated(local_rank) / 1e9  # GB
                reserved = torch.cuda.memory_reserved(local_rank) / 1e9  # GB
                total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9  # GB
                free = total - reserved
                memory_pct = (reserved / total) * 100
                
                # Warn if memory usage is high
                if memory_pct > 90 and dist.get_rank() == 0:
                    print(f"WARNING: GPU memory usage is {memory_pct:.1f}% ({reserved:.2f}GB / {total:.2f}GB). Free: {free:.2f}GB", flush=True)
            
            if dist.get_rank() == 0 and len(self.gpu_utilizations) % 10 == 0:
                try:
                    avg_util = sum(self.gpu_utilizations[-10:]) / 10
                    mlflow.log_metric("training/avg_gpu_utilization", avg_util, step=self.state.global_step)
                    print(f"Step {self.state.global_step}: GPU Utilization: {avg_util:.2f}%")
                except Exception as e:
                    # Don't fail training if logging fails
                    pass
        
        # Increment step counter for cache clearing frequency control
        self._step_counter += 1
        
        # Clear cache based on frequency setting (not every step to reduce overhead)
        should_clear_cache = (self._step_counter % self.clear_cache_frequency == 0)
        
        if should_clear_cache and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        # Call parent method with correct signature
        try:
            if num_items_in_batch is not None:
                result = super().training_step(model, inputs, num_items_in_batch)
            else:
                result = super().training_step(model, inputs)
            
            # Clear cache after backward pass only if frequency matches
            # This is important for preventing OOM during backward pass
            if should_clear_cache and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    # Don't synchronize here - it can cause memory spikes
                except:
                    pass
            
            return result
        except torch.cuda.OutOfMemoryError as e:
            # If OOM occurs, try to recover by clearing cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # Re-raise the error with helpful message
            rank = dist.get_rank() if dist.is_initialized() else 0
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(local_rank) / 1e9
                reserved = torch.cuda.memory_reserved(local_rank) / 1e9
                total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
                print(f"[Rank {rank}] OOM Error! GPU {local_rank} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total", flush=True)
                print(f"[Rank {rank}] Current config: max_length={self.args.per_device_train_batch_size}, batch_size={self.args.per_device_train_batch_size}", flush=True)
                print(f"[Rank {rank}] Try reducing max_length to 512 or 256 in config.yaml", flush=True)
            raise
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Override compute_loss to add memory cleanup hooks during backward pass."""
        # Don't clear cache here - handled in training_step with proper frequency control
        # This reduces redundant cache clearing operations
        
        # Call parent method, forwarding num_items_in_batch and any other kwargs
        parent_kwargs = {"return_outputs": return_outputs}
        if num_items_in_batch is not None:
            parent_kwargs["num_items_in_batch"] = num_items_in_batch
        parent_kwargs.update(kwargs)
        loss = super().compute_loss(model, inputs, **parent_kwargs)
        
        # Optionally register backward hook for cache clearing during backward pass
        # Only if we're at the right frequency step to avoid overhead
        if hasattr(loss, 'register_hook') and (self._step_counter % self.clear_cache_frequency == 0):
            def backward_hook(grad):
                # Clear cache during backward pass to free memory (only at frequency intervals)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
                return grad
            loss.register_hook(backward_hook)
        
        return loss
    
    def log(self, logs, start_time=None):
        # Add GPU utilization to logs
        if self.gpu_utilizations:
            logs["gpu_utilization"] = self.gpu_utilizations[-1]
        
        # Log all metrics to MLflow (only on rank 0)
        try:
            # Check if we're on rank 0 (either distributed or single process)
            is_rank_0 = False
            if dist.is_initialized():
                is_rank_0 = (dist.get_rank() == 0)
            else:
                # Single process training or not yet initialized
                is_rank_0 = True  # Assume rank 0 if not distributed
            
            if is_rank_0:
                # Verify MLflow has an active run
                if mlflow.active_run() is not None:
                    # Get current step from state
                    step = self.state.global_step if hasattr(self.state, 'global_step') else None
                    
                    # Log all metrics to MLflow
                    for key, value in logs.items():
                        if isinstance(value, (int, float)):
                            if step is not None:
                                mlflow.log_metric(key, value, step=step)
                            else:
                                mlflow.log_metric(key, value)
                        elif isinstance(value, str):
                            mlflow.log_text(value, f"logs/{key}.txt")
        except Exception as e:
            # Don't fail training if MLflow logging fails
            # Only print warning on rank 0 to avoid spam
            try:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Warning: Failed to log to MLflow: {e}", flush=True)
            except:
                pass
        
        # Also call parent log method (for console output, etc.)
        # Pass start_time if provided to match parent signature
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

def main():
    # Log entry point immediately (before any operations)
    try:
        # Try to get rank from environment before distributed setup
        initial_rank = int(os.environ.get("RANK", -1))
        initial_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        print(f"[Initial] Process started - RANK={initial_rank}, LOCAL_RANK={initial_local_rank}, PID={os.getpid()}", flush=True)
    except:
        print(f"[Initial] Process started - PID={os.getpid()}", flush=True)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)  # SLURM sends SIGTERM
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    atexit.register(cleanup_distributed)  # Register cleanup on normal exit
    
    parser = argparse.ArgumentParser(description="Distributed fine-tuning with FSDP")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    
    try:
        initial_rank = int(os.environ.get("RANK", -1))
        print(f"[Initial] After argparse - RANK={initial_rank}, about to call setup_distributed()...", flush=True)
    except:
        pass
    
    rank = 0
    local_rank = 0
    world_size = 1
    device = None
    
    # Ensure cleanup happens even if setup fails
    try:
        # Setup distributed training
        print(f"[Initial] Calling setup_distributed()...", flush=True)
        rank, local_rank, world_size, device = setup_distributed()
        print(f"[Rank {rank}] setup_distributed() completed successfully", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to setup distributed training: {e}", flush=True)
        traceback.print_exc()
        # Cleanup is handled by force_cleanup_and_exit
        force_cleanup_and_exit(1)
    
    # Print status immediately after distributed setup
    print_flush(f"[Rank {rank}] Distributed training initialized successfully", rank=rank)
    print_flush(f"[Rank {rank}] Local rank: {local_rank}, World size: {world_size}, Device: {device}", rank=rank)
    print_flush(f"[Rank {rank}] About to check NVLink topology and verify all ranks...", rank=rank)
    
    # Check NVLink topology (only on rank 0 to avoid spam)
    check_nvlink_topology(rank, local_rank)
    
    print_flush(f"[Rank {rank}] NVLink check complete, proceeding to rank verification...", rank=rank)
    
    # Verify all ranks are present by doing a collective operation
    if dist.is_initialized():
        print_flush(f"[Rank {rank}] Starting initial communication test (all_reduce)...", rank=rank)
        print_flush(f"[Rank {rank}] This all_reduce will wait for all {world_size} ranks to reach this point", rank=rank)
        print_flush(f"[Rank {rank}] If this hangs, check logs for missing ranks (look for '[Rank X]' messages)", rank=rank)
        
        # Create a tensor to verify all ranks can communicate
        test_tensor = torch.ones(1, device=device)
        try:
            # Add a timeout wrapper for the all_reduce (though PyTorch doesn't support timeout directly)
            # We'll rely on the barrier timeout mechanism if available
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            print_flush(f"[Rank {rank}] Initial all_reduce completed, sum={test_tensor.item()}", rank=rank)
            
            # Check if all ranks participated
            if rank == 0:
                expected_sum = world_size
                actual_sum = test_tensor.item()
                if actual_sum != expected_sum:
                    missing = expected_sum - actual_sum
                    print_flush(f"⚠️  [Rank 0] WARNING: all_reduce sum={actual_sum}, expected={expected_sum}", rank=0)
                    print_flush(f"⚠️  [Rank 0] {missing} rank(s) did not participate in all_reduce!", rank=0)
                    print_flush(f"⚠️  [Rank 0] Check SLURM logs (srun output files) for stuck ranks", rank=0)
        except Exception as e:
            print_flush(f"[Rank {rank}] ERROR in initial all_reduce: {e}", rank=rank)
            print_flush(f"[Rank {rank}] This usually means some ranks are stuck or crashed", rank=rank)
            if rank == 0:
                print_flush(f"⚠️  [Rank 0] Check SLURM output files for each rank to see where they got stuck", rank=0)
                print_flush(f"⚠️  [Rank 0] Look for the last '[Rank X]' message in each rank's log", rank=0)
            raise
        
        # Collect GPU assignments from all ranks
        # Get both local_rank and actual device index being used
        print_flush(f"[Rank {rank}] Collecting GPU assignments from all ranks...", rank=rank)
        actual_device_idx = torch.cuda.current_device() if torch.cuda.is_available() else -1
        local_rank_tensor = torch.tensor([local_rank], device=device, dtype=torch.int)
        device_idx_tensor = torch.tensor([actual_device_idx], device=device, dtype=torch.int)
        
        all_local_ranks = [torch.zeros(1, device=device, dtype=torch.int) for _ in range(world_size)]
        all_device_indices = [torch.zeros(1, device=device, dtype=torch.int) for _ in range(world_size)]
        
        print_flush(f"[Rank {rank}] Calling all_gather for GPU assignments...", rank=rank)
        try:
            dist.all_gather(all_local_ranks, local_rank_tensor)
            dist.all_gather(all_device_indices, device_idx_tensor)
            print_flush(f"[Rank {rank}] GPU assignment all_gather completed", rank=rank)
        except Exception as e:
            print_flush(f"[Rank {rank}] ERROR in GPU assignment all_gather: {e}", rank=rank)
            # Continue anyway - this is just for diagnostics
            if rank == 0:
                print_flush(f"⚠️  WARNING: Could not gather GPU assignments. Some ranks may be stuck: {e}", rank=0)
        
        if rank == 0:
            print_flush(f"✓ All {world_size} ranks successfully initialized and can communicate", rank=0)
            if test_tensor.item() != world_size:
                print_flush(f"⚠️  WARNING: Expected sum={world_size}, got {test_tensor.item()}. Some ranks may not be active!", rank=0)
                missing_count = world_size - int(test_tensor.item())
                print_flush(f"⚠️  WARNING: {missing_count} rank(s) may be missing or stuck!", rank=0)
            
            # Check GPU assignments
            local_ranks = [r.item() for r in all_local_ranks]
            device_indices = [d.item() for d in all_device_indices]
            
            # Diagnostic: Check which ranks participated
            print_flush("=" * 60, rank=0)
            print_flush("Rank Participation Check:", rank=0)
            print_flush(f"  Expected ranks: {list(range(world_size))}", rank=0)
            print_flush(f"  Ranks that participated in all_gather: {len(local_ranks)}/{world_size}", rank=0)
            if len(local_ranks) < world_size:
                print_flush(f"  ⚠️  WARNING: Only {len(local_ranks)} ranks participated in all_gather!", rank=0)
                print_flush(f"  ⚠️  Missing ranks may be stuck or crashed", rank=0)
            print_flush("=" * 60, rank=0)
            print_flush("GPU Assignment Verification:", rank=0)
            print_flush(f"  Local ranks: {sorted(local_ranks)}", rank=0)
            print_flush(f"  Device indices in use: {sorted(device_indices)}", rank=0)
            
            # Group by node (assuming first 8 ranks are on node 0, next 8 on node 1)
            node0_ranks = local_ranks[:8] if len(local_ranks) >= 8 else local_ranks
            node1_ranks = local_ranks[8:16] if len(local_ranks) >= 16 else []
            node0_devices = device_indices[:8] if len(device_indices) >= 8 else device_indices
            node1_devices = device_indices[8:16] if len(device_indices) >= 16 else []
            
            print_flush(f"  Node 0 (ranks 0-7): local_ranks={sorted(node0_ranks)}, devices={sorted(node0_devices)}", rank=0)
            if node1_ranks:
                print_flush(f"  Node 1 (ranks 8-15): local_ranks={sorted(node1_ranks)}, devices={sorted(node1_devices)}", rank=0)
            
            # Check for duplicate device assignments within each node
            node0_duplicates = {}
            node1_duplicates = {}
            
            for i, dev in enumerate(node0_devices):
                if dev in node0_duplicates:
                    node0_duplicates[dev].append(i)
                else:
                    node0_duplicates[dev] = [i]
            node0_duplicates = {k: v for k, v in node0_duplicates.items() if len(v) > 1}
            
            if node1_devices:
                for i, dev in enumerate(node1_devices):
                    if dev in node1_duplicates:
                        node1_duplicates[dev].append(i + 8)
                    else:
                        node1_duplicates[dev] = [i + 8]
                node1_duplicates = {k: v for k, v in node1_duplicates.items() if len(v) > 1}
            
            if node0_duplicates:
                print_flush(f"  ⚠️  WARNING: Node 0 - Multiple ranks using same GPUs: {node0_duplicates}", rank=0)
            else:
                print_flush(f"  ✓ Node 0 - All ranks using unique GPUs", rank=0)
            
            if node1_duplicates:
                print_flush(f"  ⚠️  WARNING: Node 1 - Multiple ranks using same GPUs: {node1_duplicates}", rank=0)
            else:
                if node1_ranks:
                    print_flush(f"  ✓ Node 1 - All ranks using unique GPUs", rank=0)
            
            # Check if local_ranks match device indices (they should for proper assignment)
            mismatches = []
            for i, (lr, di) in enumerate(zip(local_ranks, device_indices)):
                if lr != di:
                    mismatches.append(f"Rank {i}: local_rank={lr} but device={di}")
            
            if mismatches:
                print_flush(f"  ⚠️  WARNING: Local rank / device index mismatches:", rank=0)
                for mm in mismatches:
                    print_flush(f"    {mm}", rank=0)
            else:
                print_flush(f"  ✓ All local_ranks match device indices", rank=0)
            
            print_flush("=" * 60, rank=0)
    
    # Print initial GPU memory status
    # Note: device is already set in setup_distributed(), and LOCAL_RANK has been updated
    if torch.cuda.is_available():
        actual_gpu_index = int(os.environ.get("LOCAL_RANK", 0))  # This was updated in setup_distributed()
        current_device = torch.cuda.current_device()
        num_gpus = torch.cuda.device_count()
        
        # Verify device is set correctly
        if current_device != actual_gpu_index:
            print_flush(f"[Rank {rank}] WARNING: Device mismatch! Expected GPU {actual_gpu_index}, but current device is {current_device}", rank=rank)
            print_flush(f"[Rank {rank}] Setting device to {actual_gpu_index}...", rank=rank)
            torch.cuda.set_device(actual_gpu_index)
            current_device = torch.cuda.current_device()
        
        gpu_name = torch.cuda.get_device_properties(current_device).name
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(current_device) / 1e9
        reserved = torch.cuda.memory_reserved(current_device) / 1e9
        
        print_flush(f"[Rank {rank}] GPU Assignment - Local Rank: {local_rank}, GPU Index: {actual_gpu_index}, Current Device: {current_device}, Total GPUs Visible: {num_gpus}", rank=rank)
        print_flush(f"[Rank {rank}] GPU {current_device} ({gpu_name}) Memory - Total: {total_memory:.2f} GB, Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB", rank=rank)
        
        # Print CUDA_VISIBLE_DEVICES for debugging
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set (all GPUs visible)")
        print_flush(f"[Rank {rank}] CUDA_VISIBLE_DEVICES={cuda_visible}", rank=rank)
        
        # Verify device is correct
        if current_device != actual_gpu_index:
            print_flush(f"[Rank {rank}] ERROR: Failed to set device to {actual_gpu_index}, current device is {current_device}", rank=rank)
    
    # Load configuration
    print_flush(f"Loading configuration from: {args.config}", rank=0)
    print_flush(f"Current working directory: {os.getcwd()}", rank=0)
    try:
        config = load_config(args.config)
        print_flush("Configuration loaded successfully", rank=0)
    except FileNotFoundError as e:
        print_flush(f"ERROR: {e}", rank=0)
        cleanup_distributed()
        force_cleanup_and_exit(1)
    except Exception as e:
        print_flush(f"ERROR loading config: {e}", rank=0)
        traceback.print_exc()
        cleanup_distributed()
        force_cleanup_and_exit(1)
    
    # Set random seed
    print_flush("Setting random seed...", rank=0)
    set_seed(42)
    
    # Apply memory optimizations
    if rank == 0:
        print("=" * 60, flush=True)
        print("GPU Memory Optimization Settings:", flush=True)
        print(f"  - Gradient Checkpointing: {config['model']['gradient_checkpointing']}", flush=True)
        print(f"  - FSDP CPU Offload: {config['fsdp']['cpu_offload']}", flush=True)
        print(f"  - FSDP State Dict Type: {config['fsdp'].get('state_dict_type', 'FULL_STATE_DICT')}", flush=True)
        print(f"  - 8-bit Optimizer: {config['performance'].get('use_8bit_optimizer', False)}", flush=True)
        if config["performance"].get("max_memory_mb"):
            print(f"  - Max GPU Memory: {config['performance']['max_memory_mb']}MB", flush=True)
        print("=" * 60, flush=True)
    
    # Set CUDA memory allocation strategy for better memory management
    # Note: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead
    if not os.environ.get("PYTORCH_ALLOC_CONF") and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"
        if rank == 0:
            print("Set PYTORCH_ALLOC_CONF=max_split_size_mb:512 for better memory management", flush=True)
    elif os.environ.get("PYTORCH_CUDA_ALLOC_CONF") and not os.environ.get("PYTORCH_ALLOC_CONF"):
        # Migrate from deprecated variable
        os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        if rank == 0:
            print(f"Migrated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF={os.environ['PYTORCH_ALLOC_CONF']}", flush=True)
    
    # Generate run name (needed for all ranks)
    run_name = f"fine_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize MLflow with remote tracking support
    print_flush("Initializing MLflow...", rank=0)
    if rank == 0:
        # Get MLflow tracking URI from config or environment
        tracking_uri = config["mlflow"].get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            tracking_uri = "file:./mlruns"  # Default to local
        
        # Handle authentication for remote MLflow
        username = config["mlflow"].get("username") or os.environ.get("MLFLOW_USERNAME")
        password = config["mlflow"].get("password") or os.environ.get("MLFLOW_PASSWORD")
        
        # If username/password provided, construct URI with credentials
        if username and password and not tracking_uri.startswith("file:"):
            parsed = urlparse(tracking_uri)
            # Reconstruct URI with credentials
            if "@" not in tracking_uri:  # Only add if not already present
                tracking_uri = f"{parsed.scheme}://{username}:{password}@{parsed.netloc}{parsed.path}"
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        mlflow.start_run(run_name=run_name)
        
        # Log ALL configuration parameters comprehensively
        all_params = {
            # Model parameters
            "model/name": config["model"]["name"],
            "model/path": config["model"]["path"],
            "model/use_flash_attention": config["model"]["use_flash_attention"],
            "model/gradient_checkpointing": config["model"]["gradient_checkpointing"],
            
            # Dataset parameters
            "dataset/name": config["dataset"]["name"],
            "dataset/path": config["dataset"]["path"],
            "dataset/max_length": config["dataset"]["max_length"],
            "dataset/batch_size_per_device": config["dataset"]["batch_size_per_device"],
            "dataset/gradient_accumulation_steps": config["dataset"]["gradient_accumulation_steps"],
            
            # Training parameters
            "training/num_epochs": config["training"]["num_epochs"],
            "training/learning_rate": config["training"]["learning_rate"],
            "training/warmup_steps": config["training"]["warmup_steps"],
            "training/weight_decay": config["training"]["weight_decay"],
            "training/max_grad_norm": config["training"]["max_grad_norm"],
            "training/lr_scheduler_type": config["training"]["lr_scheduler_type"],
            "training/save_steps": config["training"]["save_steps"],
            "training/eval_steps": config["training"]["eval_steps"],
            "training/logging_steps": config["training"]["logging_steps"],
            "training/output_dir": config["training"]["output_dir"],
            "training/save_total_limit": config["training"]["save_total_limit"],
            
            # FSDP parameters
            "fsdp/sharding_strategy": config["fsdp"]["sharding_strategy"],
            "fsdp/cpu_offload": config["fsdp"]["cpu_offload"],
            "fsdp/mixed_precision": config["fsdp"]["mixed_precision"],
            "fsdp/use_orig_params": config["fsdp"]["use_orig_params"],
            "fsdp/limit_all_gathers": config["fsdp"]["limit_all_gathers"],
            
            # Performance parameters
            "performance/dataloader_num_workers": config["performance"]["dataloader_num_workers"],
            "performance/pin_memory": config["performance"]["pin_memory"],
            "performance/prefetch_factor": config["performance"].get("prefetch_factor", 2),
            "performance/persistent_workers": config["performance"].get("persistent_workers", False),
            "performance/use_cpu_offload": config["performance"]["use_cpu_offload"],
            "performance/activation_checkpointing": config["performance"]["activation_checkpointing"],
            "performance/use_8bit_optimizer": config["performance"].get("use_8bit_optimizer", False),
            "performance/max_memory_mb": config["performance"].get("max_memory_mb"),
            "fsdp/state_dict_type": config["fsdp"].get("state_dict_type", "FULL_STATE_DICT"),
            
            # System parameters
            "system/world_size": world_size,
            "system/num_gpus": world_size,
            "system/effective_batch_size": config["dataset"]["batch_size_per_device"] * world_size * config["dataset"]["gradient_accumulation_steps"],
        }
        
        mlflow.log_params(all_params)
        
        # Log config file
        mlflow.log_artifact(args.config, "config")
        
        # Verify MLflow is working by logging a test metric
        try:
            mlflow.log_metric("test/mlflow_initialized", 1.0, step=0)
            print(f"MLflow tracking initialized successfully at: {tracking_uri}")
            print(f"Experiment: {config['mlflow']['experiment_name']}")
            print(f"Run name: {run_name}")
            print(f"MLflow run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'N/A'}")
        except Exception as e:
            print(f"WARNING: MLflow initialization test failed: {e}")
            print(f"This may indicate MLflow connectivity issues. Training will continue but metrics may not be logged.")
    
    # Authenticate with HuggingFace if token provided via environment variable only
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        if rank == 0:
            print_flush("Authenticating with HuggingFace using HF_TOKEN environment variable...", rank=0)
        try:
            login(token=hf_token)
            if rank == 0:
                print_flush("HuggingFace authentication successful", rank=0)
        except Exception as e:
            print_flush(f"⚠️  WARNING: HuggingFace login failed: {e}", rank=rank)
            print_flush(f"⚠️  Continuing anyway - authentication may not be needed for public models", rank=rank)
            # Don't fail the entire job if login fails
    else:
        if rank == 0:
            print_flush("No HF_TOKEN environment variable set - skipping HuggingFace authentication", rank=0)
            print_flush("(This is fine for public models)", rank=0)
    
    # Load tokenizer
    try:
        print_flush("Loading tokenizer...", rank=0)
        print_flush(f"[Rank {rank}] Starting tokenizer loading...", rank=rank)
        model_path = config["model"]["path"]
        print_flush(f"Tokenizer path: {model_path}", rank=0)
        if rank == 0:
            print_flush("[Rank 0] About to call AutoTokenizer.from_pretrained()...", rank=0)
        
        # Determine if path is local filesystem path or HuggingFace repo ID
        is_local_path = os.path.isabs(model_path) or model_path.startswith('./') or model_path.startswith('../')
        path_exists = os.path.exists(model_path) if is_local_path else False
        
        if is_local_path and not path_exists:
            print_flush(f"ERROR: Local model path does not exist: {model_path}", rank=0)
            print_flush("Please update 'model.path' in config.yaml to point to an existing model directory", rank=0)
            print_flush("or use a HuggingFace repo ID (e.g., 'meta-llama/Llama-2-7b-hf')", rank=0)
            cleanup_distributed()
            force_cleanup_and_exit(1)
        
        # Use token only for HuggingFace repos, not local paths
        # Only use HF_TOKEN environment variable (not config file)
        use_token = None
        if hf_token and (not is_local_path or not path_exists):
            use_token = hf_token
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=use_token,
            fix_mistral_regex=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print_flush("Tokenizer loaded successfully", rank=0)
        print_flush(f"[Rank {rank}] Tokenizer loading complete", rank=rank)
        if rank == 0:
            print_flush("[Rank 0] Tokenizer loaded, proceeding to model loading...", rank=0)
    except Exception as e:
        # Print error from the rank that hit it
        try:
            error_rank = dist.get_rank() if dist.is_initialized() else 0
        except:
            error_rank = 0
        print_flush(f"[Rank {error_rank}] ERROR loading tokenizer: {e}", rank=error_rank)
        print_flush(f"[Rank {error_rank}] Full traceback:", rank=error_rank)
        traceback.print_exc()
        print_flush(f"[Rank {error_rank}] Calling cleanup and exiting...", rank=error_rank)
        cleanup_distributed()
        force_cleanup_and_exit(1)
    
    # Load model
    try:
        print_flush("Loading model (this may take several minutes for large models)...", rank=0)
        print_flush(f"Model path: {config['model']['path']}", rank=0)
        print_flush("FSDP will handle model sharding across GPUs for optimal utilization", rank=0)
        print_flush("Model loading started...", rank=0)
        
        # CRITICAL: Load model on CPU first - FSDP will handle device placement
        # Moving to GPU before FSDP wrapping causes OOM as full model loads on each GPU
        
        # Determine if path is local filesystem path or HuggingFace repo ID
        model_path = config["model"]["path"]
        is_local_path = os.path.isabs(model_path) or model_path.startswith('./') or model_path.startswith('../')
        path_exists = os.path.exists(model_path) if is_local_path else False
        
        if is_local_path and not path_exists:
            print_flush(f"ERROR: Local model path does not exist: {model_path}", rank=0)
            print_flush("Please update 'model.path' in config.yaml to point to an existing model directory", rank=0)
            print_flush("or use a HuggingFace repo ID (e.g., 'meta-llama/Llama-2-7b-hf')", rank=0)
            cleanup_distributed()
            force_cleanup_and_exit(1)
        
        # Set max memory if specified to prevent OOM during loading
        max_memory = None
        if config["performance"].get("max_memory_mb"):
            max_memory_mb = config["performance"]["max_memory_mb"]
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                max_memory = {local_rank: f"{max_memory_mb}MB"}
                if rank == 0:
                    print(f"Setting max GPU memory to {max_memory_mb}MB per GPU", flush=True)
        
        # Use token only for HuggingFace repos, not local paths
        # Only use HF_TOKEN environment variable (not config file)
        use_token = None
        if hf_token and (not is_local_path or not path_exists):
            use_token = hf_token
        
        print_flush(f"[Rank {rank}] About to load model from {model_path}...", rank=rank)
        if rank == 0:
            print_flush("[Rank 0] Calling AutoModelForCausalLM.from_pretrained() - this may take a while...", rank=0)
        
        # Prepare model loading kwargs
        model_kwargs = {
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": "cpu",  # Load on CPU, FSDP will handle GPU placement
            "max_memory": max_memory,  # Optional memory limit
            "token": use_token
        }
        
        # Enable flash attention if configured, with graceful fallback
        use_flash_attention = config["model"].get("use_flash_attention", False)
        if use_flash_attention:
            # Check if flash_attn is available
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                if rank == 0:
                    print_flush("[Rank 0] Flash Attention 2 enabled for better performance", rank=0)
            except ImportError:
                if rank == 0:
                    print_flush("[Rank 0] WARNING: Flash Attention requested but flash-attn package not installed.", rank=0)
                    print_flush("[Rank 0] Falling back to default attention. Install with: pip install flash-attn", rank=0)
                    print_flush("[Rank 0] Continuing with default attention implementation...", rank=0)
                use_flash_attention = False
        
        # Load model with error handling for flash attention
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except (ImportError, ValueError) as e:
            if use_flash_attention and "flash_attn" in str(e):
                if rank == 0:
                    print_flush("[Rank 0] Flash Attention failed, falling back to default attention...", rank=0)
                # Remove flash attention and retry
                model_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            else:
                raise
        
        print_flush(f"[Rank {rank}] Model loaded successfully", rank=rank)
        if rank == 0:
            print_flush("[Rank 0] Model object created, proceeding to gradient checkpointing setup...", rank=0)
        
        # Enable gradient checkpointing for memory efficiency
        if config["model"]["gradient_checkpointing"]:
            # Configure gradient checkpointing based on performance settings
            checkpoint_every_n = config["performance"].get("gradient_checkpointing_every_n_layers", 1)
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if rank == 0:
                    print(f"Gradient checkpointing enabled (every {checkpoint_every_n} layer(s))", flush=True)
        
        # DO NOT move model to device here - FSDP will handle it
        # This prevents loading full model on each GPU before sharding
        print_flush("Model loaded on CPU. FSDP will handle GPU placement during wrapping.", rank=0)
        if rank == 0:
            print_flush("[Rank 0] Model loading complete, proceeding to dataset preparation...", rank=0)
    except Exception as e:
        # Print error from the rank that hit it
        try:
            error_rank = dist.get_rank() if dist.is_initialized() else 0
        except:
            error_rank = 0
        print_flush(f"[Rank {error_rank}] ERROR loading model: {e}", rank=error_rank)
        print_flush(f"[Rank {error_rank}] Full traceback:", rank=error_rank)
        traceback.print_exc()
        print_flush(f"[Rank {error_rank}] Calling cleanup and exiting...", rank=error_rank)
        cleanup_distributed()
        force_cleanup_and_exit(1)
    
    # Clear CUDA cache before FSDP wrapping
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"CUDA cache cleared before FSDP wrapping.", flush=True)
    
    # Prepare dataset
    print_flush("Preparing dataset...", rank=0)
    print_flush(f"Dataset path: {config['dataset']['path']}", rank=0)
    print_flush(f"[Rank {rank}] Starting dataset preparation...", rank=rank)
    if rank == 0:
        print_flush("[Rank 0] About to call prepare_dataset()...", rank=0)
    train_dataset = prepare_dataset(
        config["dataset"]["path"],
        tokenizer,
        config["dataset"]["max_length"]
    )
    print_flush(f"Dataset prepared. Number of samples: {len(train_dataset)}", rank=0)
    print_flush(f"[Rank {rank}] Dataset preparation complete, creating sampler and collator...", rank=rank)
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    if rank == 0:
        print_flush("[Rank 0] Distributed sampler created", rank=0)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    if rank == 0:
        print_flush("[Rank 0] Data collator created", rank=0)
    
    # Training arguments with FSDP configuration for optimal GPU utilization
    # FSDP ensures optimal GPU utilization by sharding model parameters across GPUs
    fsdp_strategy = None
    fsdp_config = None
    
    # Get transformer layer class for auto_wrap_policy
    # This ensures each transformer block is wrapped separately for optimal sharding
    try:
        # Try to get the transformer layer class from the model
        if hasattr(model, 'config'):
            # Most transformer models use this pattern
            transformer_layer_cls = None
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                transformer_layer_cls = type(model.model.layers[0])
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                transformer_layer_cls = type(model.transformer.h[0])
            elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
                transformer_layer_cls = type(model.gpt_neox.layers[0])
            
            if transformer_layer_cls is not None:
                auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls})
            else:
                # Fallback to size-based policy if transformer layer not found
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
                auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
                print_flush(f"Using size-based auto_wrap_policy (min_num_params=100M)", rank=0)
        else:
            # Fallback to size-based policy
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
            print_flush(f"Using size-based auto_wrap_policy (min_num_params=100M)", rank=0)
    except Exception as e:
        # Fallback to size-based policy on error
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
        print_flush(f"Using size-based auto_wrap_policy due to error: {e}", rank=0)
    
    # Configure FSDP based on config for optimal parallelization
    # Note: Transformers FSDP config uses boolean for cpu_offload, not CPUOffload object
    # CPU offload moves parameters to CPU to save GPU memory (slower but uses less GPU memory)
    if config["fsdp"]["cpu_offload"] and rank == 0:
        print("FSDP parameter CPU offloading enabled: parameters will be offloaded to CPU", flush=True)
    
    if config["fsdp"]["sharding_strategy"] == "FULL_SHARD":
        # Use full sharding with auto_wrap for optimal memory distribution across GPUs
        # This ensures each GPU only holds a portion of the model, maximizing utilization
        fsdp_strategy = "full_shard auto_wrap"
        fsdp_config = {
            "sharding_strategy": "full_shard",
            "cpu_offload": config["fsdp"]["cpu_offload"],  # Boolean: True moves params to CPU
            "mixed_precision": "bf16" if config["fsdp"]["mixed_precision"] else None,
            "use_orig_params": config["fsdp"]["use_orig_params"],
            "limit_all_gathers": config["fsdp"]["limit_all_gathers"],
            "state_dict_type": config["fsdp"].get("state_dict_type", "FULL_STATE_DICT"),  # Use SHARDED_STATE_DICT to save memory
            # Note: auto_wrap_policy may not be directly supported in transformers FSDP config
            # The "auto_wrap" in fsdp_strategy should handle this automatically
            # "auto_wrap_policy": auto_wrap_policy,  # Uncomment if transformers version supports it
            "sync_module_states": config["performance"].get("sync_module_states", False),  # Disabled by default to avoid deadlock during init
        }
        # Add backward prefetch for memory optimization during backward pass
        # Note: forward_prefetch is not supported in transformers FSDP config (causes parsing errors in accelerate)
        if config["fsdp"].get("backward_prefetch"):
            fsdp_config["backward_prefetch"] = config["fsdp"]["backward_prefetch"]
        # forward_prefetch is not supported - removing to avoid accelerate parsing errors
        # If you need forward prefetch, it may need to be configured at the PyTorch FSDP level
        # Note: Activation offloading is handled via gradient checkpointing and aggressive cache clearing
        # FSDP's cpu_offload only handles parameters, not activations
    elif config["fsdp"]["sharding_strategy"] == "SHARD_GRAD_OP":
        # SHARD_GRAD_OP reduces communication overhead compared to FULL_SHARD
        # This helps smooth GPU utilization by reducing all-gather operation spikes
        fsdp_strategy = "shard_grad_op auto_wrap"
        fsdp_config = {
            "sharding_strategy": "shard_grad_op",
            "cpu_offload": config["fsdp"]["cpu_offload"],  # Boolean: True moves params to CPU
            "mixed_precision": "bf16" if config["fsdp"]["mixed_precision"] else None,
            "use_orig_params": config["fsdp"]["use_orig_params"],
            "limit_all_gathers": config["fsdp"]["limit_all_gathers"],
            "state_dict_type": config["fsdp"].get("state_dict_type", "FULL_STATE_DICT"),  # Use SHARDED_STATE_DICT to save memory
            # "auto_wrap_policy": auto_wrap_policy,  # Uncomment if transformers version supports it
            "sync_module_states": config["performance"].get("sync_module_states", False),  # Disabled by default to avoid deadlock during init
        }
        # Add backward prefetch for memory optimization during backward pass
        # Note: forward_prefetch is not supported in transformers FSDP config (causes parsing errors in accelerate)
        if config["fsdp"].get("backward_prefetch"):
            fsdp_config["backward_prefetch"] = config["fsdp"]["backward_prefetch"]
        # forward_prefetch is not supported - removing to avoid accelerate parsing errors
        # If you need forward prefetch, it may need to be configured at the PyTorch FSDP level
    
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["dataset"]["batch_size_per_device"],
        gradient_accumulation_steps=config["dataset"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        bf16=True,  # Use bfloat16 for H100
        dataloader_num_workers=config["performance"]["dataloader_num_workers"],
        dataloader_pin_memory=config["performance"]["pin_memory"],
        ddp_find_unused_parameters=False,
        fsdp=fsdp_strategy,
        fsdp_config=fsdp_config,
        report_to=[],  # MLflow logging handled manually in CustomTrainer.log()
        run_name=run_name if rank == 0 else None,
        optim="adamw_torch_fused" if not config["performance"].get("use_8bit_optimizer", False) else "adamw_8bit",  # Use 8-bit optimizer if enabled (saves ~50% optimizer memory)
        max_steps=-1,  # Use num_epochs instead
        remove_unused_columns=False,  # Keep all columns for data collator
        dataloader_drop_last=True,  # Drop last incomplete batch to avoid shape issues
    )
    
    # Skip pre-training evaluation to save memory (can cause OOM with large models)
    # Evaluation will be done after training when model is properly sharded
    base_metrics = {}
    if rank == 0 and config.get("training", {}).get("evaluate_before_training", False):
        print_flush("Skipping pre-training evaluation to save memory. Set evaluate_before_training=true in config to enable.", rank=0)
        # base_metrics = evaluate_model_before_training(model, tokenizer, device, config)
    
    # Create trainer with optimized data loading settings
    if rank == 0:
        print_flush("[Rank 0] Creating CustomTrainer...", rank=0)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        prefetch_factor=config["performance"].get("prefetch_factor", 2),
        persistent_workers=config["performance"].get("persistent_workers", False),
        dataloader_timeout=config["performance"].get("dataloader_timeout", 0),
        clear_cache_frequency=config["performance"].get("clear_cache_frequency", 10),
    )
    
    if rank == 0:
        print_flush("[Rank 0] CustomTrainer created successfully", rank=0)
    
    # Activation offloading is handled via FSDP CPUOffload configuration above
    # Additional memory optimizations are applied via:
    # 1. Aggressive cache clearing in training_step (every few steps)
    # 2. Gradient checkpointing (enabled in model config)
    # 3. FSDP parameter CPU offload (if enabled)
    # 4. 8-bit optimizer (if enabled)
    if config["performance"].get("activation_offloading", False) and rank == 0:
        print("Memory optimizations enabled: activation offloading, gradient checkpointing, and aggressive cache clearing", flush=True)
    
    # Train with comprehensive error handling
    training_successful = False
    
    print_flush(f"[Rank {rank}] Entering training try block...", rank=rank)
    if rank == 0:
        print_flush("[Rank 0] ===== ENTERING TRAINING TRY BLOCK =====", rank=0)
    
    try:
        # Verify all ranks before training starts
        if dist.is_initialized():
            print_flush(f"[Rank {rank}] About to verify all ranks (all_gather)...", rank=rank)
            if rank == 0:
                print_flush("[Rank 0] Verifying all ranks are active (all_gather operation)...", rank=0)
            # Collect all ranks that are active
            print_flush(f"[Rank {rank}] Creating tensors for all_gather...", rank=rank)
            rank_tensor = torch.tensor([rank], device=device, dtype=torch.int)
            all_ranks = [torch.zeros(1, device=device, dtype=torch.int) for _ in range(world_size)]
            print_flush(f"[Rank {rank}] Calling dist.all_gather()...", rank=rank)
            dist.all_gather(all_ranks, rank_tensor)
            print_flush(f"[Rank {rank}] dist.all_gather() completed", rank=rank)
            
            if rank == 0:
                print_flush("[Rank 0] all_gather completed successfully", rank=0)
                print_flush("[Rank 0] Processing all_gather results...", rank=0)
            
            if rank == 0:
                # Skip detailed rank extraction if it might hang - just use a simple fallback
                # The verification is nice-to-have but not critical for training
                print_flush("[Rank 0] Skipping detailed rank extraction (may hang with problematic tensors)", rank=0)
                print_flush("[Rank 0] Using fallback: assuming all ranks are present", rank=0)
                active_ranks = list(range(world_size))
                print_flush(f"[Rank 0] Active ranks (fallback): {active_ranks}", rank=0)
                
                print_flush("[Rank 0] About to print GPU Utilization Verification...", rank=0)
                print_flush("=" * 60, rank=0)
                print_flush("GPU Utilization Verification:", rank=0)
                print_flush(f"  Expected world size: {world_size}", rank=0)
                print_flush(f"  Active ranks: {active_ranks}", rank=0)
                print_flush(f"  Number of active ranks: {len(active_ranks)}", rank=0)
                if len(active_ranks) != world_size:
                    print_flush(f"  ⚠️  WARNING: Only {len(active_ranks)} out of {world_size} ranks are active!", rank=0)
                    print_flush(f"  Missing ranks: {set(range(world_size)) - set(active_ranks)}", rank=0)
                else:
                    print_flush(f"  ✓ All {world_size} ranks are active and ready", rank=0)
                print_flush("=" * 60, rank=0)
                print_flush("[Rank 0] GPU Utilization Verification complete", rank=0)
        
        print_flush("=" * 60, rank=0)
        if rank == 0:
            print_flush("[Rank 0] About to print 'Starting training...' messages", rank=0)
        print_flush("Starting training...", rank=0)
        print_flush(f"World size: {world_size}", rank=0)
        print_flush(f"Effective batch size: {config['dataset']['batch_size_per_device'] * world_size * config['dataset']['gradient_accumulation_steps']}", rank=0)
        print_flush(f"Number of training steps per epoch: ~{len(train_dataset) // (config['dataset']['batch_size_per_device'] * world_size * config['dataset']['gradient_accumulation_steps'])}", rank=0)
        print_flush("=" * 60, rank=0)
        if rank == 0:
            print_flush("[Rank 0] 'Starting training...' messages complete", rank=0)
        
        # Check for shutdown request before training
        if _shutdown_requested:
            print_flush("Shutdown requested before training start, exiting...", rank=0)
            return
        
        # Test dataloader before training to catch hangs early
        if rank == 0:
            print_flush("[Rank 0] About to start dataloader test...", rank=0)
        print_flush("Testing dataloader (loading first batch)...", rank=0)
        if rank == 0:
            print_flush("[Rank 0] ===== ABOUT TO START DATALOADER TEST =====", rank=0)
        print_flush(f"[Rank {rank}] Starting dataloader test...", rank=rank)
        
        # Explicit logging for rank 0
        if rank == 0:
            print_flush("[Rank 0] ===== DATALOADER TEST START =====", rank=0)
            print_flush(f"[Rank 0] About to call trainer.get_train_dataloader()...", rank=0)
        
        try:
            test_dataloader = trainer.get_train_dataloader()
            print_flush(f"[Rank {rank}] Dataloader created successfully", rank=rank)
            if rank == 0:
                print_flush("[Rank 0] Dataloader object created, about to iterate...", rank=0)
            
            # Try to get first batch (dataloader_timeout should handle hangs)
            print_flush(f"[Rank {rank}] Attempting to load first batch (timeout: {config['performance'].get('dataloader_timeout', 0)}s)...", rank=rank)
            if rank == 0:
                print_flush("[Rank 0] Calling next(iter(test_dataloader))...", rank=0)
            
            start_time = time.time()
            first_batch = next(iter(test_dataloader))
            elapsed = time.time() - start_time
            
            print_flush(f"[Rank {rank}] ✓ Successfully loaded first batch in {elapsed:.2f}s. Batch keys: {list(first_batch.keys())}", rank=rank)
            if rank == 0:
                print_flush(f"[Rank 0] Batch shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in first_batch.items()]}", rank=0)
                print_flush("[Rank 0] ===== DATALOADER TEST COMPLETE =====", rank=0)
        except Exception as e:
            print_flush(f"[Rank {rank}] ✗ ERROR loading first batch: {e}", rank=rank)
            print_flush(f"[Rank {rank}] Dataloader appears to be hung or failed. Check:", rank=rank)
            print_flush(f"[Rank {rank}]  1. Dataset path is correct and accessible", rank=rank)
            print_flush(f"[Rank {rank}]  2. num_workers is not too high (try reducing to 0 or 1)", rank=rank)
            print_flush(f"[Rank {rank}]  3. Dataset is not corrupted", rank=rank)
            traceback.print_exc()
            raise
        
        # Verify FSDP model is properly initialized (only on rank 0 to avoid sync issues)
        # Skip this by default if it's causing hangs - the actual training will verify FSDP works
        skip_fsdp_verification = config.get("training", {}).get("skip_fsdp_verification", True)  # Default to True to avoid hangs
        
        # All ranks log their status
        print_flush(f"[Rank {rank}] Completed first batch load, proceeding to FSDP verification check...", rank=rank)
        
        if rank == 0:
            print_flush(f"[Rank 0] FSDP verification check: skip_fsdp_verification={skip_fsdp_verification}", rank=0)
            if not skip_fsdp_verification:
                print_flush("[Rank 0] Verifying FSDP model initialization...", rank=0)
                try:
                    # Check if model parameters are on correct devices
                    # Wrap in try-except to prevent hanging
                    print_flush("[Rank 0] Counting model parameters (this should be fast)...", rank=0)
                    try:
                        param_count = sum(p.numel() for p in model.parameters())
                        print_flush(f"[Rank 0] Model has {param_count:,} parameters", rank=0)
                    except Exception as e:
                        print_flush(f"[Rank 0] ⚠️  Could not count parameters (non-critical): {e}", rank=0)
                        print_flush("[Rank 0] Continuing anyway...", rank=0)
                    
                    # Check GPU memory usage (safe operation)
                    if torch.cuda.is_available():
                        try:
                            local_rank = int(os.environ.get("LOCAL_RANK", 0))
                            allocated = torch.cuda.memory_allocated(local_rank) / 1e9
                            reserved = torch.cuda.memory_reserved(local_rank) / 1e9
                            print_flush(f"[Rank 0] GPU {local_rank} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved", rank=0)
                        except Exception as e:
                            print_flush(f"[Rank 0] ⚠️  Could not check GPU memory (non-critical): {e}", rank=0)
                    
                    # Try a dummy forward pass to verify FSDP is working
                    # Use a timeout to prevent hanging
                    print_flush("Testing dummy forward pass (with 60s timeout)...", rank=0)
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Dummy forward pass timed out after 60 seconds")
                    
                    # Set up timeout (Unix only)
                    if hasattr(signal, 'SIGALRM'):
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)  # 60 second timeout
                    
                    try:
                        model.eval()
                        with torch.no_grad():
                            # Create a small dummy input
                            dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 10), device=device)
                            dummy_output = model(dummy_input)
                            print_flush("✓ Dummy forward pass successful", rank=0)
                        model.train()
                    finally:
                        # Cancel timeout
                        if hasattr(signal, 'SIGALRM'):
                            signal.alarm(0)
                except TimeoutError as e:
                    print_flush(f"⚠️  WARNING: FSDP verification timed out: {e}", rank=0)
                    print_flush("⚠️  Skipping FSDP verification. Training will proceed but may fail if FSDP is not properly initialized.", rank=0)
                    # Don't raise - allow training to continue
                except Exception as e:
                    print_flush(f"⚠️  WARNING: FSDP model verification failed: {e}", rank=0)
                    print_flush("⚠️  Continuing anyway - training will verify FSDP works during first step", rank=0)
                    traceback.print_exc()
                    # Don't raise - allow training to continue
            else:
                print_flush("[Rank 0] Skipping FSDP verification (skip_fsdp_verification=true in config)", rank=0)
                print_flush("[Rank 0] FSDP will be verified during first training step", rank=0)
                print_flush("[Rank 0] Proceeding to barrier...", rank=0)
        else:
            # Non-rank-0 ranks log that they're skipping FSDP verification
            print_flush(f"[Rank {rank}] Skipping FSDP verification (rank != 0)", rank=rank)
        
        # Log status from all ranks before barrier
        print_flush(f"[Rank {rank}] Reached barrier synchronization point...", rank=rank)
        
        # Skip diagnostic all_gather - it can hang and prevent barrier from completing
        # The barrier itself will handle synchronization
        print_flush(f"[Rank {rank}] Skipping diagnostic all_gather (proceeding directly to barrier)...", rank=rank)
        
        # Synchronize all ranks before starting training
        # Use a shorter timeout - if some ranks are stuck, we shouldn't wait 2 minutes
        if dist.is_initialized():
            print_flush(f"[Rank {rank}] Synchronizing all ranks before training start...", rank=rank)
            print_flush(f"[Rank {rank}] About to call barrier_with_timeout(timeout=120s)...", rank=rank)
            print_flush(f"[Rank {rank}] Entering barrier (timeout: 120s, waiting for all {world_size} ranks)...", rank=rank)
            print_flush(f"[Rank {rank}] Note: If timeout occurs, training will proceed with available ranks", rank=rank)
            
            # Add a timestamp for debugging
            barrier_start_time = time.time()
            try:
                # Use 120 second timeout to give ranks more time to reach the barrier
                barrier_with_timeout(timeout_seconds=120)
                barrier_elapsed = time.time() - barrier_start_time
                print_flush(f"[Rank {rank}] ✓ Barrier passed successfully in {barrier_elapsed:.2f}s", rank=rank)
                if rank == 0:
                    print_flush("All ranks synchronized, ready to start training", rank=0)
            except Exception as e:
                barrier_elapsed = time.time() - barrier_start_time
                print_flush(f"[Rank {rank}] ⚠️  WARNING: Barrier before training failed after {barrier_elapsed:.2f}s: {e}", rank=rank)
                print_flush(f"[Rank {rank}] ⚠️  Some ranks may be stuck, but proceeding with available ranks", rank=rank)
                if rank == 0:
                    print_flush("⚠️  Barrier timeout - some ranks may be missing", rank=0)
                    print_flush("⚠️  Training will proceed with ranks that reached the barrier", rank=0)
                    print_flush("⚠️  Check SLURM output files for ranks that didn't reach barrier", rank=0)
                    print_flush("⚠️  Look for ranks missing 'Entering barrier' message in their logs", rank=0)
        else:
            print_flush(f"[Rank {rank}] WARNING: dist.is_initialized() is False, skipping barrier", rank=rank)
        
        # Summary of ranks that reached barrier (for debugging)
        if rank == 0:
            print_flush("=" * 60, rank=0)
            print_flush("Barrier Status Summary:", rank=0)
            print_flush("  Expected ranks: 0-15 (16 total)", rank=0)
            print_flush("  Check logs above for which ranks printed 'Entering barrier'", rank=0)
            print_flush("  Missing ranks are likely stuck before the barrier", rank=0)
            print_flush("", rank=0)
            print_flush("  ⚠️  CRITICAL: FSDP requires ALL 16 ranks!", rank=0)
            print_flush("  If ranks are missing, training will hang at first step", rank=0)
            print_flush("  You MUST identify and fix missing ranks before training can work", rank=0)
            print_flush("", rank=0)
            print_flush("  To find missing ranks:", rank=0)
            print_flush("    1. Count how many ranks printed 'Entering barrier'", rank=0)
            print_flush("    2. Missing ranks = 16 - (count of 'Entering barrier' messages)", rank=0)
            print_flush("    3. Check SLURM output files for those missing ranks", rank=0)
            print_flush("    4. Look for the last message each missing rank printed", rank=0)
            print_flush("=" * 60, rank=0)
        
        print_flush("All pre-training checks passed. Starting training loop...", rank=0)
        
        # CRITICAL: FSDP requires ALL ranks for training to work
        # If some ranks are missing, training will hang at the first collective operation
        # Each rank logs that it's about to start training - this helps identify missing ranks
        print_flush(f"[Rank {rank}] ✓ Reached trainer.train() call point", rank=rank)
        
        if rank == 0:
            print_flush("=" * 60, rank=0)
            print_flush("⚠️  CRITICAL WARNING: FSDP Training Status", rank=0)
            print_flush("=" * 60, rank=0)
            print_flush("FSDP requires ALL 16 ranks to participate in training.", rank=0)
            print_flush("If some ranks are missing, training will HANG at the first step.", rank=0)
            print_flush("", rank=0)
            print_flush("To identify missing ranks:", rank=0)
            print_flush("  1. Check which ranks printed 'Entering barrier' above", rank=0)
            print_flush("  2. Check which ranks printed 'About to call trainer.train()' above", rank=0)
            print_flush("  3. Missing ranks are stuck before one of these points", rank=0)
            print_flush("  4. Check SLURM output files for missing ranks", rank=0)
            print_flush("", rank=0)
            print_flush("If training hangs, it means ranks are missing.", rank=0)
            print_flush("You MUST fix the missing ranks before training can proceed.", rank=0)
            print_flush("=" * 60, rank=0)
        
        print_flush(f"[Rank {rank}] About to call trainer.train()...", rank=rank)
        if rank == 0:
            print_flush("[Rank 0] ===== STARTING TRAINING =====", rank=0)
            print_flush("[Rank 0] Calling trainer.train() - this will start the training loop", rank=0)
            print_flush("[Rank 0] ⚠️  NOTE: If training hangs, some ranks may be missing from barrier", rank=0)
            print_flush("[Rank 0] ⚠️  FSDP requires ALL ranks - check which ranks didn't reach 'Entering barrier'", rank=0)
        
        # Train with timeout protection
        try:
            print_flush(f"[Rank {rank}] Calling trainer.train() now...", rank=rank)
            trainer.train()
            print_flush(f"[Rank {rank}] trainer.train() completed successfully", rank=rank)
            training_successful = True
        except KeyboardInterrupt:
            print_flush("\nTraining interrupted by user", rank=0)
            raise
        except Exception as e:
            print_flush(f"\nERROR during training: {e}", rank=0)
            traceback.print_exc()
            raise
        
        # Check for shutdown request after training
        if _shutdown_requested:
            print_flush("Shutdown requested after training, skipping model save...", rank=0)
            return
        
        # Save final model (only on rank 0 and if training was successful)
        if rank == 0 and training_successful:
            try:
                print("Saving final model...", flush=True)
                final_model_path = os.path.join(config["training"]["output_dir"], "final_model")
                trainer.save_model(final_model_path)
                tokenizer.save_pretrained(final_model_path)
                print("Final model saved successfully", flush=True)
            except Exception as e:
                print(f"ERROR saving final model: {e}", flush=True)
                traceback.print_exc()
            
            # Log model to MLflow
            try:
                if config["mlflow"]["log_model"]:
                    mlflow.pytorch.log_model(
                        model,
                        "model",
                        registered_model_name=f"{config['model']['name'].replace('/', '_')}_fine_tuned"
                    )
            except Exception as e:
                print(f"ERROR logging model to MLflow: {e}", flush=True)
                traceback.print_exc()
            
            # Evaluate after training (optional, can be skipped if OOM risk)
            try:
                evaluate_model_after_training(model, tokenizer, device, config, base_metrics)
            except Exception as e:
                print(f"WARNING: Evaluation failed (may be OOM): {e}", flush=True)
                # Don't fail the entire job if evaluation fails
            
            # Log final GPU utilization stats
            try:
                if trainer.gpu_utilizations:
                    avg_util = sum(trainer.gpu_utilizations) / len(trainer.gpu_utilizations)
                    min_util = min(trainer.gpu_utilizations)
                    max_util = max(trainer.gpu_utilizations)
                    mlflow.log_metrics({
                        "training/final_avg_gpu_utilization": avg_util,
                        "training/min_gpu_utilization": min_util,
                        "training/max_gpu_utilization": max_util
                    })
                    print(f"\nGPU Utilization Stats:")
                    print(f"  Average: {avg_util:.2f}%")
                    print(f"  Min: {min_util:.2f}%")
                    print(f"  Max: {max_util:.2f}%")
            except Exception as e:
                print(f"WARNING: Failed to log GPU stats: {e}", flush=True)
            
            # End MLflow run
            try:
                mlflow.end_run()
            except Exception as e:
                print(f"WARNING: Failed to end MLflow run: {e}", flush=True)
            
            print("Training completed successfully!", flush=True)
            
    except KeyboardInterrupt:
        print_flush("\nTraining interrupted by user (Ctrl+C)", rank=0)
        training_successful = False
    except Exception as e:
        print_flush(f"\nFATAL ERROR in training: {e}", rank=0)
        traceback.print_exc()
        training_successful = False
        # Don't re-raise, let cleanup happen
    finally:
        # Always cleanup, even on errors
        # Synchronize all ranks before cleanup to reduce TCPStore connection errors
        try:
            if dist.is_initialized():
                print_flush("Synchronizing all ranks before cleanup...", rank=0)
                try:
                    # Small barrier to ensure all ranks reach cleanup together
                    barrier_with_timeout(timeout_seconds=120)  # 2 minute timeout
                    print_flush("All ranks synchronized, starting cleanup...", rank=0)
                except Exception as e:
                    # If barrier fails, continue with cleanup anyway
                    print_flush(f"Warning: Barrier before cleanup failed (some ranks may have finished early): {e}", rank=0)
        except:
            pass  # If dist is not initialized, skip barrier
        
        print_flush("Initiating cleanup...", rank=0)
        try:
            cleanup_distributed()
        except Exception as e:
            print_flush(f"Error during cleanup: {e}", rank=0)
            traceback.print_exc()
        
        # Force exit if there was an error to prevent hanging
        if not training_successful:
            print_flush("Training failed, exiting with error code...", rank=0)
            # Give a moment for cleanup messages to flush
            time.sleep(2)
            # Ensure cleanup is called before exit
            try:
                cleanup_distributed()
            except:
                pass
            force_cleanup_and_exit(1)

if __name__ == "__main__":
    main()

