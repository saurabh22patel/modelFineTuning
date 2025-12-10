#!/usr/bin/env python3
"""
Distributed fine-tuning script with FSDP for high GPU utilization.
Supports multi-node multi-GPU training with MLflow integration.
"""

import os
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
from accelerate.utils import set_seed
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
import mlflow
import mlflow.pytorch
from datetime import datetime, timedelta
import time
import signal
import atexit
import traceback
from huggingface_hub import login
from urllib.parse import urlparse

# Helper functions for backward-compatible barrier and destroy_process_group calls
def barrier_with_timeout(timeout_seconds=300):
    """Call barrier with timeout if supported, otherwise use threading-based timeout."""
    try:
        dist.barrier(timeout=timedelta(seconds=timeout_seconds))
    except TypeError:
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
            raise RuntimeError(f"Barrier timed out after {timeout_seconds} seconds. Some ranks may be stuck.")
        elif not barrier_result["success"]:
            if barrier_result["error"]:
                raise barrier_result["error"]
            else:
                raise RuntimeError("Barrier failed for unknown reason")

def destroy_process_group_with_timeout(timeout_seconds=10):
    """Call destroy_process_group with timeout if supported, otherwise without timeout."""
    try:
        dist.destroy_process_group(timeout=timedelta(seconds=timeout_seconds))
    except TypeError:
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

def check_nvlink_topology(rank, local_rank):
    """Check NVLink topology and report connectivity."""
    if not torch.cuda.is_available() or rank != 0:
        return
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', 'topo', '-m'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            num_gpus = torch.cuda.device_count()
            p2p_enabled = []
            for i in range(num_gpus):
                for j in range(i + 1, num_gpus):
                    try:
                        if torch.cuda.can_device_access_peer(i, j):
                            p2p_enabled.append((i, j))
                    except:
                        pass
            
            print_flush("=" * 60, rank=0)
            print_flush("NVLink Topology Check:", rank=0)
            if p2p_enabled:
                print_flush(f"  ✓ P2P enabled between {len(p2p_enabled)} GPU pairs", rank=0)
                if len(p2p_enabled) == num_gpus * (num_gpus - 1) // 2:
                    print_flush(f"  ✓ Full mesh connectivity", rank=0)
                else:
                    print_flush(f"  ⚠️  P2P not enabled - check NCCL_P2P_DISABLE", rank=0)
            print_flush("=" * 60, rank=0)
    except Exception as e:
        print_flush(f"Note: Could not check NVLink topology: {e}", rank=0)

def setup_distributed():
    """Initialize distributed training."""
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        if "MASTER_ADDR" not in os.environ:
            raise ValueError(
                "MASTER_ADDR not set in environment! "
                "Check that slurm_train.sh is properly setting MASTER_ADDR."
            )
        
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Handle CUDA_VISIBLE_DEVICES correctly
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if cuda_visible_devices:
        try:
            visible_physical_gpus = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
            if rank == 0:
                print(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}, visible GPUs: {visible_physical_gpus}", flush=True)
        except ValueError:
            visible_physical_gpus = []
        
        if local_rank >= num_visible_gpus:
            actual_gpu_index = local_rank % num_visible_gpus if num_visible_gpus > 0 else 0
        else:
            actual_gpu_index = local_rank
    else:
        actual_gpu_index = local_rank
    
    if num_visible_gpus == 0:
        raise RuntimeError(f"[Rank {rank}] No GPUs available!")
    
    if actual_gpu_index >= num_visible_gpus:
        raise RuntimeError(f"[Rank {rank}] GPU index {actual_gpu_index} >= num_visible_gpus {num_visible_gpus}")
    
    # Initialize process group
    timeout = int(os.environ.get("TORCH_DISTRIBUTED_INIT_TIMEOUT", 1800))
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout)
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[Rank {rank}] WARNING: Failed to initialize (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
                time.sleep(retry_delay)
            else:
                print(f"[Rank {rank}] ERROR: Failed to initialize after {max_retries} attempts: {e}", flush=True)
                raise
    
    # Synchronize all ranks
    try:
        barrier_with_timeout(timeout_seconds=300)
    except Exception as e:
        print(f"[Rank {rank}] WARNING: Barrier synchronization failed: {e}", flush=True)
    
    torch.cuda.set_device(actual_gpu_index)
    device = torch.device(f"cuda:{actual_gpu_index}")
    os.environ["LOCAL_RANK"] = str(actual_gpu_index)
    
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
    try:
        cleanup_distributed()
    except Exception as e:
        print(f"[Rank {rank}] Error during signal-triggered cleanup: {e}", flush=True)
    
    import threading
    def force_exit():
        time.sleep(2)
        print(f"[Rank {rank} PID {os.getpid()}] Force exiting after cleanup...", flush=True)
        os._exit(1)
    threading.Thread(target=force_exit, daemon=True).start()

def cleanup_distributed():
    """Clean up distributed training with error handling."""
    global _cleanup_done
    if _cleanup_done:
        return
    
    rank = 0
    is_initialized = False
    
    try:
        is_initialized = dist.is_initialized()
        if is_initialized:
            rank = dist.get_rank()
    except:
        pass
    
    try:
        if is_initialized:
            print(f"[Rank {rank}] Cleaning up distributed process group...", flush=True)
            try:
                barrier_with_timeout(timeout_seconds=5)
            except Exception as e:
                print(f"[Rank {rank}] Cleanup barrier timed out (expected): {e}", flush=True)
            
            try:
                destroy_process_group_with_timeout(timeout_seconds=10)
                print(f"[Rank {rank}] Process group destroyed successfully", flush=True)
            except Exception as e:
                error_str = str(e)
                if any(x in error_str for x in ["TCPStore", "Connection", "should dump", "Broken pipe"]):
                    print(f"[Rank {rank}] Note: TCPStore connection closed during cleanup (harmless)", flush=True)
                else:
                    print(f"[Rank {rank}] Warning: Error during cleanup: {e}", flush=True)
    except Exception as e:
        print(f"[Rank {rank}] Note: Process group may already be destroyed: {e}", flush=True)
    finally:
        _cleanup_done = True
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
        try:
            if dist.is_initialized():
                try:
                    destroy_process_group_with_timeout(timeout_seconds=5)
                except:
                    pass
        except:
            pass
    os._exit(exit_code)

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    
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
    
    if "input_ids" in dataset.column_names and "labels" in dataset.column_names:
        print("Dataset is already pre-tokenized. Using pre-tokenized data.", flush=True)
        return dataset
    
    print("Dataset is not pre-tokenized. Tokenizing on the fly...", flush=True)
    text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
    
    def tokenize_function(examples):
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

def evaluate_model_after_training(model, tokenizer, device, config):
    """Evaluate model performance after fine-tuning."""
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
                fine_tuned_metrics[f"fine_tuned/prompt_{i+1}_length"] = len(generated_text)
                fine_tuned_metrics[f"fine_tuned/prompt_{i+1}_tokens"] = len(outputs[0])
        
        mlflow.log_text("\n".join(all_generations), "evaluation/fine_tuned_model_generations.txt")
        mlflow.log_metrics(fine_tuned_metrics)
        if fine_tuned_metrics:
            mlflow.log_metric("evaluation/fine_tuned_model_avg_length", 
                            sum(fine_tuned_metrics.values()) / len(fine_tuned_metrics))
        
        print(f"Fine-tuned model evaluation complete. Logged {len(test_prompts)} generations to MLflow.")
        model.train()
        torch.cuda.empty_cache()

class CustomTrainer(Trainer):
    """Custom trainer with GPU utilization monitoring and optimized data loading."""
    
    def __init__(self, *args, prefetch_factor=None, persistent_workers=None, dataloader_timeout=None, clear_cache_frequency=10, **kwargs):
        model = kwargs.get('model')
        self._model_already_fsdp = isinstance(model, FSDP) if model is not None else False
        
        super().__init__(*args, **kwargs)
        
        self.gpu_utilizations = []
        self.last_log_time = time.time()
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.dataloader_timeout = dataloader_timeout
        self.clear_cache_frequency = clear_cache_frequency
        self._step_counter = 0
    
    def _wrap_model(self, model, training=True):
        """Override to prevent re-wrapping if model is already FSDP-wrapped."""
        if isinstance(model, FSDP):
            if dist.is_initialized() and dist.get_rank() == 0:
                print("[Rank 0] Model is already FSDP-wrapped, skipping Accelerate wrapping", flush=True)
            return model
        return super()._wrap_model(model, training=training)
    
    def _inner_training_loop(self, *args, **kwargs):
        """Override to skip model preparation if model is already FSDP-wrapped."""
        if self._model_already_fsdp:
            if hasattr(self, 'accelerator') and self.accelerator is not None:
                original_prepare_model = self.accelerator.prepare_model
                
                def skip_fsdp_prepare_model(model, device_placement=None):
                    if isinstance(model, FSDP):
                        if dist.is_initialized() and dist.get_rank() == 0:
                            print("[Rank 0] Skipping Accelerate model preparation - model is already FSDP-wrapped", flush=True)
                        return model
                    return original_prepare_model(model, device_placement=device_placement)
                
                self.accelerator.prepare_model = skip_fsdp_prepare_model
                
                try:
                    return super()._inner_training_loop(*args, **kwargs)
                finally:
                    self.accelerator.prepare_model = original_prepare_model
            else:
                return super()._inner_training_loop(*args, **kwargs)
        else:
            return super()._inner_training_loop(*args, **kwargs)
    
    def get_train_dataloader(self):
        """Override to add persistent_workers and prefetch_factor for better GPU utilization."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_sampler = self._get_train_sampler()
        
        num_workers = self.args.dataloader_num_workers
        pin_memory = self.args.dataloader_pin_memory
        prefetch_factor = self.prefetch_factor if self.prefetch_factor is not None else 2
        persistent_workers = self.persistent_workers if self.persistent_workers is not None else False
        timeout = self.dataloader_timeout if self.dataloader_timeout is not None else 0
        
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
        global _shutdown_requested
        if _shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested during training")
        
        if time.time() - self.last_log_time > 5:
            gpu_util = get_gpu_utilization()
            self.gpu_utilizations.append(gpu_util)
            self.last_log_time = time.time()
            
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                allocated = torch.cuda.memory_allocated(local_rank) / 1e9
                reserved = torch.cuda.memory_reserved(local_rank) / 1e9
                total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
                memory_pct = (reserved / total) * 100
                
                if memory_pct > 90 and dist.get_rank() == 0:
                    print(f"WARNING: GPU memory usage is {memory_pct:.1f}% ({reserved:.2f}GB / {total:.2f}GB)", flush=True)
            
            if dist.get_rank() == 0 and len(self.gpu_utilizations) % 10 == 0:
                try:
                    avg_util = sum(self.gpu_utilizations[-10:]) / 10
                    mlflow.log_metric("training/avg_gpu_utilization", avg_util, step=self.state.global_step)
                    print(f"Step {self.state.global_step}: GPU Utilization: {avg_util:.2f}%")
                except Exception:
                    pass
        
        self._step_counter += 1
        should_clear_cache = (self._step_counter % self.clear_cache_frequency == 0)
        
        if should_clear_cache and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        try:
            if num_items_in_batch is not None:
                result = super().training_step(model, inputs, num_items_in_batch)
            else:
                result = super().training_step(model, inputs)
            
            if should_clear_cache and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            return result
        except torch.cuda.OutOfMemoryError as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            rank = dist.get_rank() if dist.is_initialized() else 0
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(local_rank) / 1e9
                reserved = torch.cuda.memory_reserved(local_rank) / 1e9
                total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
                print(f"[Rank {rank}] OOM Error! GPU {local_rank} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total", flush=True)
                print(f"[Rank {rank}] Try reducing max_length or batch_size in config.yaml", flush=True)
            raise
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Override compute_loss to add memory cleanup hooks during backward pass."""
        parent_kwargs = {"return_outputs": return_outputs}
        if num_items_in_batch is not None:
            parent_kwargs["num_items_in_batch"] = num_items_in_batch
        parent_kwargs.update(kwargs)
        loss = super().compute_loss(model, inputs, **parent_kwargs)
        
        if hasattr(loss, 'register_hook') and (self._step_counter % self.clear_cache_frequency == 0):
            def backward_hook(grad):
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
                return grad
            loss.register_hook(backward_hook)
        
        return loss
    
    def log(self, logs, start_time=None):
        if self.gpu_utilizations:
            logs["gpu_utilization"] = self.gpu_utilizations[-1]
        
        try:
            is_rank_0 = False
            if dist.is_initialized():
                is_rank_0 = (dist.get_rank() == 0)
            else:
                is_rank_0 = True
            
            if is_rank_0:
                if mlflow.active_run() is not None:
                    step = self.state.global_step if hasattr(self.state, 'global_step') else None
                    
                    for key, value in logs.items():
                        if isinstance(value, (int, float)):
                            if step is not None:
                                mlflow.log_metric(key, value, step=step)
                            else:
                                mlflow.log_metric(key, value)
                        elif isinstance(value, str):
                            mlflow.log_text(value, f"logs/{key}.txt")
        except Exception as e:
            try:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Warning: Failed to log to MLflow: {e}", flush=True)
            except:
                pass
        
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Override checkpoint saving to optimize FSDP checkpoint operations."""
        if dist.is_initialized() and dist.get_rank() == 0:
            checkpoint_start_time = time.time()
            print(f"[Rank 0] Saving checkpoint at step {self.state.global_step}...", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            super()._save_checkpoint(model, trial, metrics)
        except Exception as e:
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"Warning: Checkpoint save encountered error: {e}", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if dist.is_initialized() and dist.get_rank() == 0:
            checkpoint_time = time.time() - checkpoint_start_time
            print(f"[Rank 0] Checkpoint saved successfully in {checkpoint_time:.2f}s", flush=True)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to skip evaluation during training if disabled."""
        if eval_dataset is None:
            if dist.is_initialized() and dist.get_rank() == 0:
                print("Skipping evaluation (no eval dataset provided) to maintain GPU utilization", flush=True)
            return {}
        
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Starting evaluation (this may cause brief GPU utilization drop)...", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return metrics

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
                # Check if it's a valid checkpoint (has training_state.bin or trainer_state.json)
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


def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_distributed)
    
    parser = argparse.ArgumentParser(description="Distributed fine-tuning with FSDP")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                       help="Path to checkpoint directory to resume from, or 'latest' to resume from latest checkpoint")
    args = parser.parse_args()
    
    rank = 0
    local_rank = 0
    world_size = 1
    device = None
    
    try:
        rank, local_rank, world_size, device = setup_distributed()
        print_flush(f"[Rank {rank}] Distributed training initialized successfully", rank=rank)
    except Exception as e:
        print(f"ERROR: Failed to setup distributed training: {e}", flush=True)
        traceback.print_exc()
        force_cleanup_and_exit(1)
    
    # Check NVLink topology
    check_nvlink_topology(rank, local_rank)
    
    # Verify all ranks can communicate
    if dist.is_initialized():
        test_tensor = torch.ones(1, device=device)
        try:
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            if rank == 0:
                if test_tensor.item() != world_size:
                    print_flush(f"⚠️  WARNING: all_reduce sum={test_tensor.item()}, expected={world_size}", rank=0)
                else:
                    print_flush(f"✓ All {world_size} ranks successfully initialized and can communicate", rank=0)
        except Exception as e:
            print_flush(f"[Rank {rank}] ERROR in initial all_reduce: {e}", rank=rank)
            raise
        
    # Print GPU memory status
    if torch.cuda.is_available():
        actual_gpu_index = int(os.environ.get("LOCAL_RANK", 0))
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_properties(current_device).name
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(current_device) / 1e9
        reserved = torch.cuda.memory_reserved(current_device) / 1e9
        print_flush(f"[Rank {rank}] GPU {current_device} ({gpu_name}) Memory - Total: {total_memory:.2f} GB, Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB", rank=rank)
    
    # Load configuration
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
    
    # Set CUDA memory allocation strategy
    if not os.environ.get("PYTORCH_ALLOC_CONF") and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"
    elif os.environ.get("PYTORCH_CUDA_ALLOC_CONF") and not os.environ.get("PYTORCH_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    
    # Generate run name
    run_name = f"fine_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize MLflow
    if rank == 0:
        tracking_uri = config["mlflow"].get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            tracking_uri = "file:./mlruns"
        
        username = config["mlflow"].get("username") or os.environ.get("MLFLOW_USERNAME")
        password = config["mlflow"].get("password") or os.environ.get("MLFLOW_PASSWORD")
        
        if username and password and not tracking_uri.startswith("file:"):
            parsed = urlparse(tracking_uri)
            if "@" not in tracking_uri:
                tracking_uri = f"{parsed.scheme}://{username}:{password}@{parsed.netloc}{parsed.path}"
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        mlflow.start_run(run_name=run_name)
        
        # Log configuration parameters
        all_params = {
            "model/name": config["model"]["name"],
            "model/path": config["model"]["path"],
            "model/use_flash_attention": config["model"]["use_flash_attention"],
            "model/gradient_checkpointing": config["model"]["gradient_checkpointing"],
            "dataset/name": config["dataset"]["name"],
            "dataset/path": config["dataset"]["path"],
            "dataset/max_length": config["dataset"]["max_length"],
            "dataset/batch_size_per_device": config["dataset"]["batch_size_per_device"],
            "dataset/gradient_accumulation_steps": config["dataset"]["gradient_accumulation_steps"],
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
            "fsdp/sharding_strategy": config["fsdp"]["sharding_strategy"],
            "fsdp/cpu_offload": config["fsdp"]["cpu_offload"],
            "fsdp/mixed_precision": config["fsdp"]["mixed_precision"],
            "fsdp/use_orig_params": config["fsdp"]["use_orig_params"],
            "fsdp/limit_all_gathers": config["fsdp"]["limit_all_gathers"],
            "performance/dataloader_num_workers": config["performance"]["dataloader_num_workers"],
            "performance/pin_memory": config["performance"]["pin_memory"],
            "performance/prefetch_factor": config["performance"].get("prefetch_factor", 2),
            "performance/persistent_workers": config["performance"].get("persistent_workers", False),
            "performance/use_cpu_offload": config["performance"]["use_cpu_offload"],
            "performance/activation_checkpointing": config["performance"]["activation_checkpointing"],
            "performance/use_8bit_optimizer": config["performance"].get("use_8bit_optimizer", False),
            "performance/max_memory_mb": config["performance"].get("max_memory_mb"),
            "fsdp/state_dict_type": config["fsdp"].get("state_dict_type", "FULL_STATE_DICT"),
            "system/world_size": world_size,
            "system/num_gpus": world_size,
            "system/effective_batch_size": config["dataset"]["batch_size_per_device"] * world_size * config["dataset"]["gradient_accumulation_steps"],
        }
        
        mlflow.log_params(all_params)
        mlflow.log_artifact(args.config, "config")
        
        try:
            mlflow.log_metric("test/mlflow_initialized", 1.0, step=0)
            print(f"MLflow tracking initialized successfully at: {tracking_uri}")
        except Exception as e:
            print(f"WARNING: MLflow initialization test failed: {e}")
    
    # Authenticate with HuggingFace
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        if rank == 0:
            print_flush("Authenticating with HuggingFace...", rank=0)
        try:
            login(token=hf_token)
            if rank == 0:
                print_flush("HuggingFace authentication successful", rank=0)
        except Exception as e:
            print_flush(f"⚠️  WARNING: HuggingFace login failed: {e}", rank=rank)
    
    # Helper function to check model path
    def check_model_path(model_path):
        """Check if model path exists and determine if token is needed."""
        is_local_path = os.path.isabs(model_path) or model_path.startswith('./') or model_path.startswith('../')
        path_exists = os.path.exists(model_path) if is_local_path else False
        
        if is_local_path and not path_exists:
            print_flush(f"ERROR: Local model path does not exist: {model_path}", rank=0)
            cleanup_distributed()
            force_cleanup_and_exit(1)
        
        use_token = None
        if hf_token and (not is_local_path or not path_exists):
            use_token = hf_token
        
        return use_token
    
    # Load tokenizer
    try:
        print_flush("Loading tokenizer...", rank=0)
        model_path = config["model"]["path"]
        use_token = check_model_path(model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=use_token,
            fix_mistral_regex=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print_flush("Tokenizer loaded successfully", rank=0)
    except Exception as e:
        error_rank = dist.get_rank() if dist.is_initialized() else 0
        print_flush(f"[Rank {error_rank}] ERROR loading tokenizer: {e}", rank=error_rank)
        traceback.print_exc()
        cleanup_distributed()
        force_cleanup_and_exit(1)
    
    # Load model
    try:
        print_flush("Loading model (this may take several minutes for large models)...", rank=0)
        model_path = config["model"]["path"]
        check_model_path(model_path)  # Validate path
        
        max_memory = None
        if config["performance"].get("max_memory_mb"):
            max_memory_mb = config["performance"]["max_memory_mb"]
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                max_memory = {local_rank: f"{max_memory_mb}MB"}
        
        use_token = check_model_path(model_path)  # Get token if needed
        
        model_kwargs = {
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": "cpu",
            "max_memory": max_memory,
            "token": use_token
        }
        
        use_flash_attention = config["model"].get("use_flash_attention", False)
        if use_flash_attention:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                if rank == 0:
                    print_flush("Flash Attention 2 enabled", rank=0)
            except ImportError:
                if rank == 0:
                    print_flush("WARNING: Flash Attention requested but not installed. Falling back to default.", rank=0)
                use_flash_attention = False
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except (ImportError, ValueError) as e:
            if use_flash_attention and "flash_attn" in str(e):
                if rank == 0:
                    print_flush("Flash Attention failed, falling back to default attention...", rank=0)
                model_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            else:
                raise
        
        if config["model"]["gradient_checkpointing"]:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if rank == 0:
                    print("Gradient checkpointing enabled", flush=True)
        
        print_flush("Model loaded on CPU. FSDP will handle GPU placement.", rank=0)
    except Exception as e:
        error_rank = dist.get_rank() if dist.is_initialized() else 0
        print_flush(f"[Rank {error_rank}] ERROR loading model: {e}", rank=error_rank)
        traceback.print_exc()
        cleanup_distributed()
        force_cleanup_and_exit(1)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Prepare dataset
    print_flush("Preparing dataset...", rank=0)
    train_dataset = prepare_dataset(
        config["dataset"]["path"],
        tokenizer,
        config["dataset"]["max_length"]
    )
    print_flush(f"Dataset prepared. Number of samples: {len(train_dataset)}", rank=0)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Get transformer layer class for auto_wrap_policy
    transformer_layer_cls = None
    try:
        if hasattr(model, 'config'):
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                transformer_layer_cls = type(model.model.layers[0])
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                transformer_layer_cls = type(model.transformer.h[0])
            elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
                transformer_layer_cls = type(model.gpt_neox.layers[0])
    except Exception as e:
        if rank == 0:
            print_flush(f"Error detecting transformer layer class: {e}", rank=0)
    
    if transformer_layer_cls is not None:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
        if rank == 0:
            print_flush(f"Using transformer_auto_wrap_policy with {transformer_layer_cls.__name__}", rank=0)
    else:
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e7)
        if rank == 0:
            print_flush("Using size-based auto_wrap_policy", rank=0)
    
    # Wrap model with FSDP
    if rank == 0:
        print_flush("Wrapping model with FSDP...", rank=0)
    
    fsdp_kwargs = dict(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=config["fsdp"]["use_orig_params"],
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=config["fsdp"]["limit_all_gathers"],
        sync_module_states=config["performance"].get("sync_module_states", True),
        device_id=torch.cuda.current_device(),
    )
    
    model = FSDP(model, **fsdp_kwargs)
    
    if rank == 0:
        print_flush("Model wrapped with FSDP successfully", rank=0)
    
    fsdp_strategy = None
    fsdp_config = {
        "state_dict_type": config["fsdp"].get("state_dict_type", "SHARDED_STATE_DICT"),
    }
    
    use_8bit_optimizer = config["performance"].get("use_8bit_optimizer", False)
    if use_8bit_optimizer and rank == 0:
        print("WARNING: 8-bit optimizer is incompatible with FSDP checkpoint saving.", flush=True)
    
    save_safetensors = config["training"].get("save_safetensors", True)
    save_only_model = config["training"].get("save_only_model", False)
    
    if rank == 0:
        print(f"Checkpoint optimization: save_safetensors={save_safetensors}, save_only_model={save_only_model}", flush=True)
    
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
        bf16=True,
        dataloader_num_workers=config["performance"]["dataloader_num_workers"],
        dataloader_pin_memory=config["performance"]["pin_memory"],
        ddp_find_unused_parameters=False,
        fsdp=fsdp_strategy,
        fsdp_config=fsdp_config,
        report_to=[],
        run_name=run_name if rank == 0 else None,
        optim="adamw_torch_fused" if not use_8bit_optimizer else "adamw_8bit",
        max_steps=-1,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        save_safetensors=save_safetensors,
        save_only_model=save_only_model,
    )
    
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
    
    training_successful = False
    
    try:
        # Verify all ranks before training
        if dist.is_initialized():
            test_tensor = torch.ones(1, device=device)
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            if rank == 0 and test_tensor.item() != world_size:
                print_flush(f"⚠️  WARNING: Only {int(test_tensor.item())} ranks participated in verification", rank=0)
        
        print_flush("=" * 60, rank=0)
        print_flush("Starting training...", rank=0)
        print_flush(f"World size: {world_size}", rank=0)
        print_flush(f"Effective batch size: {config['dataset']['batch_size_per_device'] * world_size * config['dataset']['gradient_accumulation_steps']}", rank=0)
        print_flush("=" * 60, rank=0)
        
        if _shutdown_requested:
            print_flush("Shutdown requested before training start, exiting...", rank=0)
            return
        
        # Test dataloader
        print_flush("Testing dataloader...", rank=0)
        try:
            test_dataloader = trainer.get_train_dataloader()
            start_time = time.time()
            first_batch = next(iter(test_dataloader))
            elapsed = time.time() - start_time
            print_flush(f"✓ Successfully loaded first batch in {elapsed:.2f}s", rank=0)
        except Exception as e:
            print_flush(f"✗ ERROR loading first batch: {e}", rank=rank)
            traceback.print_exc()
            raise
        
        # Synchronize all ranks before training
        if dist.is_initialized():
            try:
                barrier_with_timeout(timeout_seconds=120)
                if rank == 0:
                    print_flush("All ranks synchronized, ready to start training", rank=0)
            except Exception as e:
                if rank == 0:
                    print_flush(f"⚠️  WARNING: Barrier failed: {e}", rank=0)
                    print_flush("⚠️  Proceeding with available ranks", rank=0)
        
        # Determine checkpoint to resume from
        resume_from_checkpoint = None
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint.lower() == "latest":
                resume_from_checkpoint = find_latest_checkpoint(config["training"]["output_dir"])
                if resume_from_checkpoint:
                    print_flush(f"Resuming from latest checkpoint: {resume_from_checkpoint}", rank=0)
                else:
                    print_flush("No checkpoint found, starting training from scratch", rank=0)
            else:
                resume_from_checkpoint = args.resume_from_checkpoint
                if os.path.exists(resume_from_checkpoint):
                    print_flush(f"Resuming from checkpoint: {resume_from_checkpoint}", rank=0)
                else:
                    print_flush(f"WARNING: Checkpoint path does not exist: {resume_from_checkpoint}", rank=0)
                    print_flush("Starting training from scratch", rank=0)
                    resume_from_checkpoint = None
        
        print_flush("Starting training loop...", rank=0)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print_flush("Training completed successfully", rank=rank)
        training_successful = True
        
        if _shutdown_requested:
            print_flush("Shutdown requested after training, skipping model save...", rank=0)
            return
        
        # Save final model
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
            
            try:
                evaluate_model_after_training(model, tokenizer, device, config)
            except Exception as e:
                print(f"WARNING: Evaluation failed: {e}", flush=True)
            
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
                    print(f"\nGPU Utilization Stats: Avg={avg_util:.2f}%, Min={min_util:.2f}%, Max={max_util:.2f}%")
            except Exception as e:
                print(f"WARNING: Failed to log GPU stats: {e}", flush=True)
            
            try:
                mlflow.end_run()
            except Exception as e:
                print(f"WARNING: Failed to end MLflow run: {e}", flush=True)
            
            print("Training completed successfully!", flush=True)
            
    except KeyboardInterrupt:
        print_flush("\nTraining interrupted by user", rank=0)
        training_successful = False
    except Exception as e:
        print_flush(f"\nFATAL ERROR in training: {e}", rank=0)
        traceback.print_exc()
        training_successful = False
    finally:
        try:
            if dist.is_initialized():
                try:
                    barrier_with_timeout(timeout_seconds=120)
                except Exception:
                    pass
        except:
            pass
        
        print_flush("Initiating cleanup...", rank=0)
        try:
            cleanup_distributed()
        except Exception as e:
            print_flush(f"Error during cleanup: {e}", rank=0)
            traceback.print_exc()
        
        if not training_successful:
            print_flush("Training failed, exiting with error code...", rank=0)
            time.sleep(2)
            try:
                cleanup_distributed()
            except:
                pass
            force_cleanup_and_exit(1)

if __name__ == "__main__":
    main()
