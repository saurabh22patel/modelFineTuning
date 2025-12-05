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
import mlflow
import mlflow.pytorch
from datetime import datetime
import json
import psutil
import time

def setup_distributed():
    """Initialize distributed training."""
    if "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        # Get master node address
        if "SLURM_STEP_NODELIST" in os.environ:
            master_addr = os.environ["SLURM_STEP_NODELIST"].split(",")[0]
        else:
            master_addr = os.environ["SLURM_NODELIST"].split(",")[0]
        
        # Set master port
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        # Local or torchrun environment
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return rank, local_rank, world_size, device

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_dataset(dataset_path, tokenizer, max_length):
    """Prepare dataset for training."""
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Determine text column
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
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Tokenizing"
    )
    
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
    """Evaluate model performance before fine-tuning."""
    if dist.get_rank() == 0:
        print("Evaluating model before fine-tuning...")
        
        # Simple evaluation: generate text
        model.eval()
        test_prompt = "The future of artificial intelligence is"
        
        with torch.no_grad():
            inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log to MLflow
        mlflow.log_text(generated_text, "evaluation/before_training_generation.txt")
        mlflow.log_metric("evaluation/before_training_length", len(generated_text))
        
        print(f"Before training generation: {generated_text}")
        model.train()

def evaluate_model_after_training(model, tokenizer, device, config):
    """Evaluate model performance after fine-tuning."""
    if dist.get_rank() == 0:
        print("Evaluating model after fine-tuning...")
        
        model.eval()
        test_prompt = "The future of artificial intelligence is"
        
        with torch.no_grad():
            inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log to MLflow
        mlflow.log_text(generated_text, "evaluation/after_training_generation.txt")
        mlflow.log_metric("evaluation/after_training_length", len(generated_text))
        
        print(f"After training generation: {generated_text}")
        model.train()

class CustomTrainer(Trainer):
    """Custom trainer with GPU utilization monitoring."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_utilizations = []
        self.last_log_time = time.time()
    
    def training_step(self, model, inputs):
        # Monitor GPU utilization
        if time.time() - self.last_log_time > 5:  # Log every 5 seconds
            gpu_util = get_gpu_utilization()
            self.gpu_utilizations.append(gpu_util)
            self.last_log_time = time.time()
            
            if dist.get_rank() == 0 and len(self.gpu_utilizations) % 10 == 0:
                avg_util = sum(self.gpu_utilizations[-10:]) / 10
                mlflow.log_metric("training/avg_gpu_utilization", avg_util, step=self.state.global_step)
                print(f"Step {self.state.global_step}: GPU Utilization: {avg_util:.2f}%")
        
        return super().training_step(model, inputs)
    
    def log(self, logs):
        # Add GPU utilization to logs
        if self.gpu_utilizations:
            logs["gpu_utilization"] = self.gpu_utilizations[-1]
        super().log(logs)

def main():
    parser = argparse.ArgumentParser(description="Distributed fine-tuning with FSDP")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()
    
    # Load configuration
    if rank == 0:
        print("Loading configuration...")
    config = load_config(args.config)
    
    # Set random seed
    set_seed(42)
    
    # Generate run name (needed for all ranks)
    run_name = f"fine_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize MLflow
    if rank == 0:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        mlflow.start_run(run_name=run_name)
        
        # Log configuration
        mlflow.log_params({
            "model_name": config["model"]["name"],
            "dataset_name": config["dataset"]["name"],
            "num_epochs": config["training"]["num_epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "batch_size_per_device": config["dataset"]["batch_size_per_device"],
            "gradient_accumulation_steps": config["dataset"]["gradient_accumulation_steps"],
            "world_size": world_size,
            "num_gpus": world_size
        })
        
        # Log config file
        mlflow.log_artifact(args.config, "config")
    
    # Load tokenizer
    if rank == 0:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if rank == 0:
        print("Loading model...")
    
    # For FSDP, don't use device_map
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing for memory efficiency
    if config["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    # Move model to device (FSDP will handle distribution)
    model = model.to(device)
    
    # Prepare dataset
    if rank == 0:
        print("Preparing dataset...")
    train_dataset = prepare_dataset(
        config["dataset"]["path"],
        tokenizer,
        config["dataset"]["max_length"]
    )
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    fsdp_strategy = None
    fsdp_config = None
    if config["fsdp"]["sharding_strategy"] == "FULL_SHARD":
        fsdp_strategy = "full_shard auto_wrap"
        fsdp_config = {
            "sharding_strategy": "full_shard",
            "cpu_offload": config["fsdp"]["cpu_offload"],
            "mixed_precision": "bf16" if config["fsdp"]["mixed_precision"] else None,
            "use_orig_params": config["fsdp"]["use_orig_params"],
            "limit_all_gathers": config["fsdp"]["limit_all_gathers"],
        }
    
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
        report_to="mlflow" if rank == 0 else [],
        run_name=run_name if rank == 0 else None,
    )
    
    # Evaluate before training
    if rank == 0:
        evaluate_model_before_training(model, tokenizer, device, config)
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    if rank == 0:
        print("Starting training...")
        print(f"World size: {world_size}")
        print(f"Effective batch size: {config['dataset']['batch_size_per_device'] * world_size * config['dataset']['gradient_accumulation_steps']}")
    
    trainer.train()
    
    # Save final model
    if rank == 0:
        print("Saving final model...")
        final_model_path = os.path.join(config["training"]["output_dir"], "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # Log model to MLflow
        if config["mlflow"]["log_model"]:
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name=f"{config['model']['name'].replace('/', '_')}_fine_tuned"
            )
        
        # Evaluate after training
        evaluate_model_after_training(model, tokenizer, device, config)
        
        # Log final GPU utilization stats
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
        
        mlflow.end_run()
        print("Training completed!")
    
    # Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    main()

