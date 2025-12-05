"""
Utility functions for training and monitoring.
"""

import torch
import os
import subprocess

def check_gpu_availability():
    """Check if GPUs are available and print information."""
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return False
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    return True

def optimize_for_h100():
    """Set optimizations specific to H100 GPUs."""
    # Enable TensorFloat-32 for faster training on H100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def get_optimal_batch_size(model_size_gb, gpu_memory_gb, sequence_length=2048):
    """
    Estimate optimal batch size based on model size and GPU memory.
    
    Args:
        model_size_gb: Model size in GB
        gpu_memory_gb: Available GPU memory in GB
        sequence_length: Sequence length
    
    Returns:
        Estimated batch size per device
    """
    # Rough estimation: leave 20% memory for overhead
    available_memory = gpu_memory_gb * 0.8
    
    # Memory per sample (rough estimate)
    # Model weights + activations + gradients
    memory_per_sample = (model_size_gb * 3) + (sequence_length * 0.001)  # Rough estimate
    
    batch_size = int(available_memory / memory_per_sample)
    return max(1, batch_size)

def verify_distributed_setup():
    """Verify that distributed training is set up correctly."""
    if not torch.distributed.is_available():
        print("PyTorch distributed is not available!")
        return False
    
    if not torch.distributed.is_initialized():
        print("Distributed training is not initialized!")
        return False
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"Distributed training initialized:")
    print(f"  Rank: {rank}")
    print(f"  World Size: {world_size}")
    print(f"  Local Rank: {local_rank}")
    
    return True

