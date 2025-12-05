#!/usr/bin/env python3
"""
Script to download and prepare a model for fine-tuning.
Supports HuggingFace models and saves them in a format ready for distributed training.
"""

import argparse
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model(model_name: str, output_dir: str, cache_dir: str = None):
    """
    Download a model from HuggingFace and save it locally.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-hf')
        output_dir: Directory to save the model
        cache_dir: Optional cache directory for HuggingFace cache
    """
    print(f"Downloading model: {model_name}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set cache directory if provided
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Download model
        print("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Save model and tokenizer
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Model successfully downloaded and saved to {output_dir}")
        print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model for fine-tuning")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save the model (default: ./models)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace downloads (optional)"
    )
    
    args = parser.parse_args()
    download_model(args.model_name, args.output_dir, args.cache_dir)

