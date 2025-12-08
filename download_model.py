#!/usr/bin/env python3

import argparse
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

def print_flush(msg):
    """Print message and flush stdout/stderr to ensure output appears immediately"""
    print(msg, flush=True)

def download_model(model_name: str, output_dir: str, cache_dir: str = None, hf_token: str = None):
    print_flush(f"Downloading model: {model_name}")
    print_flush(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print_flush("Authenticating with HuggingFace...")
        login(token=token)
    else:
        print_flush("Warning: No HuggingFace token provided. If model is gated, download will fail.")
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    try:
        print_flush("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
            fix_mistral_regex=True
        )
        print_flush("Tokenizer downloaded successfully.")
        
        print_flush("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=token
        )
        print_flush("Model downloaded successfully.")
        
        print_flush(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print_flush("Model saved successfully.")
        
        model_size = sum(p.numel() for p in model.parameters()) / 1e9
        print_flush(f"Model successfully downloaded and saved to {output_dir}")
        print_flush(f"Model size: {model_size:.2f}B parameters")
        
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False

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
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (can also use HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    success = download_model(args.model_name, args.output_dir, args.cache_dir, args.hf_token)
    sys.exit(0 if success else 1)

