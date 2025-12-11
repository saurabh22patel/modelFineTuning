#!/usr/bin/env python3
"""
Consolidate FSDP checkpoint into a single model file.
This script loads a checkpoint and saves it as a standard HuggingFace model.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
import yaml

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def consolidate_checkpoint(checkpoint_path, output_path, config_path=None, model_path=None):
    """
    Consolidate an FSDP checkpoint into a single model file.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        output_path: Path where the consolidated model will be saved
        config_path: Path to config.yaml (optional, for model config)
        model_path: Path to original model (optional, for model config)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Determine model path
    if model_path is None:
        if config_path:
            config = load_config(config_path)
            model_path = config["model"]["path"]
        else:
            # Try to infer from checkpoint
            config_json = os.path.join(checkpoint_path, "config.json")
            if os.path.exists(config_json):
                # Load model config to get model name
                from transformers import AutoConfig
                model_config = AutoConfig.from_pretrained(checkpoint_path)
                # Try common model paths
                model_path = checkpoint_path
            else:
                raise ValueError("Cannot determine model path. Provide --model_path or --config")
    
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    
    # Load checkpoint state dict
    print("Loading checkpoint state dict...")
    
    # Try to load from different possible formats
    checkpoint_state_dict = None
    
    # Try safetensors first (if available)
    try:
        from safetensors import safe_open
        safetensors_file = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(safetensors_file):
            print("Loading from safetensors format...")
            state_dict = {}
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            checkpoint_state_dict = state_dict
    except ImportError:
        pass
    except Exception as e:
        print(f"Could not load safetensors: {e}")
    
    # Fallback to pytorch_model.bin
    if checkpoint_state_dict is None:
        pytorch_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(pytorch_file):
            print("Loading from pytorch_model.bin...")
            checkpoint_state_dict = torch.load(pytorch_file, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model file found in checkpoint. Expected pytorch_model.bin or model.safetensors in {checkpoint_path}"
            )
    
    # Handle FSDP checkpoint format
    # FSDP checkpoints may have keys prefixed with "_fsdp_wrapped_module."
    state_dict = {}
    for key, value in checkpoint_state_dict.items():
        # Remove FSDP prefix if present
        new_key = key.replace("_fsdp_wrapped_module.", "")
        state_dict[new_key] = value
    
    # Load state dict into model
    print("Loading state dict into model...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:10]}...")  # Show first 10
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:10]}...")  # Show first 10
    
    # Save consolidated model
    print(f"Saving consolidated model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Determine if we should use safetensors
    use_safetensors = True
    if config_path:
        config = load_config(config_path)
        use_safetensors = config["training"].get("save_safetensors", True)
    
    model.save_pretrained(
        output_path,
        safe_serialization=use_safetensors
    )
    tokenizer.save_pretrained(output_path)
    
    print("Consolidation complete!")
    print(f"Model saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Consolidate FSDP checkpoint into final model")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint directory")
    parser.add_argument("output_path", type=str, help="Path where consolidated model will be saved")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model_path", type=str, default=None, help="Path to original model (overrides config)")
    
    args = parser.parse_args()
    
    try:
        consolidate_checkpoint(
            args.checkpoint_path,
            args.output_path,
            config_path=args.config if os.path.exists(args.config) else None,
            model_path=args.model_path
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

