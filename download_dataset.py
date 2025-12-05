#!/usr/bin/env python3
"""
Script to download and prepare a dataset for fine-tuning.
Supports HuggingFace datasets and custom data formats.
"""

import argparse
import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def download_dataset(dataset_name: str, output_dir: str, tokenizer_path: str = None, 
                     max_length: int = 2048, split: str = "train"):
    """
    Download and prepare a dataset for fine-tuning.
    
    Args:
        dataset_name: HuggingFace dataset identifier or path to local data
        output_dir: Directory to save the processed dataset
        tokenizer_path: Path to tokenizer for preprocessing
        max_length: Maximum sequence length
        split: Dataset split to use (default: 'train')
    """
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load dataset
        print("Loading dataset...")
        if os.path.exists(dataset_name):
            # Local file/directory
            if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
                with open(dataset_name, 'r') as f:
                    data = [json.loads(line) if dataset_name.endswith('.jsonl') else json.load(f)]
                dataset = Dataset.from_list(data)
            else:
                dataset = load_dataset(dataset_name, split=split)
        else:
            # HuggingFace dataset
            dataset = load_dataset(dataset_name, split=split)
        
        print(f"Dataset loaded: {len(dataset)} examples")
        
        # Load tokenizer if provided for preprocessing
        tokenizer = None
        if tokenizer_path:
            print(f"Loading tokenizer from {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Save dataset
        dataset_path = os.path.join(output_dir, "dataset")
        dataset.save_to_disk(dataset_path)
        
        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "num_examples": len(dataset),
            "max_length": max_length,
            "split": split,
            "features": list(dataset.features.keys())
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset successfully saved to {output_dir}")
        print(f"Metadata: {metadata}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset for fine-tuning")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset identifier or path to local data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets",
        help="Directory to save the dataset (default: ./datasets)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer for preprocessing (optional)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: 'train')"
    )
    
    args = parser.parse_args()
    download_dataset(
        args.dataset_name,
        args.output_dir,
        args.tokenizer_path,
        args.max_length,
        args.split
    )

