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
from huggingface_hub import login

def download_dataset(dataset_name: str, output_dir: str, tokenizer_path: str = None, 
                     max_length: int = 2048, split: str = "train", model_name: str = None, hf_token: str = None):
    """
    Download and prepare a dataset for fine-tuning with pre-tokenization.
    
    Args:
        dataset_name: HuggingFace dataset identifier or path to local data
        output_dir: Directory to save the processed dataset
        tokenizer_path: Path to tokenizer for preprocessing
        max_length: Maximum sequence length
        split: Dataset split to use (default: 'train')
        model_name: HuggingFace model name to load tokenizer from (if tokenizer_path not provided)
    """
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Authenticate with HuggingFace if token provided
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("Authenticating with HuggingFace...")
        login(token=token)
    
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
        
        # Load tokenizer for pre-tokenization
        tokenizer = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        elif model_name:
            print(f"Loading tokenizer from model: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
        else:
            raise ValueError("Either tokenizer_path or model_name must be provided for pre-tokenization")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine text column
        text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
        print(f"Using text column: {text_column}")
        
        # Pre-tokenize the dataset
        print(f"Pre-tokenizing dataset with max_length={max_length}...")
        def tokenize_function(examples):
            # Tokenize the texts
            tokenized = tokenizer(
                examples[text_column],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            # Set labels for language modeling (labels are same as input_ids)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col != text_column],
            num_proc=8,  # Use multiple processes for faster tokenization
            desc="Pre-tokenizing dataset"
        )
        
        # Remove the original text column if it still exists
        if text_column in tokenized_dataset.column_names:
            tokenized_dataset = tokenized_dataset.remove_columns([text_column])
        
        print(f"Pre-tokenization complete. Tokenized dataset has {len(tokenized_dataset)} examples")
        
        # Save tokenized dataset
        dataset_path = os.path.join(output_dir, "dataset")
        print(f"Saving tokenized dataset to {dataset_path}...")
        tokenized_dataset.save_to_disk(dataset_path)
        
        # Save tokenizer for later use
        tokenizer_path_save = os.path.join(output_dir, "tokenizer")
        print(f"Saving tokenizer to {tokenizer_path_save}...")
        tokenizer.save_pretrained(tokenizer_path_save)
        
        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "num_examples": len(tokenized_dataset),
            "max_length": max_length,
            "split": split,
            "features": list(tokenized_dataset.features.keys()),
            "pre_tokenized": True,
            "tokenizer_path": tokenizer_path or model_name
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Pre-tokenized dataset successfully saved to {output_dir}")
        print(f"Metadata: {metadata}")
        
    except Exception as e:
        print(f"Error downloading and tokenizing dataset: {e}")
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
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model name to load tokenizer from (if tokenizer_path not provided)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (can also use HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    download_dataset(
        args.dataset_name,
        args.output_dir,
        args.tokenizer_path,
        args.max_length,
        args.split,
        args.model_name,
        args.hf_token
    )

