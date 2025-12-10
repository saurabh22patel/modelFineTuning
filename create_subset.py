#!/usr/bin/env python3
"""
Create a subset of a pre-tokenized dataset based on the total number of tokens.
This script loads a full dataset and creates a subset containing approximately
the specified number of tokens.
"""

import argparse
import os
import json
from datasets import load_from_disk
import numpy as np


def count_non_padding_tokens(input_ids, pad_token_id):
    """Count the number of non-padding tokens in a sequence."""
    if pad_token_id is None:
        # If no pad token, count all tokens
        return len(input_ids)
    return int(np.sum(np.array(input_ids) != pad_token_id))


def create_subset(
    input_dataset_path: str,
    output_dataset_path: str,
    target_tokens: int,
    pad_token_id: int = None,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Create a subset of the dataset containing approximately target_tokens tokens.
    
    Args:
        input_dataset_path: Path to the full dataset directory
        output_dataset_path: Path to save the subset dataset
        target_tokens: Target number of tokens in the subset
        pad_token_id: Padding token ID (if None, will try to infer from dataset)
        shuffle: Whether to shuffle the dataset before selecting examples
        seed: Random seed for shuffling
    """
    print(f"Loading dataset from {input_dataset_path}...")
    dataset = load_from_disk(input_dataset_path)
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Check if dataset is pre-tokenized
    if "input_ids" not in dataset.column_names:
        raise ValueError(
            "Dataset must be pre-tokenized with 'input_ids' column. "
            f"Available columns: {list(dataset.column_names)}"
        )
    
    # Try to infer pad_token_id from the dataset if not provided
    if pad_token_id is None:
        # Check metadata for pad_token_id
        metadata_path = os.path.join(input_dataset_path, "..", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # Try to get pad_token_id from tokenizer if available
                tokenizer_path = metadata.get("tokenizer_path")
                if tokenizer_path:
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                        pad_token_id = tokenizer.pad_token_id
                        print(f"Inferred pad_token_id={pad_token_id} from tokenizer")
                    except Exception as e:
                        print(f"Could not load tokenizer: {e}")
        
        # If still None, try to infer from data (most common padding value)
        if pad_token_id is None:
            print("Attempting to infer pad_token_id from dataset...")
            sample_input_ids = dataset[0]["input_ids"]
            if isinstance(sample_input_ids, list):
                # Check the last few tokens - padding is usually at the end
                last_tokens = sample_input_ids[-10:]
                pad_token_id = max(set(last_tokens), key=last_tokens.count)
                print(f"Inferred pad_token_id={pad_token_id} from data")
            else:
                pad_token_id = 0  # Default fallback
                print(f"Using default pad_token_id={pad_token_id}")
    
    # Shuffle if requested
    if shuffle:
        print(f"Shuffling dataset with seed={seed}...")
        dataset = dataset.shuffle(seed=seed)
    
    # Calculate tokens per example and accumulate
    print("Counting tokens in each example...")
    selected_indices = []
    total_tokens = 0
    
    for idx in range(len(dataset)):
        input_ids = dataset[idx]["input_ids"]
        num_tokens = count_non_padding_tokens(input_ids, pad_token_id)
        
        if total_tokens + num_tokens <= target_tokens:
            selected_indices.append(idx)
            total_tokens += num_tokens
        else:
            # Check if adding this example would exceed target by too much
            # If it's close, we might want to include it anyway
            if total_tokens == 0:
                # If first example exceeds target, include it anyway
                selected_indices.append(idx)
                total_tokens += num_tokens
            break
    
    print(f"\nSelected {len(selected_indices)} examples")
    print(f"Total tokens: {total_tokens:,} (target: {target_tokens:,})")
    print(f"Difference: {total_tokens - target_tokens:,} tokens ({((total_tokens - target_tokens) / target_tokens * 100):.2f}%)")
    
    # Create subset
    print(f"\nCreating subset...")
    subset = dataset.select(selected_indices)
    
    # Save subset
    print(f"Saving subset to {output_dataset_path}...")
    os.makedirs(os.path.dirname(output_dataset_path) if os.path.dirname(output_dataset_path) else ".", exist_ok=True)
    subset.save_to_disk(output_dataset_path)
    
    # Save metadata
    metadata = {
        "source_dataset_path": input_dataset_path,
        "num_examples": len(subset),
        "total_tokens": int(total_tokens),
        "target_tokens": target_tokens,
        "pad_token_id": pad_token_id,
        "shuffled": shuffle,
        "seed": seed if shuffle else None,
        "features": list(subset.features.keys())
    }
    
    metadata_path = os.path.join(output_dataset_path, "..", "subset_metadata.json")
    if os.path.dirname(output_dataset_path):
        metadata_path = os.path.join(os.path.dirname(output_dataset_path), "subset_metadata.json")
    else:
        metadata_path = "subset_metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Subset successfully created and saved to {output_dataset_path}")
    print(f"  - Examples: {len(subset)}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Metadata saved to: {metadata_path}")
    
    return subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a subset of a pre-tokenized dataset based on token count"
    )
    parser.add_argument(
        "--input_dataset_path",
        type=str,
        required=True,
        help="Path to the full dataset directory"
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        required=True,
        help="Path to save the subset dataset"
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        required=True,
        help="Target number of tokens in the subset (e.g., 1000000 for 1M tokens)"
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=None,
        help="Padding token ID (will try to infer if not provided)"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Don't shuffle the dataset before selecting examples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    create_subset(
        args.input_dataset_path,
        args.output_dataset_path,
        args.target_tokens,
        args.pad_token_id,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )

