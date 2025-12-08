#!/usr/bin/env python3
"""
Download and process the teknium/OpenHermes-2.5 dataset for fine-tuning.
This script handles the conversational format and converts it to instruction-following format.
"""

import argparse
import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from huggingface_hub import login


def format_conversation_for_training(conversations, tokenizer, max_length=2048):
    """
    Format OpenHermes conversations into instruction-following format.
    
    OpenHermes-2.5 uses a ShareGPT-like format with conversations containing
    messages with 'from' and 'value' fields. Also handles other common formats.
    
    Args:
        conversations: List of conversation messages (dicts or strings)
        tokenizer: Tokenizer with chat template support
        max_length: Maximum sequence length (not used but kept for compatibility)
    
    Returns:
        Formatted text string ready for tokenization
    """
    if not conversations or len(conversations) == 0:
        return None
    
    # Handle case where conversations might be a single dict instead of list
    if isinstance(conversations, dict):
        conversations = [conversations]
    
    # Check if tokenizer has a chat template (for models like Llama-3.1)
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        # Convert OpenHermes format to standard chat format
        messages = []
        for msg in conversations:
            # Handle different message formats
            if isinstance(msg, str):
                # If it's a string, treat as user message
                messages.append({"role": "user", "content": msg})
                continue
            
            if not isinstance(msg, dict):
                continue
            
            # Try different field name variations
            role = None
            content = None
            
            # Try 'from' field (OpenHermes format)
            if 'from' in msg:
                role = msg.get('from', '').lower()
                content = msg.get('value', '') or msg.get('content', '')
            # Try 'role' field (standard format)
            elif 'role' in msg:
                role = msg.get('role', '').lower()
                content = msg.get('content', '') or msg.get('value', '')
            # Try 'role' and 'message' fields
            elif 'message' in msg:
                role = msg.get('role', 'user').lower()
                content = msg.get('message', '')
            else:
                # Fallback: use first string value as content
                for key, value in msg.items():
                    if isinstance(value, str) and value.strip():
                        content = value
                        role = key.lower() if key != 'value' else 'user'
                        break
            
            if not content or not role:
                continue
            
            # Map various role names to standard roles
            if role in ['human', 'user', 'prompt']:
                messages.append({"role": "user", "content": str(content)})
            elif role in ['gpt', 'assistant', 'response', 'answer']:
                messages.append({"role": "assistant", "content": str(content)})
            elif role == 'system':
                messages.append({"role": "system", "content": str(content)})
            else:
                # Unknown role, default to user
                messages.append({"role": "user", "content": str(content)})
        
        if not messages:
            return None
        
        # Apply chat template
        try:
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return formatted_text
        except Exception as e:
            # Fallback to manual formatting if chat template fails
            pass
    
    # Fallback: Manual formatting for models without chat templates
    formatted_parts = []
    for msg in conversations:
        if isinstance(msg, str):
            formatted_parts.append(f"User: {msg}")
            continue
        
        if not isinstance(msg, dict):
            continue
        
        # Extract role and content
        role = msg.get('from', msg.get('role', 'user')).lower()
        content = msg.get('value', msg.get('content', msg.get('message', '')))
        
        if not content:
            continue
        
        # Format based on role
        if role in ['human', 'user', 'prompt']:
            formatted_parts.append(f"User: {content}")
        elif role in ['gpt', 'assistant', 'response', 'answer']:
            formatted_parts.append(f"Assistant: {content}")
        elif role == 'system':
            formatted_parts.append(f"System: {content}")
        else:
            formatted_parts.append(f"User: {content}")
    
    if not formatted_parts:
        return None
    
    return "\n\n".join(formatted_parts)


def download_dataset(
    dataset_name: str = "teknium/OpenHermes-2.5",
    output_dir: str = "./datasets",
    tokenizer_path: str = None,
    max_length: int = 2048,
    split: str = "train",
    model_name: str = None,
    hf_token: str = None
):
    """
    Download and process the OpenHermes-2.5 dataset for fine-tuning.
    
    Args:
        dataset_name: HuggingFace dataset identifier (default: teknium/OpenHermes-2.5)
        output_dir: Directory to save the processed dataset
        tokenizer_path: Path to local tokenizer (optional)
        max_length: Maximum sequence length for tokenization
        split: Dataset split to use (default: 'train')
        model_name: HuggingFace model name to load tokenizer from
        hf_token: HuggingFace token for gated models
    """
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Authenticate with HuggingFace if token provided
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("Authenticating with HuggingFace...")
        login(token=token)
    
    try:
        # Load dataset
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset(dataset_name, split=split, token=token)
        print(f"Dataset loaded: {len(dataset)} examples")
        
        # Display dataset structure
        if len(dataset) > 0:
            print(f"\nDataset features: {list(dataset.features.keys())}")
            print(f"Sample example keys: {list(dataset[0].keys())}")
            if 'conversations' in dataset[0]:
                print(f"Sample conversation structure: {dataset[0]['conversations'][:2] if len(dataset[0]['conversations']) > 0 else 'Empty'}")
        
        # Load tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=token, fix_mistral_regex=True)
        elif model_name:
            print(f"Loading tokenizer from model: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=token,
                fix_mistral_regex=True
            )
        else:
            raise ValueError("Either tokenizer_path or model_name must be provided")
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Process conversations into formatted text
        print("\nProcessing conversations into instruction-following format...")
        
        # Detect conversation field name
        conv_field = None
        for field in ['conversations', 'messages', 'conversation']:
            if field in dataset.column_names:
                conv_field = field
                break
        
        if conv_field is None:
            # Try to find any list field that might contain conversations
            if len(dataset) > 0:
                for key in dataset[0].keys():
                    if isinstance(dataset[0][key], list):
                        conv_field = key
                        break
        
        if conv_field is None:
            raise ValueError(
                f"Could not find conversations field in dataset. "
                f"Available fields: {list(dataset.column_names)}"
            )
        
        print(f"Using field '{conv_field}' for conversations")
        
        def process_conversations(examples):
            """Process conversations and format them for training."""
            formatted_texts = []
            
            for conversations in examples.get(conv_field, []):
                formatted_text = format_conversation_for_training(
                    conversations,
                    tokenizer,
                    max_length
                )
                formatted_texts.append(formatted_text)
            
            return {"text": formatted_texts}
        
        # Process the dataset
        # Determine which columns to keep (don't remove conversations field yet)
        columns_to_remove = [col for col in dataset.column_names]
        processed_dataset = dataset.map(
            process_conversations,
            batched=True,
            num_proc=8,
            desc="Formatting conversations",
            remove_columns=columns_to_remove
        )
        
        # Filter out None/empty examples
        print("Filtering out empty examples...")
        processed_dataset = processed_dataset.filter(
            lambda x: x['text'] is not None and len(x['text'].strip()) > 0,
            num_proc=8
        )
        print(f"After filtering: {len(processed_dataset)} examples")
        
        # Tokenize the dataset
        print(f"\nPre-tokenizing dataset with max_length={max_length}...")
        
        def tokenize_function(examples):
            """Tokenize the formatted texts."""
            texts = examples['text']
            
            # Ensure all are strings
            cleaned_texts = []
            for t in texts:
                if isinstance(t, str):
                    cleaned_texts.append(t)
                elif t is None:
                    cleaned_texts.append("")
                else:
                    cleaned_texts.append(str(t))
            
            # Tokenize
            tokenized = tokenizer(
                cleaned_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = processed_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            num_proc=8,
            desc="Tokenizing dataset"
        )
        
        print(f"Pre-tokenization complete. Tokenized dataset has {len(tokenized_dataset)} examples")
        
        # Save dataset
        dataset_path = os.path.join(output_dir, "dataset")
        print(f"\nSaving tokenized dataset to {dataset_path}...")
        tokenized_dataset.save_to_disk(dataset_path)
        
        # Save tokenizer
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
            "tokenizer_path": tokenizer_path or model_name,
            "format": "instruction_following",
            "source_format": "conversational"
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Dataset successfully processed and saved to {output_dir}")
        print(f"  - Examples: {len(tokenized_dataset)}")
        print(f"  - Max length: {max_length}")
        print(f"  - Features: {list(tokenized_dataset.features.keys())}")
        print(f"  - Metadata saved to: {metadata_path}")
        
    except Exception as e:
        print(f"Error downloading and processing dataset: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process the OpenHermes-2.5 dataset for fine-tuning"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="teknium/OpenHermes-2.5",
        help="HuggingFace dataset identifier (default: teknium/OpenHermes-2.5)"
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
        help="Path to local tokenizer (optional)"
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
        help="HuggingFace model name to load tokenizer from (required if tokenizer_path not provided)"
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
