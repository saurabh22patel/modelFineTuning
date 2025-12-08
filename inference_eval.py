#!/usr/bin/env python3
"""
Inference and evaluation script for comparing base and fine-tuned models.
Captures comprehensive performance metrics and logs them to MLflow.
"""

import argparse
import os
import json
import time
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_from_disk
import mlflow
import mlflow.pytorch
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict, Tuple, Optional


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_mlflow(config: dict, run_name: str = None):
    """Setup MLflow tracking."""
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
    
    if run_name is None:
        run_name = f"inference_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    mlflow.start_run(run_name=run_name)
    return run_name


def load_model_and_tokenizer(model_path: str, device: torch.device, hf_token: str = None, 
                             use_flash_attention: bool = False):
    """Load model and tokenizer from path."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token if not os.path.exists(model_path) else None,
        fix_mistral_regex=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "token": hf_token if not os.path.exists(model_path) else None
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def get_test_prompts(dataset_path: Optional[str] = None, num_prompts: int = 20) -> List[str]:
    """
    Get test prompts for evaluation.
    Can load from dataset or use predefined prompts.
    """
    prompts = []
    
    # Try to load prompts from dataset if provided
    if dataset_path and os.path.exists(dataset_path):
        try:
            print(f"Loading test prompts from dataset: {dataset_path}")
            dataset = load_from_disk(dataset_path)
            
            # Try to extract prompts from the dataset
            # For OpenHermes, we might need to decode some examples
            if len(dataset) > 0:
                # Sample some examples
                indices = np.random.choice(len(dataset), min(num_prompts, len(dataset)), replace=False)
                for idx in indices:
                    example = dataset[int(idx)]
                    # Try to extract a prompt from the example
                    # This is dataset-specific, so we'll use a fallback
                    if 'input_ids' in example:
                        # We can't easily decode just the prompt part, so skip
                        pass
        except Exception as e:
            print(f"Warning: Could not load prompts from dataset: {e}")
    
    # Predefined test prompts covering various domains
    predefined_prompts = [
        # General knowledge
        "Explain the concept of machine learning in simple terms.",
        "What are the main differences between supervised and unsupervised learning?",
        "Describe the process of photosynthesis.",
        
        # Code generation
        "Write a Python function to calculate the factorial of a number.",
        "How do you implement a binary search algorithm?",
        "Create a function that reverses a linked list.",
        
        # Reasoning
        "If a train travels 60 miles per hour and needs to cover 120 miles, how long will it take?",
        "What is the square root of 144?",
        "If you have 10 apples and give away 3, how many do you have left?",
        
        # Creative writing
        "Write a short story about a robot learning to paint.",
        "Describe a futuristic city where AI and humans coexist.",
        "Tell me about a day in the life of a space explorer.",
        
        # Instruction following
        "List 5 benefits of regular exercise.",
        "What are the steps to make a good cup of coffee?",
        "Explain how to set up a home network.",
        
        # Technical explanations
        "What is the difference between HTTP and HTTPS?",
        "Explain how neural networks work.",
        "Describe the architecture of a transformer model.",
        
        # Problem solving
        "How would you debug a program that runs slowly?",
        "What are some strategies for managing large datasets?",
        "Explain the concept of overfitting in machine learning.",
    ]
    
    # Use predefined prompts if we don't have enough from dataset
    if len(prompts) < num_prompts:
        prompts.extend(predefined_prompts[:num_prompts - len(prompts)])
    
    return prompts[:num_prompts]


def format_prompt_for_model(prompt: str, tokenizer, use_chat_template: bool = True) -> str:
    """Format prompt using chat template if available."""
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return formatted
        except Exception as e:
            print(f"Warning: Could not apply chat template: {e}")
    
    return prompt


def generate_text(model, tokenizer, prompt: str, device: torch.device, 
                  max_new_tokens: int = 256, temperature: float = 0.7, 
                  top_p: float = 0.9, use_chat_template: bool = True) -> Dict:
    """
    Generate text from a prompt and return metrics.
    
    Returns:
        Dictionary with generated text, metrics, and timing information
    """
    # Format prompt
    formatted_prompt = format_prompt_for_model(prompt, tokenizer, use_chat_template)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove prompt)
    prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        # For chat templates, the response is after the prompt
        # Simple heuristic: response is what comes after the prompt
        if generated_text.startswith(prompt_text):
            response_text = generated_text[len(prompt_text):].strip()
        else:
            response_text = generated_text
    else:
        # For non-chat models, response is everything after the prompt
        if generated_text.startswith(prompt_text):
            response_text = generated_text[len(prompt_text):].strip()
        else:
            response_text = generated_text
    
    # Calculate metrics
    total_tokens = outputs[0].shape[0]
    new_tokens = total_tokens - input_length
    tokens_per_second = new_tokens / generation_time if generation_time > 0 else 0
    
    return {
        "prompt": prompt,
        "formatted_prompt": formatted_prompt,
        "generated_text": generated_text,
        "response_text": response_text,
        "input_length": input_length,
        "total_tokens": total_tokens,
        "new_tokens": new_tokens,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
        "response_length": len(response_text),
    }


def calculate_perplexity(model, tokenizer, texts: List[str], device: torch.device, 
                        max_length: int = 512) -> float:
    """
    Calculate perplexity on a set of texts.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             max_length=max_length, padding="max_length").to(device)
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Count non-padding tokens
            num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def evaluate_model(model, tokenizer, device: torch.device, test_prompts: List[str],
                  model_type: str = "base", max_new_tokens: int = 256,
                  calculate_ppl: bool = False, eval_dataset_path: Optional[str] = None) -> Dict:
    """
    Evaluate a model on test prompts and return comprehensive metrics.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        device: Device to run inference on
        test_prompts: List of test prompts
        model_type: Type of model ("base" or "fine_tuned")
        max_new_tokens: Maximum tokens to generate
        calculate_ppl: Whether to calculate perplexity
        eval_dataset_path: Optional path to evaluation dataset for perplexity
    
    Returns:
        Dictionary with all metrics and generations
    """
    print(f"\nEvaluating {model_type} model...")
    
    all_generations = []
    all_metrics = []
    
    # Generate responses for all prompts
    for i, prompt in enumerate(test_prompts):
        print(f"  Processing prompt {i+1}/{len(test_prompts)}...", end='\r')
        
        result = generate_text(
            model, tokenizer, prompt, device,
            max_new_tokens=max_new_tokens,
            use_chat_template=True
        )
        
        all_generations.append({
            "prompt": prompt,
            "response": result["response_text"],
            "full_generation": result["generated_text"]
        })
        
        all_metrics.append({
            f"{model_type}/prompt_{i+1}_response_length": result["response_length"],
            f"{model_type}/prompt_{i+1}_tokens": result["new_tokens"],
            f"{model_type}/prompt_{i+1}_generation_time": result["generation_time"],
            f"{model_type}/prompt_{i+1}_tokens_per_second": result["tokens_per_second"],
        })
    
    print()  # New line after progress
    
    # Calculate aggregate metrics
    response_lengths = [m[f"{model_type}/prompt_{i+1}_response_length"] 
                       for i, m in enumerate(all_metrics)]
    token_counts = [m[f"{model_type}/prompt_{i+1}_tokens"] 
                   for i, m in enumerate(all_metrics)]
    generation_times = [m[f"{model_type}/prompt_{i+1}_generation_time"] 
                       for i, m in enumerate(all_metrics)]
    tokens_per_second = [m[f"{model_type}/prompt_{i+1}_tokens_per_second"] 
                        for i, m in enumerate(all_metrics)]
    
    aggregate_metrics = {
        f"{model_type}/avg_response_length": np.mean(response_lengths),
        f"{model_type}/median_response_length": np.median(response_lengths),
        f"{model_type}/min_response_length": np.min(response_lengths),
        f"{model_type}/max_response_length": np.max(response_lengths),
        f"{model_type}/avg_tokens": np.mean(token_counts),
        f"{model_type}/avg_generation_time": np.mean(generation_times),
        f"{model_type}/total_generation_time": np.sum(generation_times),
        f"{model_type}/avg_tokens_per_second": np.mean(tokens_per_second),
        f"{model_type}/min_tokens_per_second": np.min(tokens_per_second),
        f"{model_type}/max_tokens_per_second": np.max(tokens_per_second),
    }
    
    # Calculate perplexity if requested
    perplexity = None
    if calculate_ppl:
        print(f"  Calculating perplexity for {model_type} model...")
        if eval_dataset_path and os.path.exists(eval_dataset_path):
            try:
                eval_dataset = load_from_disk(eval_dataset_path)
                # Sample some examples for perplexity calculation
                sample_size = min(100, len(eval_dataset))
                indices = np.random.choice(len(eval_dataset), sample_size, replace=False)
                texts = []
                for idx in indices:
                    example = eval_dataset[int(idx)]
                    if 'input_ids' in example:
                        # Decode the input_ids to get text
                        text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
                        texts.append(text)
                
                if texts:
                    perplexity = calculate_perplexity(model, tokenizer, texts, device)
                    aggregate_metrics[f"{model_type}/perplexity"] = perplexity
            except Exception as e:
                print(f"  Warning: Could not calculate perplexity: {e}")
    
    return {
        "generations": all_generations,
        "metrics": all_metrics,
        "aggregate_metrics": aggregate_metrics,
        "perplexity": perplexity
    }


def compare_models(base_results: Dict, fine_tuned_results: Dict) -> Dict:
    """Compare base and fine-tuned model results."""
    comparison_metrics = {}
    
    base_metrics = base_results["aggregate_metrics"]
    ft_metrics = fine_tuned_results["aggregate_metrics"]
    
    # Compare response lengths
    if "base/avg_response_length" in base_metrics and "fine_tuned/avg_response_length" in ft_metrics:
        base_avg_len = base_metrics["base/avg_response_length"]
        ft_avg_len = ft_metrics["fine_tuned/avg_response_length"]
        comparison_metrics["comparison/avg_response_length_diff"] = ft_avg_len - base_avg_len
        comparison_metrics["comparison/avg_response_length_pct_change"] = (
            (ft_avg_len - base_avg_len) / base_avg_len * 100 if base_avg_len > 0 else 0
        )
    
    # Compare generation speed
    if "base/avg_tokens_per_second" in base_metrics and "fine_tuned/avg_tokens_per_second" in ft_metrics:
        base_speed = base_metrics["base/avg_tokens_per_second"]
        ft_speed = ft_metrics["fine_tuned/avg_tokens_per_second"]
        comparison_metrics["comparison/avg_speed_diff"] = ft_speed - base_speed
        comparison_metrics["comparison/avg_speed_pct_change"] = (
            (ft_speed - base_speed) / base_speed * 100 if base_speed > 0 else 0
        )
    
    # Compare perplexity if available
    if base_results.get("perplexity") and fine_tuned_results.get("perplexity"):
        base_ppl = base_results["perplexity"]
        ft_ppl = fine_tuned_results["perplexity"]
        comparison_metrics["comparison/perplexity_diff"] = ft_ppl - base_ppl
        comparison_metrics["comparison/perplexity_pct_change"] = (
            (ft_ppl - base_ppl) / base_ppl * 100 if base_ppl > 0 else 0
        )
        comparison_metrics["comparison/perplexity_improvement"] = base_ppl - ft_ppl  # Positive is better
    
    return comparison_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base and fine-tuned models with comprehensive metrics"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to base model (overrides config)"
    )
    parser.add_argument(
        "--fine_tuned_model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=20,
        help="Number of test prompts to use"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per prompt"
    )
    parser.add_argument(
        "--calculate_perplexity",
        action="store_true",
        help="Calculate perplexity on evaluation dataset"
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default=None,
        help="Path to evaluation dataset for perplexity calculation"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Authenticate with HuggingFace if needed
    hf_token = config["model"].get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("Authenticating with HuggingFace...")
        login(token=hf_token)
    
    # Setup MLflow
    run_name = setup_mlflow(config, args.run_name)
    print(f"MLflow run: {run_name}")
    
    # Get model paths
    base_model_path = args.base_model_path or config["model"]["path"]
    fine_tuned_model_path = args.fine_tuned_model_path or os.path.join(
        config["training"]["output_dir"], "final_model"
    )
    
    # Get test prompts
    dataset_path = config["dataset"].get("path")
    test_prompts = get_test_prompts(dataset_path, args.num_prompts)
    print(f"Using {len(test_prompts)} test prompts")
    
    # Log test prompts to MLflow
    mlflow.log_text("\n".join([f"{i+1}. {p}" for i, p in enumerate(test_prompts)]), 
                   "evaluation/test_prompts.txt")
    
    # Evaluate base model
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL")
    print("="*60)
    
    base_model, base_tokenizer = load_model_and_tokenizer(
        base_model_path,
        device,
        hf_token,
        config["model"].get("use_flash_attention", False)
    )
    
    base_results = evaluate_model(
        base_model,
        base_tokenizer,
        device,
        test_prompts,
        model_type="base",
        max_new_tokens=args.max_new_tokens,
        calculate_ppl=args.calculate_perplexity,
        eval_dataset_path=args.eval_dataset_path or dataset_path
    )
    
    # Log base model results
    mlflow.log_metrics(base_results["aggregate_metrics"])
    mlflow.log_text(
        "\n\n".join([
            f"Prompt {i+1}: {g['prompt']}\n\nResponse: {g['response']}\n"
            for i, g in enumerate(base_results["generations"])
        ]),
        "evaluation/base_model_generations.txt"
    )
    
    # Clean up base model
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate fine-tuned model
    print("\n" + "="*60)
    print("EVALUATING FINE-TUNED MODEL")
    print("="*60)
    
    if not os.path.exists(fine_tuned_model_path):
        print(f"Warning: Fine-tuned model path does not exist: {fine_tuned_model_path}")
        print("Skipping fine-tuned model evaluation.")
        fine_tuned_results = None
    else:
        fine_tuned_model, fine_tuned_tokenizer = load_model_and_tokenizer(
            fine_tuned_model_path,
            device,
            hf_token,
            config["model"].get("use_flash_attention", False)
        )
        
        fine_tuned_results = evaluate_model(
            fine_tuned_model,
            fine_tuned_tokenizer,
            device,
            test_prompts,
            model_type="fine_tuned",
            max_new_tokens=args.max_new_tokens,
            calculate_ppl=args.calculate_perplexity,
            eval_dataset_path=args.eval_dataset_path or dataset_path
        )
        
        # Log fine-tuned model results
        mlflow.log_metrics(fine_tuned_results["aggregate_metrics"])
        mlflow.log_text(
            "\n\n".join([
                f"Prompt {i+1}: {g['prompt']}\n\nResponse: {g['response']}\n"
                for i, g in enumerate(fine_tuned_results["generations"])
            ]),
            "evaluation/fine_tuned_model_generations.txt"
        )
        
        # Compare models
        print("\n" + "="*60)
        print("COMPARING MODELS")
        print("="*60)
        
        comparison_metrics = compare_models(base_results, fine_tuned_results)
        mlflow.log_metrics(comparison_metrics)
        
        # Print comparison summary
        print("\nComparison Summary:")
        for key, value in comparison_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Clean up
        del fine_tuned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Log configuration
    mlflow.log_params({
        "evaluation/num_prompts": len(test_prompts),
        "evaluation/max_new_tokens": args.max_new_tokens,
        "evaluation/base_model_path": base_model_path,
        "evaluation/fine_tuned_model_path": fine_tuned_model_path,
        "evaluation/calculate_perplexity": args.calculate_perplexity,
    })
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Base Model Metrics:")
    for key, value in base_results["aggregate_metrics"].items():
        print(f"  {key}: {value:.4f}")
    
    if fine_tuned_results:
        print(f"\nFine-tuned Model Metrics:")
        for key, value in fine_tuned_results["aggregate_metrics"].items():
            print(f"  {key}: {value:.4f}")
    
    print(f"\nResults logged to MLflow run: {run_name}")
    mlflow.end_run()
    print("Evaluation complete!")


if __name__ == "__main__":
    main()

