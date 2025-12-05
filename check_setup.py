#!/usr/bin/env python3
"""
Script to verify the setup is correct before starting training.
"""

import os
import sys
import yaml
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    if os.path.exists(dirpath):
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {dirpath}")
        return False

def check_config(config_path):
    """Check configuration file."""
    print("\n=== Checking Configuration ===")
    if not check_file_exists(config_path, "Config file"):
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['model', 'dataset', 'training', 'fsdp', 'mlflow']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing section in config: {section}")
                return False
            print(f"✓ Config section: {section}")
        
        # Check model path
        model_path = config['model'].get('path', './models')
        if not os.path.exists(model_path):
            print(f"⚠ Model path does not exist: {model_path}")
            print("  Run: python download_model.py --model_name <model> --output_dir {model_path}")
        else:
            print(f"✓ Model path exists: {model_path}")
        
        # Check dataset path
        dataset_path = config['dataset'].get('path', './datasets/dataset')
        if not os.path.exists(dataset_path):
            print(f"⚠ Dataset path does not exist: {dataset_path}")
            print("  Run: python download_dataset.py --dataset_name <dataset> --output_dir ./datasets")
        else:
            print(f"✓ Dataset path exists: {dataset_path}")
        
        return True
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n=== Checking Dependencies ===")
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'accelerate',
        'mlflow',
        'yaml'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_cuda():
    """Check CUDA availability."""
    print("\n=== Checking CUDA ===")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_slurm():
    """Check SLURM environment."""
    print("\n=== Checking SLURM ===")
    slurm_vars = ['SLURM_JOB_ID', 'SLURM_NODELIST', 'SLURM_NTASKS']
    in_slurm = any(os.environ.get(var) for var in slurm_vars)
    
    if in_slurm:
        print("✓ Running in SLURM environment")
        for var in slurm_vars:
            if var in os.environ:
                print(f"  {var}: {os.environ[var]}")
    else:
        print("⚠ Not in SLURM environment (this is OK if running locally)")
    
    return True

def check_directories():
    """Check if required directories exist."""
    print("\n=== Checking Directories ===")
    directories = [
        ('logs', 'Logs directory'),
        ('checkpoints', 'Checkpoints directory'),
        ('mlruns', 'MLflow runs directory'),
    ]
    
    all_exist = True
    for dirpath, description in directories:
        if not check_directory_exists(dirpath, description):
            print(f"  Creating {dirpath}...")
            os.makedirs(dirpath, exist_ok=True)
            all_exist = False
    
    return True

def main():
    print("=" * 50)
    print("Fine-Tuning Setup Verification")
    print("=" * 50)
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    results = []
    results.append(check_dependencies())
    results.append(check_cuda())
    results.append(check_config(config_path))
    results.append(check_directories())
    results.append(check_slurm())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All checks passed! Ready to start training.")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

