#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --output=logs/install_flash_attn_%j.out
#SBATCH --error=logs/install_flash_attn_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=main

set -e

echo "=========================================="
echo "Flash Attention Installation Job"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Configuration
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_PATH="/root/llmtune/venv/bin/activate"

# Setup
mkdir -p logs
cd "$PROJECT_DIR"

# Activate virtual environment
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH"
elif [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    echo "Activating local virtual environment"
    source "$PROJECT_DIR/venv/bin/activate"
else
    echo "ERROR: Virtual environment not found!"
    echo "Expected: $VENV_PATH or $PROJECT_DIR/venv/bin/activate"
    exit 1
fi

# Check Python and CUDA
echo ""
echo "Environment check:"
python3 --version
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'torch not installed')"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo ""

# Check if flash-attn is already installed
echo "Checking if flash-attn is already installed..."
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "✓ Flash attention is already installed!"
    python3 -c "import flash_attn; print(f'Version: {flash_attn.__version__ if hasattr(flash_attn, \"__version__\") else \"unknown\"}')" 2>/dev/null || true
    echo ""
    echo "Installation complete (already installed)."
    exit 0
fi

echo "Flash attention not found. Starting installation..."
echo ""

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Set build parameters to avoid overwhelming the system
export MAX_JOBS=8
export FLASH_ATTENTION_SKIP_CUDA_BUILD=0

# Try installing with pre-built wheels first (faster)
echo "Attempting to install flash-attn with pre-built wheels..."
echo "This may take 5-10 minutes..."
if pip install flash-attn --no-build-isolation 2>&1 | tee logs/flash_attn_install.log; then
    echo ""
    echo "✓ Flash attention installed successfully with pre-built wheels!"
else
    echo ""
    echo "Pre-built wheel installation failed. Installing from source..."
    echo "This will take 15-30 minutes..."
    echo ""
    
    # Install from source
    if pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tee -a logs/flash_attn_install.log; then
        echo ""
        echo "✓ Flash attention installed successfully from source!"
    else
        echo ""
        echo "✗ ERROR: Flash attention installation failed!"
        echo "Check logs/flash_attn_install.log for details"
        exit 1
    fi
fi

# Verify installation
echo ""
echo "Verifying installation..."
if python3 -c "import flash_attn; print('✓ Flash attention imported successfully!')" 2>/dev/null; then
    python3 -c "import flash_attn; print(f'Version: {flash_attn.__version__ if hasattr(flash_attn, \"__version__\") else \"unknown\"}')" 2>/dev/null || true
    echo ""
    echo "=========================================="
    echo "Installation completed successfully!"
    echo "End time: $(date)"
    echo "=========================================="
    exit 0
else
    echo "✗ ERROR: Flash attention import failed after installation!"
    exit 1
fi







