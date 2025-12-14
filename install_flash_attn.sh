#!/bin/bash
# Quick script to install flash-attn (for interactive use with nohup or screen)
# For SLURM clusters, use: sbatch slurm_install_flash_attn.sh

set -e

echo "Flash Attention Installation Script"
echo "===================================="
echo ""
echo "NOTE: This may take 15-30 minutes. If your SSH session might disconnect,"
echo "      use one of these methods:"
echo ""
echo "  1. Use SLURM job (recommended):"
echo "     sbatch slurm_install_flash_attn.sh"
echo ""
echo "  2. Use nohup:"
echo "     nohup ./install_flash_attn.sh > logs/flash_attn_install.log 2>&1 &"
echo ""
echo "  3. Use screen/tmux:"
echo "     screen -S flash_attn"
echo "     ./install_flash_attn.sh"
echo "     (Press Ctrl+A then D to detach)"
echo ""
echo "Press Enter to continue, or Ctrl+C to cancel..."
read

# Find virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/root/llmtune/venv/bin/activate"

if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "ERROR: Virtual environment not found!"
    exit 1
fi

# Check if already installed
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "✓ Flash attention is already installed!"
    exit 0
fi

echo "Installing flash-attention..."
echo "This may take 15-30 minutes..."

# Upgrade pip
pip install --upgrade pip

# Set build parameters
export MAX_JOBS=4

# Try pre-built wheels first
echo "Attempting pre-built wheel installation..."
if pip install flash-attn --no-build-isolation; then
    echo "✓ Installed with pre-built wheels!"
else
    echo "Installing from source (this will take longer)..."
    pip install flash-attn --no-build-isolation --no-cache-dir
fi

# Verify
if python3 -c "import flash_attn; print('✓ Installation successful!')"; then
    echo "Flash attention is ready to use!"
else
    echo "✗ Installation verification failed!"
    exit 1
fi






