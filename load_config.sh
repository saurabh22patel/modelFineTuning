#!/bin/bash
# Helper script to load configuration from config.yaml for bash scripts
# This script reads the config.yaml file and exports environment variables

CONFIG_FILE="${1:-config.yaml}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file not found: $CONFIG_FILE" >&2
    return 0  # Return success to allow scripts to continue with defaults
fi

# Use Python to read YAML config (works with system Python, before venv activation)
# This extracts the venv_path from the config file
# Check if python3 is available
if command -v python3 >/dev/null 2>&1; then
    VENV_PATH_FROM_CONFIG=$(python3 <<EOF 2>/dev/null
import sys
import os

try:
    import yaml
except ImportError:
    # PyYAML not available, silently fail
    sys.exit(0)

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get venv_path from config
    venv_path = None
    if config and 'environment' in config and config['environment']:
        venv_path = config['environment'].get('venv_path')
    
    # Expand ~ to home directory if present
    if venv_path:
        venv_path = os.path.expanduser(venv_path)
        print(venv_path)
except Exception:
    # Silently fail - let scripts use defaults
    sys.exit(0)
EOF
)
else
    # Python3 not available, skip config loading
    VENV_PATH_FROM_CONFIG=""
fi

# Export the venv path if found
if [ -n "$VENV_PATH_FROM_CONFIG" ] && [ "$VENV_PATH_FROM_CONFIG" != "null" ] && [ "$VENV_PATH_FROM_CONFIG" != "None" ]; then
    export VENV_PATH="$VENV_PATH_FROM_CONFIG"
fi

