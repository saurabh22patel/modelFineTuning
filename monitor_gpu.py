#!/usr/bin/env python3
"""
GPU monitoring utility to track GPU utilization during training.
Can be run in parallel with training to monitor GPU usage.
"""

import subprocess
import time
import json
import argparse
from datetime import datetime

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'utilization_gpu': float(parts[2]),
                            'utilization_memory': float(parts[3]),
                            'memory_used_gb': float(parts[4]) / 1024,
                            'memory_total_gb': float(parts[5]) / 1024,
                            'temperature': float(parts[6]),
                            'power_draw': float(parts[7])
                        })
            return gpus
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
    return []

def monitor_gpus(interval=5, output_file=None):
    """Monitor GPUs and log statistics."""
    stats_history = []
    
    print("Starting GPU monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            gpus = get_gpu_stats()
            if gpus:
                timestamp = datetime.now().isoformat()
                stats = {
                    'timestamp': timestamp,
                    'gpus': gpus,
                    'avg_utilization': sum(g['utilization_gpu'] for g in gpus) / len(gpus) if gpus else 0
                }
                stats_history.append(stats)
                
                # Print summary
                print(f"\n[{timestamp}] Average GPU Utilization: {stats['avg_utilization']:.2f}%")
                for gpu in gpus:
                    print(f"  GPU {gpu['index']}: {gpu['utilization_gpu']:.1f}% util, "
                          f"{gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB, "
                          f"{gpu['temperature']:.1f}°C, {gpu['power_draw']:.1f}W")
                
                # Save to file if specified
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(stats_history, f, indent=2)
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        if output_file and stats_history:
            with open(output_file, 'w') as f:
                json.dump(stats_history, f, indent=2)
            print(f"Statistics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU utilization")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="Output file to save statistics (optional)")
    
    args = parser.parse_args()
    monitor_gpus(args.interval, args.output)

