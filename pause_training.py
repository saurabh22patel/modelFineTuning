#!/usr/bin/env python3
"""
Helper script to pause training gracefully by sending SIGTERM signal.
This allows the training script to save the current checkpoint before stopping.
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def find_training_processes():
    """Find running training processes."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        processes = []
        for line in result.stdout.split('\n'):
            if 'train.py' in line or 'slurm_train.sh' in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        cmd = ' '.join(parts[10:])  # Command starts around column 10
                        processes.append((pid, cmd))
                    except (ValueError, IndexError):
                        continue
        
        return processes
    except Exception as e:
        print(f"Error finding processes: {e}")
        return []


def pause_training(pid=None, graceful=True, wait_time=60):
    """
    Pause training by sending a signal to the training process.
    
    Args:
        pid: Process ID to pause. If None, will try to find it automatically.
        graceful: If True, sends SIGTERM for graceful shutdown. If False, sends SIGKILL.
        wait_time: Maximum time to wait for graceful shutdown (seconds).
    """
    if pid is None:
        processes = find_training_processes()
        if not processes:
            print("No training processes found.")
            print("\nTo pause a specific process, use:")
            print("  python pause_training.py --pid <PID>")
            print("\nOr find the PID manually:")
            print("  ps aux | grep train.py")
            return False
        
        if len(processes) > 1:
            print(f"Found {len(processes)} training processes:")
            for i, (p, cmd) in enumerate(processes, 1):
                print(f"  {i}. PID {p}: {cmd[:80]}...")
            print("\nPlease specify the PID using --pid option")
            return False
        
        pid = processes[0][0]
        print(f"Found training process: PID {pid}")
    
    # Verify process exists
    try:
        os.kill(pid, 0)  # Signal 0 just checks if process exists
    except OSError:
        print(f"Process {pid} does not exist or you don't have permission to access it.")
        return False
    
    if graceful:
        print(f"Sending SIGTERM to process {pid} for graceful shutdown...")
        print("The training script will save the current checkpoint before stopping.")
        try:
            os.kill(pid, signal.SIGTERM)
            
            # Wait for graceful shutdown
            print(f"Waiting up to {wait_time} seconds for graceful shutdown...")
            for i in range(wait_time):
                try:
                    os.kill(pid, 0)  # Check if process still exists
                    time.sleep(1)
                    if (i + 1) % 10 == 0:
                        print(f"  Still waiting... ({i+1}s)")
                except OSError:
                    print(f"\n✓ Process {pid} stopped gracefully.")
                    return True
            
            print(f"\n⚠ Process {pid} did not stop within {wait_time} seconds.")
            print("You may need to force kill it with --force option.")
            return False
        except PermissionError:
            print(f"Permission denied. You may need to run as the same user or use sudo.")
            return False
    else:
        print(f"Force killing process {pid} with SIGKILL...")
        print("⚠ WARNING: This may result in loss of unsaved progress!")
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
            try:
                os.kill(pid, 0)
                print(f"✗ Process {pid} still running after SIGKILL.")
                return False
            except OSError:
                print(f"✓ Process {pid} killed.")
                return True
        except PermissionError:
            print(f"Permission denied. You may need to run as the same user or use sudo.")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Pause training gracefully by sending SIGTERM signal"
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Process ID of the training process (auto-detected if not provided)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force kill the process (SIGKILL) instead of graceful shutdown (SIGTERM)"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=60,
        help="Maximum time to wait for graceful shutdown in seconds (default: 60)"
    )
    
    args = parser.parse_args()
    
    success = pause_training(
        pid=args.pid,
        graceful=not args.force,
        wait_time=args.wait
    )
    
    if success:
        print("\n✓ Training paused successfully.")
        print("You can resume training later using:")
        print("  python resume_training.py")
        sys.exit(0)
    else:
        print("\n✗ Failed to pause training.")
        sys.exit(1)


if __name__ == "__main__":
    main()

