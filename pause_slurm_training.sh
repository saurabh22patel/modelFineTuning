#!/bin/bash
"""
Helper script to pause SLURM training jobs gracefully.
This sends SIGTERM to the SLURM job, allowing the training script to save checkpoints.
"""

# Function to find training jobs
find_training_jobs() {
    squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" | grep -E "(llm_finetune|train)" || echo ""
}

# Function to pause a specific job
pause_job() {
    local job_id=$1
    local graceful=${2:-true}
    
    if [ -z "$job_id" ]; then
        echo "Error: Job ID required"
        return 1
    fi
    
    # Check if job exists
    if ! squeue -j $job_id &>/dev/null; then
        echo "Error: Job $job_id not found or already completed"
        return 1
    fi
    
    if [ "$graceful" = true ]; then
        echo "Sending SIGTERM to job $job_id for graceful shutdown..."
        echo "The training script will save the current checkpoint before stopping."
        scancel --signal=SIGTERM $job_id
        echo "✓ Signal sent to job $job_id"
        echo ""
        echo "Monitor the job status with: squeue -j $job_id"
        echo "Or check logs in: logs/train_${job_id}.out"
    else
        echo "⚠ WARNING: Force canceling job $job_id (may lose unsaved progress!)"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            scancel $job_id
            echo "✓ Job $job_id canceled"
        else
            echo "Canceled."
            return 1
        fi
    fi
}

# Main script
main() {
    # Check if job ID provided as argument
    if [ $# -gt 0 ]; then
        if [ "$1" = "--force" ] || [ "$1" = "-f" ]; then
            if [ -z "$2" ]; then
                echo "Error: Job ID required when using --force"
                echo "Usage: $0 [job_id] [--force]"
                exit 1
            fi
            pause_job "$2" false
            exit $?
        else
            pause_job "$1" true
            exit $?
        fi
    fi
    
    # No arguments - show interactive menu
    echo "=== SLURM Training Jobs ==="
    jobs=$(find_training_jobs)
    
    if [ -z "$jobs" ]; then
        echo "No training jobs found."
        echo ""
        echo "To pause a specific job:"
        echo "  $0 <job_id>"
        echo ""
        echo "To find your jobs:"
        echo "  squeue -u $USER"
        exit 0
    fi
    
    echo "$jobs"
    echo ""
    echo "To pause a job gracefully (saves checkpoint):"
    echo "  $0 <job_id>"
    echo ""
    echo "To force cancel a job (may lose progress):"
    echo "  $0 <job_id> --force"
    echo ""
    read -p "Enter job ID to pause (or press Enter to exit): " job_id
    
    if [ -n "$job_id" ]; then
        pause_job "$job_id" true
    else
        echo "Exiting."
    fi
}

main "$@"

