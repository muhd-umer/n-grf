#!/bin/bash

set -e

echo "================================================================="
echo "Starting training process..."
echo "Once training completes, this instance will automatically stop."
echo "================================================================="

python train.py "$@" | tee $LOG_FILE
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "================================================================="
    echo "Training completed successfully at $(date)"
    echo "Shutting down instance in 60 seconds..."
    echo "Run 'sudo shutdown -c' to cancel if needed"

    sudo shutdown -h +1 "Training completed, instance shutting down."
else
    echo "================================================================="
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Instance will NOT be shut down automatically."
    echo "Please check the logs and take appropriate action."
    exit $TRAIN_EXIT_CODE
fi
