#!/bin/bash

# scheduler.sh - Wait 3 minutes then run train.sh 8 times with index*3 values

set -e  # Exit on any error

NUM_RUNS=8



echo "Wait complete. Starting $NUM_RUNS runs of $SCRIPT_PATH"

# Run the script 8 times
for i in $(seq 0 $((NUM_RUNS - 1))); do
    value=$((i * 3))
    echo "Run $((i + 1))/$NUM_RUNS"
    
    sbatch prompt_classify.sh $value
    sleep 180
done

echo "All $NUM_RUNS runs completed successfully!"