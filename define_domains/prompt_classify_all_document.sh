#!/bin/bash
#SBATCH -J all_documents_classify
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH -N 1 -c 24 --mem=100G --gres=gpu:L40S:1
#SBATCH -t 0-48
#SBATCH -a 24-31
#SBATCH -p general
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-[23,27,31,37],babel-1-[23,27],babel-1-31,babel-3-21,babel-4-[1,17,25,21,33,37],babel-6-[9,13],babel-7-[9,17],babel-11-[9,21],babel-14-[1,21,37],babel-15-32,babel-10-5,babel-14-13,babel-14-25,babel-14-29,babel-2-25,babel-13-1,babel-12-9,babel-6-5,babel-12-13,babel-15-36,babel-10-13

config=${CONFIG:-taxonomies/skills.yaml}  # defines taxonomy and instructions
model=${MODEL:-8B}  # Llama model to use
size=${SIZE:-10K}  # how many samples to process (across job array)
seed=${SEED:-43}   # random seed for order of categories and few-shot examples
home_dir=${HOME_DIR:-/data/group_data/cx_group/WebOrganizer/Corpus-30B/documents/}  # directory to search for jsonl files

export HF_HOME=/data/group_data/cx_group/
export NCCL_P2P_DISABLE=1

# Set job-specific environment variable for exit control
job_id=$SLURM_ARRAY_TASK_ID
export JOB_CONTINUE_PROCESSING="true"
export JOB_EXIT_FLAG_FILE="/tmp/job_exit_${job_id}.flag"


# Convert size to number of samples to process
if [[ $size == *M ]]; then
    max_index=$((${size%M} * 1000000))
elif [[ $size = *K ]]; then
    max_index=$((${size%K} * 1000))
else
    max_index=$size
fi

# Get number of available GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_nodes=${NUM_NODES:-${SLURM_JOB_NUM_NODES:-1}}
job_id=$SLURM_ARRAY_TASK_ID
port=$((58661 + job_id))

echo "job id: $job_id"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "num_nodes: $num_nodes"
echo "num_gpus: $num_gpus"
echo "port: $port"
echo "JOB_CONTINUE_PROCESSING: $JOB_CONTINUE_PROCESSING"
echo "JOB_EXIT_FLAG_FILE: $JOB_EXIT_FLAG_FILE"

export OUTLINES_CACHE_DIR=/scratch/  # Fixes some job issues with outlines cache on shared filesystem

# Find and collect jsonl files from home_dir
echo "Searching for .jsonl files in: $home_dir"
if [ ! -d "$home_dir" ]; then
    echo "Error: Directory $home_dir does not exist!"
    exit 1
fi

# Get first 1500 jsonl files from home_dir
mapfile -t all_jsonl_files < <(find "$home_dir" -name "*.jsonl" -type f | head -1500)
total_files=${#all_jsonl_files[@]}

echo "Found $total_files .jsonl files in $home_dir"

if [ $total_files -eq 0 ]; then
    echo "No .jsonl files found in $home_dir"
    exit 1
fi

# Calculate job distribution
num_jobs=32  # Default to 32 if not set
files_per_job=$((total_files / num_jobs))
remainder=$((total_files % num_jobs))

# Calculate start and end indices for this job
start_idx=$((job_id * files_per_job))
if [ $job_id -lt $remainder ]; then
    start_idx=$((start_idx + job_id))
    end_idx=$((start_idx + files_per_job))
else
    start_idx=$((start_idx + remainder))
    end_idx=$((start_idx + files_per_job - 1))
fi

echo "Job $job_id processing files from index $start_idx to $end_idx (total: $((end_idx - start_idx + 1)) files)"

# Extract files for this job
job_files=("${all_jsonl_files[@]:$start_idx:$((end_idx - start_idx + 1))}")

if [ ${#job_files[@]} -eq 0 ]; then
    echo "No files assigned to job $job_id"
    exit 0
fi

# Start sglang server
if [ $num_nodes -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    srun bash -c 'HOST_IP=$(hostname -i) python -m sglang.launch_server \
        --model-path "Qwen/Qwen3-8B" \
        --port "'$port'" \
        --tp "'$(($num_gpus * $num_nodes))'" \
        --nnodes "'$num_nodes'" \
        --node-rank "$SLURM_NODEID" \
        --watchdog-timeout 1800 \
        --nccl-init "'$master_addr':'$port'"' &
else
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3-8B \
        --port $port \
        --watchdog-timeout 1800 \
        --tp $num_gpus &
fi

echo "Waiting for sglang server to be ready on port $port..."
while ! nc -z localhost $port; do
  sleep 1
done
echo "Server is ready."

config_name=$(basename $config)
config_name=${config_name%.yaml}

# Function to check if job should continue processing
check_continue_processing() {
    # Check environment variable
    if [ "$JOB_CONTINUE_PROCESSING" != "true" ]; then
        return 1
    fi
    
    # Check for exit flag file
    if [ -f "$JOB_EXIT_FLAG_FILE" ]; then
        return 1
    fi
    
    return 0
}

# Process each file in the job group
for ((i=0; i<${#job_files[@]}; i++)); do
    input_file="${job_files[$i]}"
    filename=$(basename "$input_file" .jsonl)
    output_file="/data/user_data/gonilude/WebOrganizer/Corpus-30B-v2/documents/${filename}.jsonl"
    
    echo "Processing file $((i+1))/${#job_files[@]}: $input_file"
    echo "Output: $output_file"
    echo "File name: $filename"
    
    python prompt_classify.py "$input_file" "$output_file" \
        --config_path ${config} \
        --num_threads 20 \
        --batch_size 1000 \
        --port $port \
        --job_id $job_id \
        --num_jobs $num_jobs \
        --no_sharding \
        --randomize_seed $seed
    
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $input_file"
    else
        echo "Error processing: $input_file"
    fi

    # Check again after processing each file
    if ! check_continue_processing; then
        echo "Exit condition detected after processing file $((i+1))/${#job_files[@]}"
        break
    fi

done

echo "Job $job_id completed processing. Processed $((i+1)) files out of ${#job_files[@]} total files"

# Clean up exit flag file if it exists
if [ -f "$JOB_EXIT_FLAG_FILE" ]; then
    rm -f "$JOB_EXIT_FLAG_FILE"
    echo "Cleaned up exit flag file: $JOB_EXIT_FLAG_FILE"
fi

kill -9 $(jobs -p)