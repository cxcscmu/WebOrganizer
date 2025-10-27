#!/bin/bash
#SBATCH -J prompt_classify
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH -N 1 -c 24 --mem=100G --gres=gpu:L40S:2
#SBATCH -t 0-48
#SBATCH -a 0-31
#SBATCH -p preempt
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-[23,27,31,37],babel-1-[23,27],babel-1-31,babel-3-21,babel-4-[1,17,25,21,33,37],babel-6-[9,13],babel-7-[9,17],babel-11-[9,21],babel-14-[1,21,37],babel-15-32,babel-10-5,babel-14-13,babel-14-25,babel-14-29,babel-2-25,babel-13-1,babel-12-9,babel-6-5,babel-12-13,babel-15-36,babel-10-13

config=${CONFIG:-taxonomies/skills.yaml}  # defines taxonomy and instructions
model=${MODEL:-14B}  # Llama model to use
size=${SIZE:-10K}  # how many samples to process (across job array)
seed=${SEED:-43}   # random seed for order of categories and few-shot examples

export HF_HOME=/data/group_data/cx_group/
export NCCL_P2P_DISABLE=1
# babel-12-9,babel-13-[1,13,17,25]

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
# job_id=$((SLURM_ARRAY_TASK_ID + $1))
job_id=$SLURM_ARRAY_TASK_ID
port=$((58661 + job_id))

echo "job id: $job_id"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "num_nodes: $num_nodes"
echo "num_gpus: $num_gpus"
echo "port: $port"


export OUTLINES_CACHE_DIR=/tmp/outlines  # Fixes some job issues with outlines cache on shared filesystem
# "'meta-llama/Llama-3.1-${model%-FP8}-Instruct${model#*B}'"

if [ $num_nodes -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    srun bash -c 'HOST_IP=$(hostname -i) python -m sglang.launch_server \
        --model-path "Qwen/Qwen3-14B" \
        --port "'$port'" \
        --tp "'$(($num_gpus * $num_nodes))'" \
        --nnodes "'$num_nodes'" \
        --node-rank "$SLURM_NODEID" \
        --watchdog-timeout 1800 \
        --nccl-init "'$master_addr':'$port'"' &
else
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3-14B \
        --port $port \
        --watchdog-timeout 1800 \
        --tp $num_gpus &
        # --enable-torch-compile \
        # --dtype bfloat16  \
fi

echo "Waiting for sglang server to be ready on port $port..."
while ! nc -z localhost $port; do
  sleep 1
done
echo "Server is ready."

config_name=$(basename $config)
config_name=${config_name%.yaml}
# num_jobs=$SLURM_ARRAY_TASK_COUNT
num_jobs=8

python prompt_classify.py datasets/dclm-refinedweb-sample10k-test.jsonl /data/user_data/gonilude/WebOrganizer/classifier_data/test/dclm-sample${size}-${config_name}-${model}-seed${seed}-job${job_id}-num_jobs${num_jobs}.jsonl \
    --config_path ${config} \
    --num_threads 20 \
    --batch_size 200 \
    --port $port \
    --job_id $job_id \
    --num_jobs $num_jobs \
    --randomize_seed $seed \


kill -9 $(jobs -p)
