#!/bin/bash
#SBATCH -J train_classifier
#SBATCH -N 1 -c 20 --gres=gpu:L40S:4 --mem=128G
#SBATCH --output=slurm/%x-%j.out
#SBATCH -t 0-6
#SBATCH -p general
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-[23,27,31,37],babel-1-[23,27],babel-1-31,babel-3-21,babel-4-[1,17,25,33,37],babel-6-[9,13],babel-7-[9,17],babel-11-[9,21],babel-12-9,babel-13-[1,13,17,25],babel-14-[1,21,37],babel-15-32,babel-4-13



model=${MODEL:-"Alibaba-NLP/gte-base-en-v1.5"}  # Model to fine-tune from
bsz=${BSZ:-512}  # Batch size
seq=${SEQ:-32}  # Sequence length
lr=${LR:-1e-4}  # Learning rate
epochs=${EPOCHS:-5}  # Number of epochs
warmup=${WARMUP:-0.1}  # Warmup ratio
dataset=${DATASET:-""}  # Dataset to fine-tune on
url=${URL:-1}  # Whether to use URL in input template


run_name="$(basename $model)_$(basename $dataset)_bsz${bsz}_lr${lr}_epochs${epochs}_warmup${warmup}_url${url}"

out_dir="checkpoints/$run_name"
mkdir -p $out_dir

nvidia-smi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_gpus=${NUM_GPUS:-$num_gpus}
master_port=54321

header="torchrun \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:$master_port \
--nnodes=1 \
--nproc_per_node=$num_gpus \
sanity_check_trainer.py"

accu=$(($bsz / $seq / $num_gpus))

export OMP_NUM_THREADS=$num_gpus

export WANDB_PROJECT="weborganizer"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline"

base_arguments=(
    --report_to wandb

    --do_train
    --do_eval
    --do_predict

    --model_name $model

    --run_name $run_name
    --output_dir $out_dir
    --gradient_accumulation_steps $accu
    --per_device_train_batch_size $seq
    --learning_rate $lr
    --max_grad_norm 1.0
    --weight_decay 0.1
    --warmup_ratio $warmup
    --logging_steps 1
    --log_level info

    --eval_strategy epoch
    --save_strategy epoch
    --load_best_model_at_end true
    --metric_for_best_mode eval_validation_accuracy_label_min
    --greater_is_better true

    --num_train_epochs $epochs
    --dataloader_num_workers 8
    --overwrite_output_dir
    --remove_unused_columns false
    --disable_tqdm true
    --bf16
    --ddp_find_unused_parameters true

    --max_length 8192
    --label_field label

    --train_dataset sanity_check_sample-10K_model-llama70B-train.jsonl
    --validation_dataset sanity_check_sample-10K_model-llama70B-val.jsonl
    --test_dataset sanity_check_sample-10K_model-llama70B-test.jsonl

    --trust_remote_code
    --use_memory_efficient_attention
    --unpad_inputs

    $@
)

if [ $url -eq 1 ]; then
    base_arguments+=(
        --template '{url}

{text}'
    )
fi


echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out
