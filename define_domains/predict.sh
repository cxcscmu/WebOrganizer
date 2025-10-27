#!/bin/bash
#SBATCH -J predict
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH -N 1 -c 24 --mem=50G --gres=gpu:L40S:1
#SBATCH -t 0-24
#SBATCH -a 0-6
#SBATCH -p preempt
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-[23,27,31,37],babel-1-[23,27],babel-1-31,babel-3-21,babel-4-[1,17,25,33,37],babel-6-[9,13],babel-7-[9,17],babel-11-[9,21],babel-12-9,babel-13-[1,13,17,25],babel-14-[1,21,37],babel-15-32,babel-4-13

export NCCL_P2P_DISABLE=1
export HF_HOME=/data/group_data/cx_group

# job_id=$SLURM_ARRAY_TASK_ID
# num_jobs=$SLURM_ARRAY_TASK_COUNT

job_id=$((SLURM_ARRAY_TASK_ID + 17))
num_jobs=$((SLURM_ARRAY_TASK_COUNT + 17))

python predict.py --model_path checkpoints/gte-base-en-v1.5__bsz512_lr1e-4_epochs5_warmup0.1_url1 --job_id $job_id --num_jobs $num_jobs
echo "Done with job $job_id"
