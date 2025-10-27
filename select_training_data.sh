#!/bin/bash
#SBATCH -J select_training_data
#SBATCH -N 1 -c 26 --gres=gpu:L40S:1 --mem=128G
#SBATCH --output=slurm/%x-%j.out
#SBATCH -t 0-24
#SBATCH -p general
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-[23,27,31,37],babel-1-[23,27],babel-1-31,babel-3-21,babel-4-[1,17,25,33,37],babel-6-[9,13],babel-7-[9,17],babel-11-[9,21],babel-12-9,babel-13-[1,13,17,25],babel-14-[1,21,37],babel-15-32,babel-4-13,babel-11-25

export NCCL_P2P_DISABLE=1

for i in {1..512}; do
    echo "Running iteration $i/512..."
    python select_training_data.py \
        "/data/user_data/gonilude/WebOrganizer/Corpus-30B-v2/" \
        "/data/group_data/cx_group/WebOrganizer/Corpus-30B/selected-1B/random${i}" \
        --ref_distribution "/data/user_data/gonilude/WebOrganizer/Corpus-30B-v2/training_mixture/random${i}.json" \
        --indices_dir "/data/group_data/cx_group/WebOrganizer/Corpus-30B/selected-1B/indices" \
        --domains_dir "skills" \
        --num_tokens 1000000000 \
        --do_sample \
        --num_proc 24
done