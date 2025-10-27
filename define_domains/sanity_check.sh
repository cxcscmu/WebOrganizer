#!/bin/bash
#SBATCH -J check
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH -N 1 -c 24 --mem=100G --gres=gpu:L40S:1
#SBATCH -t 0-24
#SBATCH -p general
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-[23,27,31,37],babel-1-[23,27],babel-1-31,babel-3-21,babel-4-[1,17,25,33,37],babel-6-[9,13],babel-7-[9,17],babel-11-[9,21],babel-12-9,babel-13-[1,13,17,25],babel-14-[1,21,37],babel-15-32,babel-4-13

python prompt_classify_vllm.py datasets/dclm-refinedweb-sample1M.jsonl test.jsonl --config_path taxonomies/skills.yaml --shard_id 0 --num_shards 1