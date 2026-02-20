#!/bin/bash
#SBATCH --job-name=llm_perf_2g_ddp_base
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/2g_ddp_base_%j.out

# load your env here

# IMPORTANT: use torchrun for DDP
torchrun --standalone --nproc_per_node=2 train.py \
  --seq_len 1024 --batch_size 4 --steps 300 --warmup 50 \
  --d_model 512 --nhead 8 --nlayers 6 --dim_ff 2048 \
  --num_workers 2