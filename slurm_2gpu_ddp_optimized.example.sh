#!/bin/bash
#SBATCH --job-name=llm_perf_2g_ddp_opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=results/2g_ddp_opt_%j.out
#SBATCH --error=results/2g_ddp_opt_%j.err

set -euo pipefail

# (Example) Activate your environment
# source ~/.bashrc
# conda activate llmperf

torchrun --standalone --nproc_per_node=2 train.py \
  --seq_len 1024 --batch_size 4 --steps 300 --warmup 50 \
  --d_model 512 --nhead 8 --nlayers 6 --dim_ff 2048 \
  --num_workers 8 --pin_memory --persistent_workers --prefetch_factor 4 \
  --amp --compile --grad_accum 2
