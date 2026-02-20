#!/bin/bash
#SBATCH --job-name=llm_perf_1g_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=results/1g_base_%j.out
#SBATCH --error=results/1g_base_%j.err

set -euo pipefail

# (Example) Activate your environment
# source ~/.bashrc
# conda activate llmperf

python -u train.py \
  --seq_len 1024 --batch_size 4 --steps 300 --warmup 50 \
  --d_model 512 --nhead 8 --nlayers 6 --dim_ff 2048 \
  --num_workers 2
