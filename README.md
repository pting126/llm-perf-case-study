# LLM Training Performance Case Study (PyTorch)

A small, reproducible case study for profiling and optimizing Transformer/LLM-style training workloads on GPUs using PyTorch.
Includes 1-GPU baseline and 2-GPU (DDP) baseline + optimizations.

## What’s inside
- `train.py`: Tiny Transformer language model + synthetic token dataset (no external data dependency)
- Slurm scripts:
  - `slurm_1gpu_baseline.*.sh`: 1-GPU baseline
  - `slurm_2gpu_ddp_baseline.*.sh`: 2-GPU DDP baseline
  - `slurm_2gpu_ddp_optimized.*.sh`: 2-GPU DDP optimized (data pipeline + AMP + optional compile)

> Note: `*.example.sh` are cluster-agnostic templates. Use `*.local.sh` for your own cluster settings (kept untracked).

## Metrics
The training script prints:
- `avg_step_ms` (after warmup)
- `tokens_per_sec(total)` = batch_size × seq_len × world_size / step_time

## Optimizations applied
- **Input pipeline:** `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, non-blocking H2D copies
- **Compute:** mixed precision (`--amp`)
- **(Optional) Compilation:** `--compile` uses `torch.compile(model)`
- **DDP efficiency:** gradient accumulation (`--grad_accum`)

## How to run
### Slurm (example templates)
```bash
sbatch slurm_1gpu_baseline.example.sh
sbatch slurm_2gpu_ddp_baseline.example.sh
sbatch slurm_2gpu_ddp_optimized.example.sh
