import os, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------
# Synthetic token dataset
# -----------------------
class RandomTokenDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.int64)
        y = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.int64)
        return x, y

# -----------------------
# Tiny Transformer LM
# -----------------------
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, nlayers: int, dim_ff: int, dropout: float):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.tok(x)               # [B, T, D]
        h = self.enc(h)               # [B, T, D]
        logits = self.lm_head(h)      # [B, T, V]
        return logits

def setup_ddp():
    # torchrun provides these
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    if ddp:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    return ddp, rank, local_rank, world_size

def cleanup_ddp(ddp: bool):
    if ddp:
        torch.distributed.destroy_process_group()

@torch.no_grad()
def print0(rank, *args):
    if rank == 0:
        print(*args, flush=True)

def main():
    p = argparse.ArgumentParser()
    # workload
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)

    # model size (tune for your GPU)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--nlayers", type=int, default=6)
    p.add_argument("--dim_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.0)

    # optimization switches
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--grad_accum", type=int, default=1)

    # dataloader tuning
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)

    # profiling
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile_steps", type=int, default=30)  # after warmup
    args = p.parse_args()

    ddp, rank, local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    # dataset: make it large enough so dataloader is steady
    dataset = RandomTokenDataset(num_samples=200000, seq_len=args.seq_len, vocab_size=args.vocab_size)

    sampler = None
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model = TinyTransformerLM(
        vocab_size=args.vocab_size, d_model=args.d_model, nhead=args.nhead,
        nlayers=args.nlayers, dim_ff=args.dim_ff, dropout=args.dropout
    ).to(device)

    if args.compile:
        # compile after moving to device
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # optional profiler
    prof = None
    if args.profile and rank == 0:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        prof = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./results/tb_trace"),
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=args.profile_steps, repeat=1),
        )

    model.train()
    it = iter(dl)

    step_times = []
    tokens_per_step = args.batch_size * args.seq_len * world_size

    print0(rank, f"DDP={ddp}, world_size={world_size}, device={device}")
    print0(rank, f"Workload: B={args.batch_size}, T={args.seq_len}, tokens/step(total)={tokens_per_step}")
    print0(rank, f"Flags: amp={args.amp}, compile={args.compile}, grad_accum={args.grad_accum}")
    print0(rank, f"DL: workers={args.num_workers}, pin={args.pin_memory}, persist={args.persistent_workers}, prefetch={args.prefetch_factor}")

    # training loop
    global_step = 0
    # quick warm start: a few cuda ops to stabilize
    torch.cuda.synchronize()

    while global_step < args.steps:
        if sampler is not None:
            sampler.set_epoch(global_step)

        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)

        # grad accumulation
        for _ in range(args.grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl)
                x, y = next(it)

            # non-blocking H2D only helps if pin_memory=True
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.float16):
                logits = model(x)  # [B,T,V]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            scaler.scale(loss / args.grad_accum).backward()

        scaler.step(opt)
        scaler.update()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # record timing after warmup
        if global_step >= args.warmup:
            step_times.append((t1 - t0) * 1000.0)

        if prof is not None:
            # start profiler after warmup, run for profile_steps
            if global_step == args.warmup:
                print0(rank, "Starting profiler capture ...")
            if global_step >= args.warmup and global_step < args.warmup + args.profile_steps:
                prof.step()
            if global_step == args.warmup + args.profile_steps:
                print0(rank, "Profiler capture done. Trace written to ./results/tb_trace")
                prof.stop()
                prof = None

        if rank == 0 and (global_step % 50 == 0 or global_step == args.steps - 1):
            if global_step >= args.warmup and len(step_times) > 0:
                avg_ms = sum(step_times[-20:]) / min(20, len(step_times))
                tok_s = (tokens_per_step / (avg_ms / 1000.0))
                print0(rank, f"step {global_step:4d} | loss {loss.item():.3f} | avg_step_ms(last~20) {avg_ms:.2f} | tokens/s {tok_s:.0f}")
            else:
                print0(rank, f"step {global_step:4d} | loss {loss.item():.3f}")

        global_step += 1

    if rank == 0 and len(step_times) > 0:
        avg_ms = sum(step_times) / len(step_times)
        p50 = sorted(step_times)[len(step_times)//2]
        tok_s = tokens_per_step / (avg_ms / 1000.0)
        print("\n=== FINAL ===")
        print(f"avg_step_ms: {avg_ms:.2f}")
        print(f"p50_step_ms: {p50:.2f}")
        print(f"tokens_per_sec(total): {tok_s:.0f}")

    cleanup_ddp(ddp)

if __name__ == "__main__":
    main()