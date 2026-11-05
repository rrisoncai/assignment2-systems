"""pytorch_attention / torch_compile (a): benchmark single-head attention at different scales.

For each (d_model, seq_len): time 100 forward passes, measure memory in use right before the
backward pass, and time 100 backward passes (through the same graph, retain_graph=True).
Batch size 8, no head dimension, causal mask. --compile also benchmarks torch.compile'd attention.
"""

import argparse
import timeit
import types
from contextlib import nullcontext

import torch
import torch._functorch.config

import cs336_basics.model
from cs336_basics.model import scaled_dot_product_attention

# NVTX ranges inside the model's attention would cause torch.compile graph breaks; make them no-ops.
cs336_basics.model.nvtx = types.SimpleNamespace(range=lambda *a, **k: nullcontext())
# We time 100 backward passes through one graph (retain_graph=True), which compiled backward only
# allows without buffer donation.
torch._functorch.config.donated_buffer = False
# Compile a specialized graph per shape (as each config would be in practice) instead of switching
# to a dynamic-shape graph after the first few sizes.
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.cache_size_limit = 64

BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
N_WARMUP = 10
N_ITERS = 100


def mib(num_bytes):
    return num_bytes / 2**20


def bench_one(attn, d_model, seq_len, device):
    Q, K, V = (torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True) for _ in range(3))
    seq = torch.arange(seq_len, device=device)
    mask = seq[None, :] <= seq[:, None]

    # Warm up both the grad and no_grad paths (torch.compile compiles each separately).
    for _ in range(N_WARMUP):
        attn(Q, K, V, mask).sum().backward()
        with torch.no_grad():
            attn(Q, K, V, mask)
    torch.cuda.synchronize()

    # Forward: inference-style, nothing saved for backward.
    with torch.no_grad():
        t0 = timeit.default_timer()
        for _ in range(N_ITERS):
            attn(Q, K, V, mask)
            torch.cuda.synchronize()
        fwd_ms = (timeit.default_timer() - t0) * 1000 / N_ITERS

    # Memory in use before backward: inputs + everything autograd saved for this forward.
    Q.grad = K.grad = V.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = attn(Q, K, V, mask)
    grad_out = torch.ones_like(out)
    torch.cuda.synchronize()
    mem_before_bwd = mib(torch.cuda.memory_allocated())
    peak_fwd = mib(torch.cuda.max_memory_allocated())

    t0 = timeit.default_timer()
    for _ in range(N_ITERS):
        out.backward(grad_out, retain_graph=True)
        torch.cuda.synchronize()
    bwd_ms = (timeit.default_timer() - t0) * 1000 / N_ITERS
    return fwd_ms, bwd_ms, mem_before_bwd, peak_fwd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="also benchmark torch.compile(attention)")
    parser.add_argument("--compiled_only", action="store_true", help="only benchmark the compiled version")
    args = parser.parse_args()

    device = torch.device("cuda")
    variants = {} if args.compiled_only else {"eager": scaled_dot_product_attention}
    if args.compile or args.compiled_only:
        variants["compiled"] = torch.compile(scaled_dot_product_attention)

    print("| variant | d_model | seq_len | forward (ms) | backward (ms) | mem before backward (MiB) | peak in forward (MiB) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, attn in variants.items():
        for d_model in D_MODELS:
            for seq_len in SEQ_LENS:
                torch.cuda.empty_cache()
                try:
                    fwd_ms, bwd_ms, mem, peak = bench_one(attn, d_model, seq_len, device)
                    print(f"| {name} | {d_model} | {seq_len} | {fwd_ms:.3f} | {bwd_ms:.3f} | {mem:.1f} | {peak:.1f} |", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"| {name} | {d_model} | {seq_len} | OOM | OOM | OOM | OOM |", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
