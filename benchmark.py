"""
Problem (benchmarking_script): 4 points
(a) Write a script to perform basic end-to-end benchmarking of the forward and backward passes in
your model. Specifically, your script should support the following:
• Given hyperparameters (e.g., number of layers), initialize a model.
• Generate a random batch of data.
• Run w warm-up steps (before you start measuring time), then time the execution of n steps
(either only forward, or both forward and backward passes, depending on an argument). For
timing, you can use the Python timeit module (e.g., either using the timeit function, or
using timeit.default_timer(), which gives you the system’s highest resolution clock, thus
a better default for benchmarking than time.time()).
• Call torch.cuda.synchronize() after each step.
Deliverable: A script that will initialize a basics Transformer model with the given hyperpa-
rameters, create a random batch of data, and time forward and backward passes.
(b) Time the forward and backward passes for the model sizes described in §1.1.2. Use 5 warmup
steps and compute the average and standard deviation of timings over 10 measurement steps.
How long does a forward pass take? How about a backward pass? Do you see high variability
across measurements, or is the standard deviation small?
3Deliverable: A 1-2 sentence response with your timings.
(c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without
the warm-up steps. How does this affect your results? Why do you think this happens? Also try
to run the script with 1 or 2 warm-up steps. Why might the result still be different?
Deliverable: A 2-3 sentence response.
"""

import argparse
import timeit
import types
from contextlib import nullcontext

import torch
import numpy as np
import torch.cuda.nvtx as nvtx

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch


MODEL_SPECS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def format_gib(num_bytes):
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--rope_theta", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--warmup_step", type=int, default=5)
    parser.add_argument("--exec_step", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--size", type=str, choices=["custom", *MODEL_SPECS.keys(), "all"], default="custom")
    # forward: inference only (no_grad); forward_backward: no optimizer (benchmarking_script);
    # train: forward + backward + AdamW step (nsys_profile (d), memory_profiling)
    parser.add_argument("--mode", type=str, choices=["forward", "forward_backward", "train"], default="forward_backward")
    parser.add_argument("--mixed_precision", action="store_true", help="BF16 autocast for the forward pass and loss")
    parser.add_argument("--compile", action="store_true", help="torch.compile the whole model")
    parser.add_argument(
        "--memory_profile", type=str, default=None,
        help="record CUDA memory history from the first step and dump a snapshot pickle to this path",
    )

    return parser.parse_args()

def run_benchmark(args, device, config_name, d_model, d_ff, num_layers, num_heads):
    print(
        f"\n=== config={config_name} "
        f"d_model={d_model} d_ff={d_ff} num_layers={num_layers} num_heads={num_heads} ==="
    )

    dataset = np.random.randint(0, args.vocab_size, size=100_000, dtype=np.int64)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    if args.compile:
        model = torch.compile(model)

    num_params = sum(p.numel() for p in model.parameters())
    param_bytes = num_params * 4
    grad_bytes = num_params * 4
    adam_state_bytes = num_params * 8
    print(
        "Memory estimate (fp32): "
        f"params={format_gib(param_bytes)}, "
        f"params+grads={format_gib(param_bytes + grad_bytes)}, "
        f"params+grads+adam={format_gib(param_bytes + grad_bytes + adam_state_bytes)}"
    )

    # Only allocate AdamW state when we actually take optimizer steps.
    optimizer = AdamW(model.parameters()) if args.mode == "train" else None

    x, y = get_batch(
        dataset=dataset,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device.type,
    )

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.mixed_precision else nullcontext()
    )

    def run_step(timings=None):
        """One step in the configured mode. If timings is given, record ms per phase."""
        sync()
        t0 = timeit.default_timer()
        with nvtx.range("forward"), torch.set_grad_enabled(args.mode != "forward"), autocast:
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        sync()
        t1 = timeit.default_timer()
        if args.mode != "forward":
            with nvtx.range("backward"):
                loss.backward()
            sync()
        t2 = timeit.default_timer()
        if optimizer is not None:
            with nvtx.range("optimizer"):
                optimizer.step()
            sync()
        t3 = timeit.default_timer()
        model.zero_grad(set_to_none=True)

        if timings is not None:
            timings["forward"].append((t1 - t0) * 1000)
            timings["backward"].append((t2 - t1) * 1000)
            timings["optimizer"].append((t3 - t2) * 1000)

    # Record from the first step so that a snapshot is still dumped if a step runs out of memory.
    if args.memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    timings = {"forward": [], "backward": [], "optimizer": []}
    try:
        with nvtx.range("warmup"):
            for step in range(args.warmup_step):
                run_step()
                print(f"warmup step {step} done")

        # With `nsys profile --capture-range=cudaProfilerApi`, only the measured steps are traced.
        if device.type == 'cuda':
            torch.cuda.cudart().cudaProfilerStart()
        for step in range(args.exec_step):
            with nvtx.range(f"step_{step}"):
                run_step(timings)
            print(
                f"execute step {step}: forward={timings['forward'][-1]:.2f} ms, "
                f"backward={timings['backward'][-1]:.2f} ms, optimizer={timings['optimizer'][-1]:.2f} ms"
            )
        if device.type == 'cuda':
            torch.cuda.cudart().cudaProfilerStop()
    finally:
        if args.memory_profile:
            torch.cuda.memory._dump_snapshot(args.memory_profile)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"Memory snapshot written to {args.memory_profile}")

    phases = {"forward": ["forward"], "forward_backward": ["forward", "backward"], "train": ["forward", "backward", "optimizer"]}
    for phase in phases[args.mode]:
        mean_ms = np.mean(timings[phase])
        std_ms = np.std(timings[phase], ddof=1) if len(timings[phase]) > 1 else 0.0
        print(f"{phase.capitalize()} average={mean_ms:.2f} ms, std={std_ms:.2f}")
    if device.type == 'cuda':
        print(f"Peak memory: {format_gib(torch.cuda.max_memory_allocated())}")


def main():
    args = parse_args()

    if args.compile:
        # NVTX push/pop calls inside the model cause torch.compile graph breaks; make them no-ops.
        cs336_basics.model.nvtx = types.SimpleNamespace(range=lambda *a, **k: nullcontext())

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config_names = list(MODEL_SPECS.keys()) if args.size == "all" else [args.size]

    for config_name in config_names:
        if config_name == "custom":
            d_model = args.d_model
            d_ff = args.d_ff
            num_layers = args.num_layers
            num_heads = args.num_heads
        else:
            spec = MODEL_SPECS[config_name]
            d_model = spec["d_model"]
            d_ff = spec["d_ff"]
            num_layers = spec["num_layers"]
            num_heads = spec["num_heads"]

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            run_benchmark(args, device, config_name, d_model, d_ff, num_layers, num_heads)
        except Exception as e:
            # einx wraps errors raised inside its generated code, so look through the cause chain.
            err = e
            while err is not None and not isinstance(err, torch.cuda.OutOfMemoryError):
                err = err.__cause__
            if err is None:
                raise
            print(f"OOM: config={config_name} mode={args.mode} context_length={args.context_length}")

if __name__ == "__main__":
    main()
