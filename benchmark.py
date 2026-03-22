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
import torch
import numpy as np
import torch.cuda.nvtx as nvtx

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

    optimizer = AdamW(
        model.parameters(),
    )

    optimizer.zero_grad(set_to_none=True)

    x, y = get_batch(
        dataset=dataset,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device.type,
    )

    for step in range(args.warmup_step):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        print(f"warmup step {step} done")

    forward_timing = []
    backward_timing = []
    for step in range(args.exec_step):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        forward_t0 = timeit.default_timer()
        with nvtx.range("forward"):
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        forward_ms = (timeit.default_timer() - forward_t0) * 1000
        forward_timing.append(forward_ms)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        backward_t0 = timeit.default_timer()
        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("optimizer"):
            optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        backward_ms = (timeit.default_timer() - backward_t0) * 1000
        backward_timing.append(backward_ms)

        optimizer.zero_grad(set_to_none=True)

        print(f"execute step {step}: forward={forward_ms:.2f} ms, backward={backward_ms:.2f} ms")

        if device.type == 'cuda':
            torch.cuda.synchronize()

    forward_mean_ms = np.mean(forward_timing)
    forward_std_ms = np.std(forward_timing, ddof=1) if len(forward_timing) > 1 else 0.0
    backward_mean_ms = np.mean(backward_timing)
    backward_std_ms = np.std(backward_timing, ddof=1) if len(backward_timing) > 1 else 0.0
    print(f"Forward average={forward_mean_ms:.2f} ms, std={forward_std_ms:.2f}")
    print(f"Backward average={backward_mean_ms:.2f} ms, std={backward_std_ms:.2f}")


def main():
    args = parse_args()

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

        run_benchmark(args, device, config_name, d_model, d_ff, num_layers, num_heads)

if __name__ == "__main__":
    main()
