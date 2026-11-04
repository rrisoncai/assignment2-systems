# CS336 Spring 2025 — Assignment 2 (Systems) Writeup

Handout: [cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

## Progress

| §  | Problem | Pts | Page | Status |
|----|---------|----:|-----:|--------|
| 1.1.3 | [benchmarking_script](#problem-benchmarking_script-4-points) | 4 | 3 | ✅ done |
| 1.1.4 | [nsys_profile](#problem-nsys_profile-5-points) | 5 | 5 | ✅ done |
| 1.1.5 | [mixed_precision_accumulation](#problem-mixed_precision_accumulation-1-point) | 1 | 6 | TODO |
| 1.1.5 | [benchmarking_mixed_precision](#problem-benchmarking_mixed_precision-2-points) | 2 | 7 | TODO |
| 1.1.6 | [memory_profiling](#problem-memory_profiling-4-points) | 4 | 8 | TODO |
| 1.2.1 | [pytorch_attention](#problem-pytorch_attention-2-points) | 2 | 9 | draft, needs fixes |
| 1.3   | [torch_compile](#problem-torch_compile-2-points) | 2 | 10 | TODO |
| 1.3.2 | [flash_forward](#problem-flash_forward-15-points) | 15 | 19 | TODO |
| 1.3.2 | [flash_backward](#problem-flash_backward-5-points) | 5 | 21 | TODO |
| 1.3.2 | [flash_benchmarking](#problem-flash_benchmarking-5-points) | 5 | 21 | TODO |
| 2.1   | [distributed_communication_single_node](#problem-distributed_communication_single_node-5-points) | 5 | 26 | TODO |
| 2.2   | [naive_ddp](#problem-naive_ddp-5-points) | 5 | 27 | TODO |
| 2.2   | [naive_ddp_benchmarking](#problem-naive_ddp_benchmarking-3-points) | 3 | 27 | TODO |
| 2.3.1 | [minimal_ddp_flat_benchmarking](#problem-minimal_ddp_flat_benchmarking-2-points) | 2 | 28 | TODO |
| 2.3.2 | [ddp_overlap_individual_parameters](#problem-ddp_overlap_individual_parameters-5-points) | 5 | 29 | TODO |
| 2.3.2 | [ddp_overlap_individual_parameters_benchmarking](#problem-ddp_overlap_individual_parameters_benchmarking-1-point) | 1 | 30 | TODO |
| 2.3.3 | [ddp_overlap_bucketed](#problem-ddp_overlap_bucketed-8-points) | 8 | 31 | TODO |
| 2.3.3 | [ddp_bucketed_benchmarking](#problem-ddp_bucketed_benchmarking-3-points) | 3 | 31 | TODO |
| 2.4   | [communication_accounting](#problem-communication_accounting-10-points) | 10 | 32 | TODO |
| 3     | [optimizer_state_sharding](#problem-optimizer_state_sharding-15-points) | 15 | 34 | TODO |
| 3     | [optimizer_state_sharding_accounting](#problem-optimizer_state_sharding_accounting-5-points) | 5 | 34 | TODO |

Hardware used: NVIDIA GeForce RTX 4090 D (24 GB; up to 4 GPUs on one PCIe switch, no NVLink) unless noted; some local runs on an RTX 4080 (16 GB). Model sizes (Table 1), vocab 10,000, batch size 4:

| Size | d_model | d_ff | num_layers | num_heads |
|------|--------:|-----:|-----------:|----------:|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 1600 | 6400 | 48 | 25 |
| 2.7B | 2560 | 10240 | 32 | 32 |

---

# 1. Assignment Overview: Optimizing Single-GPU Performance

## Problem (benchmarking_script): 4 points

### (a)
> Deliverable: A script that will initialize a basics Transformer model with the given hyperparameters, create a random batch of data, and time forward and backward passes.

**Answer:** [benchmark.py](./benchmark.py)

### (b)
> Time the forward and backward passes for the model sizes described in §1.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?
>
> Deliverable: A 1-2 sentence response with your timings.

Setup: 1× RTX 4090 D (24 GB), FP32, batch size 4, context length 512, vocab 10,000, 5 warm-up + 10 measured steps, `torch.cuda.synchronize()` after each phase. Backward excludes the optimizer step (`--mode forward_backward`).

| Size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Bwd / Fwd | Peak memory (GiB) |
|------|------------------:|-----------------:|-------------------:|------------------:|----------:|------------------:|
| small | 32.47 | 0.83 | 55.48 | 0.06 | 1.71 | 4.06 |
| medium | 84.65 | 0.18 | 164.25 | 0.18 | 1.94 | 10.57 |
| large | 169.09 | 0.23 | 337.05 | 0.46 | 1.99 | 20.27 |
| xl | OOM | | OOM | | | > 23.6 |
| 2.7B | OOM | | OOM | | | > 23.6 |

**Answer:** A forward pass takes 32 / 85 / 169 ms for small / medium / large, and a backward pass takes 55 / 164 / 337 ms. Backward converges to about 2× the forward time as the model grows, which matches backward doing twice the matmul FLOPs. xl and 2.7B run out of memory on a 24 GB GPU in FP32, because their parameters plus gradients alone are 14.9 / 25.4 GiB before any activations. Variability is very small: the standard deviation is below 1 ms (< 3% of the mean) in every configuration, and the largest value (small forward, 0.83 ms) comes from a single 34.8 ms outlier step.

### (c)
> Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?
>
> Deliverable: A 2-3 sentence response.

| Warm-up steps (small) | Forward mean ± std (ms) | Backward mean ± std (ms) | First measured step fwd / bwd (ms) |
|----------------------:|------------------------:|-------------------------:|-----------------------------------:|
| 0 | 85.11 ± 166.67 | 65.81 ± 31.87 | 559.45 / 156.50 |
| 1 | 32.81 ± 0.30 | 55.59 ± 0.19 | 33.38 / 56.10 |
| 2 | 32.13 ± 0.83 | 55.63 ± 0.36 | 31.96 / 56.56 |
| 5 | 32.47 ± 0.83 | 55.48 ± 0.06 | 32.26 / 55.39 |

**Answer:** Without warm-up, the first step is much slower: the forward takes 559 ms (about 17× steady state) and the backward 157 ms (about 2.8×). This inflates the mean forward time to 85 ms and its std to 167 ms, so the result is dominated by a single outlier. The first iteration pays one-time costs: lazy CUDA context and module/kernel loading, cuBLAS handle creation and algorithm selection, the caching allocator's first `cudaMalloc` calls, and building the autograd machinery for the first backward. With 1 or 2 warm-up steps the numbers already match the 5-step results within noise on this setup. In general they can still differ, because some costs settle over several iterations (the allocator pool still growing, the GPU ramping clocks up from its idle power state, autotuning or `torch.compile` recompilation), so a few extra warm-up steps are the safer default.

---

## Problem (nsys_profile): 5 points

> Profile your forward pass, backward pass, and optimizer step using nsys with each of the model sizes described in Table 1 and context lengths of 128, 256, 512 and 1024 (note OOMs in your report).

Setup: 1× RTX 4090 D, FP32, batch size 4, Nsight Systems 2024.6.2 (`-t cuda,nvtx`). The system nsys 2023.4 failed to import about a third of the traces ("Wrong event order"). Only the 10 measured steps are captured: [benchmark.py](./benchmark.py) calls `cudaProfilerStart/Stop` around them and nsys runs with `--capture-range=cudaProfilerApi`, so warm-up is excluded. For each config I tried a full training step first, then fell back to forward+backward, then forward-only (`no_grad`) on OOM. Sweep: [scripts/nsys_sweep.sh](./scripts/nsys_sweep.sh). Kernel attribution to NVTX ranges (via each kernel's launch on the CPU): [scripts/nsys_analyze.py](./scripts/nsys_analyze.py).

Mean per-phase time over 10 steps (GPU span of the NVTX range, ms), shown as forward / backward / optimizer:

| Size \ context_length | 128 | 256 | 512 | 1024 |
|------|----:|----:|----:|-----:|
| small | 39.8 / 43.3 / 27.8 | 40.4 / 44.8 / 27.8 | 41.5 / 56.3 / 28.3 | 75.8 / 157.5 / 28.1 |
| medium | 80.2 / 97.0 / 55.8 | 82.3 / 88.9 / 56.8 | 86.2 / 165.6 / 55.1 | 208.7 / OOM / OOM |
| large | 122.0 / 136.1 / 87.6 | 119.4 / 146.8 / 87.8 | 170.6 / 339.1 / OOM | 424.7 / OOM / OOM |
| xl | 159.4 / 180.0 / OOM | 161.8 / 293.2 / OOM | 325.9 / OOM / OOM | 796.1 / OOM / OOM |
| 2.7B | 112.9 / OOM / OOM | 209.4 / OOM / OOM | 444.4 / OOM / OOM | 1030.3 / OOM / OOM |

"OOM" means that phase doesn't fit in 24 GB. When only the optimizer OOMs, forward/backward come from the `forward_backward` run. When backward also OOMs, the forward comes from a `no_grad` inference run.

### (a)
> What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?
>
> Deliverable: A 1-2 sentence response.

**Answer:** At context 512 the forward pass takes 41.5 / 86.2 / 170.6 ms for small / medium / large (table above). This matches the Python `timeit` numbers from the same runs within 0.2 ms, since we synchronize at the end of every phase. It also matches the un-profiled benchmark (84.7 / 169.1 ms for medium / large) within about 2%. The exception is small: 41.5 ms under nsys vs. 32.5 ms without. Small is launch-bound, with only 24 ms of kernel time inside a 41 ms forward, so the per-launch overhead of CUDA API tracing shows up directly.

### (b)
> What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes?
>
> Deliverable: A 1-2 sentence response.

**Answer:** In the forward pass the top kernel is cuBLAS's FP32 SIMT GEMM `ampere_sgemm_128x64_tn`: 57.6% of forward GPU time for medium at context 512. It is invoked 168 times per forward, which is 7 per layer × 24 layers: the Q, K, V and output projections plus the three SwiGLU matrices. Small calls it 84 = 7 × 12 times. For a full training step it is no longer the top kernel. Instead the top kernels are `vectorized_elementwise_kernel` (32% for medium, about 5,000 launches per step, mostly the backward mul/add and the AdamW update) and CUTLASS's `cutlass::Kernel2<cutlass_80_simt_sgemm_...>` (the backward-pass GEMMs; 23% for medium, and the top kernel for xl at 27%).

| medium, ctx 512 | Top kernel | ms / step | calls / step | % of GPU time |
|---|---|---:|---:|---:|
| forward | `ampere_sgemm_128x64_tn` | 46.9 | 168 | 57.6% |
| full training step | `vectorized_elementwise_kernel` | 87.8 | 4996 | 32.4% |
| | `cutlass::Kernel2` (SIMT sgemm) | 62.5 | 242 | 23.1% |
| | `ampere_sgemm_128x64_tn` | 46.9 | 168 | 17.3% |

### (c)
> What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?
>
> Deliverable: A 1-2 sentence response.

**Answer:** Elementwise kernels make up about 30% of forward GPU time for medium at context 512: `vectorized_elementwise_kernel` 16.6% and `elementwise_kernel` 14.7%. These are the multiplies (SwiGLU gating, RMSNorm scale, attention scale), the residual adds, `exp`/`neg`/`where` for softmax, SiLU and masking, and copies to make tensors contiguous when splitting and merging heads. `reduce_kernel` (the softmax max/sum and the RMSNorm mean) adds about 3.6%. Their share grows with context length, because the attention-score elementwise ops scale as O(seq²): matmuls drop to 45–60% of forward time at context 1024 for small through xl (44.7% for small, 58.8% for xl).

### (d)
> Profile running one complete training step with your implementation of AdamW. How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?
>
> Deliverable: A 1-2 sentence response.

**Answer:** The matmul share of GPU time drops from the forward pass to a full training step:

| Config | Matmul share, forward | Matmul share, full train step |
|---|---:|---:|
| small, ctx 512 | 61.0% | 48.5% |
| medium, ctx 512 | 64.7% | 50.7% |
| large, ctx 256 | 82.3% | 51.8% |

The backward pass roughly doubles the matmul work, but it also adds even more elementwise work: the gradients of every pointwise op and the gradient accumulation. The AdamW step is purely elementwise: about 28 ms of small kernels per step for medium, several per parameter tensor. So elementwise kernels become the single largest category (`vectorized_elementwise_kernel` is about 32% of the step). The optimizer is also launch-bound, taking about 55 ms of wall time for 28 ms of kernels.

### (e)
> Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?
>
> Deliverable: A 1-2 sentence response.

**Answer:** The two attention matmuls (QKᵀ and PV) cost about 4·B·H·S²·d_head FLOPs per layer. Softmax costs about 5·B·H·S² (max, subtract, exp, sum, divide), so it has about 4·64/5 ≈ 50× fewer FLOPs (d_head = 64). Yet its runtime is comparable to or larger than both matmuls together, and the gap grows with context length:

| Config (forward, ms/step) | QKᵀ | softmax | PV | softmax / (QKᵀ+PV) |
|---|---:|---:|---:|---:|
| medium, ctx 256 | 1.08 | 1.20 | 1.25 | 0.52 |
| medium, ctx 512 | 4.96 | 11.51 | 3.54 | 1.35 |
| medium, ctx 1024 | 21.58 | 56.02 | 11.13 | 1.71 |
| xl, ctx 1024 | 68.44 | 174.50 | 34.59 | 1.69 |

Softmax is memory-bound: each of its elementwise passes reads and writes the whole B·H·S² score tensor from HBM. The matmuls are compute-bound and reuse data from on-chip memory. So FLOPs are a poor predictor of runtime for softmax. This is exactly the cost FlashAttention removes by fusing softmax into the tiled matmuls. The masking `where` adds roughly another 0.3× of the matmul time at context ≥ 512.

---

## Problem (mixed_precision_accumulation): 1 point

> Run the given accumulation code (fp32 / fp16 / mixed sums of 0.01 × 1000) and comment on the (accuracy of the) results.
>
> Deliverable: A 2-3 sentence response.

| Accumulator | Addend | Result |
|-------------|--------|--------|
| float32 | float32 |  |
| float16 | float16 |  |
| float32 | float16 |  |
| float32 | float16 → float32 |  |

**Answer:**

---

## Problem (benchmarking_mixed_precision): 2 points

### (a)
> For `ToyModel` trained with FP16 autocast (parameters originally FP32), what are the data types of the following?
>
> Deliverable: The data types for each of the components listed above.

| Component | dtype |
|-----------|-------|
| model parameters within the autocast context |  |
| output of `fc1` |  |
| output of `ln` |  |
| predicted logits |  |
| loss |  |
| gradients |  |

### (b)
> What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?
>
> Deliverable: A 2-3 sentence response.

**Answer:**

### (c)
> Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each model size. Compare and comment on any trends as model size changes.
>
> Deliverable: A 2-3 sentence response with your timings and commentary.

| Size | FP32 fwd (ms) | BF16 fwd (ms) | FP32 bwd (ms) | BF16 bwd (ms) |
|------|--------------:|--------------:|--------------:|--------------:|
| small |  |  |  |  |
| medium |  |  |  |  |
| large |  |  |  |  |
| xl |  |  |  |  |
| 2.7B |  |  |  |  |

**Answer:**

---

## Problem (memory_profiling): 4 points

> Profile the forward pass, backward pass, and optimizer step of the 2.7B model with context lengths of 128, 256 and 512.

### (a)
> Add an option to your profiling script to run the memory profiler. How do your memory timelines look? Can you tell which stage is running based on the peaks you see?
>
> Deliverable: Two images of the "Active memory timeline" (forward only, and full training step), and a 2-3 sentence response.

<!-- ![forward](./figures/mem_forward.png) -->
<!-- ![train step](./figures/mem_train_step.png) -->

**Answer:**

### (b)
> What is the peak memory usage of each context length when doing a forward pass? What about a full training step?
>
> Deliverable: A table with two numbers per context length.

| context_length | Forward peak (GB) | Train step peak (GB) |
|---------------:|------------------:|---------------------:|
| 128 |  |  |
| 256 |  |  |
| 512 |  |  |

### (c)
> Find the peak memory usage of the 2.7B model when using mixed-precision, for both a forward pass and a full optimizer step. Does mixed-precision significantly affect memory usage?
>
> Deliverable: A 2-3 sentence response.

**Answer:**

### (d)
> At our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MB.
>
> Deliverable: A 1-2 sentence response with your derivation.

**Answer:**

### (e)
> In memory_viz with reduced "Detail", what is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?
>
> Deliverable: A 1-2 sentence response.

**Answer:**

---

## Problem (pytorch_attention): 2 points

> Batch size 8, no head dimension; d_model ∈ {16, 32, 64, 128} × seq_len ∈ {256, 1024, 4096, 8192, 16384}. Time 100 forward passes, measure memory in use before the backward pass starts, and time 100 backward passes. At what size do you get OOM? Do the memory accounting for the smallest OOM config. How does the memory saved for backward change with sequence length? What would you do to eliminate this memory cost?
>
> Deliverable: A table with your timings, your working out for the memory usage, and a 1-2 paragraph response.

Script: [cs336_systems/benchmark_pytorch_attention.py](./cs336_systems/benchmark_pytorch_attention.py)

> **TODO (fixes before finalizing):**
> - Memory is measured as peak during a `no_grad` forward, not "memory in use before backward". Run the forward with grad, then read `torch.cuda.memory_allocated()` before calling `.backward()`.
> - The backward timing loop also includes `randn` + forward; time only `loss.backward()` (or relabel as fwd+bwd).
> - OOM arithmetic: `8 × 16384² × 4 B ≈ 8.6 GB` (not 32 GB); OOM on 16 GB comes from several such seq×seq tensors alive at once (scores, masked scores, softmax probs, mask).

**Forward time (s), 100 iterations total**

| seq_len \ d_model | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 256 | 0.0090 | 0.0098 | 0.0100 | 0.0093 |
| 1024 | 0.0479 | 0.0517 | 0.0540 | 0.0531 |
| 4096 | 1.2287 | 1.2319 | 1.2404 | 1.2787 |
| 8192 | 4.8290 | 4.8226 | 4.8511 | 5.0430 |
| 16384 | OOM | OOM | OOM | OOM |

**Backward time (s), 100 iterations total**

| seq_len \ d_model | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 256 | 0.0537 | 0.0439 | 0.0401 | 0.0560 |
| 1024 | 0.1767 | 0.1800 | 0.1829 | 0.1891 |
| 4096 | 4.1112 | 4.1102 | 4.1255 | 4.2234 |
| 8192 | 16.0184 | 16.0260 | 16.1064 | 16.5473 |
| 16384 | OOM | OOM | OOM | OOM |

**Memory before backward (MB)**

| seq_len \ d_model | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 256 | 20.72 | 53.35 | 78.35 | 128.35 |
| 1024 | 212.13 | 214.88 | 220.38 | 231.38 |
| 4096 | 3115.78 | 3126.78 | 3148.78 | 3192.78 |
| 8192 | 12397.31 | 12425.31 | 12481.31 | 12593.31 |
| 16384 | OOM | OOM | OOM | OOM |

**Memory after backward (MB)**

| seq_len \ d_model | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 256 | 29.34 | 50.09 | 75.84 | 127.34 |
| 1024 | 213.35 | 217.35 | 225.35 | 241.35 |
| 4096 | 3120.66 | 3136.66 | 3168.66 | 3232.66 |
| 8192 | 12401.06 | 12433.06 | 12497.06 | 12625.06 |
| 16384 | OOM | OOM | OOM | OOM |

**Memory accounting (seq_len = 16384):**

- OOM 边界：所有 `d_model` 配置在 `seq_len=16384` 时均 OOM，而 `seq_len=8192` 可以运行。OOM 的根本原因是 attention score 矩阵大小为 `batch × seq² × 4 bytes`，在 `seq=16384, batch=8` 时约需 **32GB**（`8 × 16384² × 4B ≈ 32GB`），超出 GPU 显存。

**Answer:**

- **时间与序列长度的关系**：forward/backward 时间随 `seq_len` 呈近似 **O(seq²)** 增长（从 1024→4096 约 ×16→×25，符合二次方特征），这正是标准 attention 的复杂度。
- **backward memory 与 forward memory 差距很小**：这是因为标准 attention 在 backward 时需要保存 attention weight 矩阵（`batch × seq × seq`），而这个矩阵在 forward 时已经占据了绝大部分内存，backward 额外增量相对较小。消除这部分内存开销的方法是使用 **FlashAttention**，它通过分块重计算避免存储完整的 attention weight 矩阵。

---

## Problem (torch_compile): 2 points

### (a)
> Extend your attention benchmarking script to include a compiled version of your PyTorch attention, and compare its performance to the uncompiled version with the same configuration as pytorch_attention.
>
> Deliverable: A table comparing forward and backward timings for compiled vs. uncompiled attention.

| seq_len | d_model | Fwd eager (s) | Fwd compiled (s) | Bwd eager (s) | Bwd compiled (s) |
|--------:|--------:|--------------:|-----------------:|--------------:|-----------------:|
|  |  |  |  |  |  |

### (b)
> Compile your entire Transformer model in your end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps?
>
> Deliverable: A table comparing your vanilla and compiled Transformer model.

| Size | Fwd vanilla (ms) | Fwd compiled (ms) | Fwd+bwd+opt vanilla (ms) | Fwd+bwd+opt compiled (ms) |
|------|-----------------:|------------------:|-------------------------:|--------------------------:|
| small |  |  |  |  |
| medium |  |  |  |  |
| large |  |  |  |  |
| xl |  |  |  |  |
| 2.7B |  |  |  |  |

**Answer:**

---

## Problem (flash_forward): 15 points

### (a)
> Pure PyTorch (no Triton) `autograd.Function` implementing the FlashAttention-2 forward pass (tiles ≥ 16×16), returning O and saving L, Q, K, V, O.
>
> Deliverable: Implement `adapters.get_flashattention_autograd_function_pytorch`; pass `uv run pytest -k test_flash_forward_pass_pytorch`.

**Answer:** <!-- link to implementation; test result -->

### (b)
> Triton kernel for the FlashAttention-2 forward pass (Algorithm 1), launch grid `(T_q, batch_size)`, wrapped in an `autograd.Function`.
>
> Deliverable: Implement `adapters.get_flashattention_autograd_function_triton`; pass `uv run pytest -k test_flash_forward_pass_triton`.

**Answer:**

### (c)
> Add an `is_causal` flag (default `False`) with in-kernel causal masking (`-1e6` on masked elements), saved as `ctx.is_causal`.
>
> Deliverable: An additional flag for the Triton `autograd.Function` implementing causal masking.

**Answer:**

---

## Problem (flash_backward): 5 points

> Implement the backward pass (Eq. 13–19) using PyTorch + `torch.compile`, computing and using the D vector, returning dQ, dK, dV.
>
> Deliverable: Pass `uv run pytest -k test_flash_backward`.

**Answer:**

---

## Problem (flash_benchmarking): 5 points

> Using `triton.testing.do_bench`, compare your (partially) Triton FlashAttention-2 to regular PyTorch attention: forward, backward, and end-to-end latencies. Batch size 1, causal masking; seq_len powers of 2 from 128 to 65536; d powers of 2 from 16 to 128; precisions bfloat16 and float32.
>
> Deliverable: A table of results (forward, backward, end-to-end latencies).

| dtype | seq_len | d | Torch fwd (ms) | Flash fwd (ms) | Torch bwd (ms) | Flash bwd (ms) | Torch e2e (ms) | Flash e2e (ms) |
|-------|--------:|--:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|
|  |  |  |  |  |  |  |  |  |

**Answer:**

---

# 2. Distributed Data Parallel Training

## Problem (distributed_communication_single_node): 5 points

> Benchmark all-reduce in the single-node multi-process setup. Vary backend + device (Gloo + CPU, NCCL + GPU), data size (1MB, 10MB, 100MB, 1GB float32), and number of processes (2, 4, 6).
>
> Deliverable: Plot(s) and/or table(s) comparing the settings, with 2-3 sentences of commentary.

| Backend | Device | World size | 1 MB (ms) | 10 MB (ms) | 100 MB (ms) | 1 GB (ms) |
|---------|--------|-----------:|----------:|-----------:|------------:|----------:|
| gloo | CPU | 2 |  |  |  |  |
| gloo | CPU | 4 |  |  |  |  |
| gloo | CPU | 6 |  |  |  |  |
| nccl | GPU | 2 |  |  |  |  |
| nccl | GPU | 4 |  |  |  |  |
| nccl | GPU | 6 |  |  |  |  |

**Answer:**

---

## Problem (naive_ddp): 5 points

> Write a script to naively perform DDP by all-reducing individual parameter gradients after the backward pass. Verify correctness by training a toy model and checking that weights match single-process training.
>
> Deliverable: The script and verification.

**Answer:**

---

## Problem (naive_ddp_benchmarking): 3 points

> Benchmark your LM trained with naive DDP. Measure total time per training step and the proportion of time spent communicating gradients (1 node × 2 GPUs, XL model).
>
> Deliverable: A description of your benchmarking setup, along with the measured time per training iteration and time spent communicating gradients.

| Setting | Time / iter (ms) | Comm time (ms) | Comm fraction |
|---------|-----------------:|---------------:|--------------:|
| naive DDP (per-parameter all-reduce) |  |  |  |

**Answer:**

---

## Problem (minimal_ddp_flat_benchmarking): 2 points

> Communicate a single flattened tensor of all gradients and compare against per-parameter all-reduce (1 node × 2 GPUs, XL model).
>
> Deliverable: Time per iteration and communication time with a single batched all-reduce; 1-2 sentences comparing batched vs. individual.

| Setting | Time / iter (ms) | Comm time (ms) |
|---------|-----------------:|---------------:|
| per-parameter all-reduce |  |  |
| flattened single all-reduce |  |  |

**Answer:**

---

## Problem (ddp_overlap_individual_parameters): 5 points

> Implement a DDP container class that broadcasts weights at init and overlaps per-parameter gradient all-reduce with the backward pass (`__init__`, `forward`, `finish_gradient_synchronization`).
>
> Deliverable: Implement `adapters.get_ddp_individual_parameters` (and optionally `adapters.ddp_individual_parameters_on_after_backward`); pass `uv run pytest tests/test_ddp_individual_parameters.py` (run ~5 times).

**Answer:**

---

## Problem (ddp_overlap_individual_parameters_benchmarking): 1 point

### (a)
> Benchmark overlapped per-parameter DDP vs. naive per-parameter and flattened DDP (1 node, 2 GPUs, XL).
>
> Deliverable: Time per iteration with overlap, with 1-2 sentences comparing.

| Setting | Time / iter (ms) |
|---------|-----------------:|
| naive per-parameter |  |
| flattened |  |
| overlapped per-parameter |  |

**Answer:**

### (b)
> Use Nsight to compare the initial DDP and the overlapping DDP traces.
>
> Deliverable: 2 screenshots showing that communication is or isn't overlapped with the backward pass.

<!-- ![naive ddp trace](./figures/nsys_naive_ddp.png) -->
<!-- ![overlap ddp trace](./figures/nsys_overlap_ddp.png) -->

**Answer:**

---

## Problem (ddp_overlap_bucketed): 8 points

> Implement a bucketed DDP container (`__init__(module, bucket_size_mb)`, `forward`, `finish_gradient_synchronization`), assigning parameters to buckets in reverse order of `model.parameters()`.
>
> Deliverable: Implement `adapters.get_ddp_bucketed` (and optionally `ddp_bucketed_on_after_backward`, `ddp_bucketed_on_train_batch_start`); pass `pytest tests/test_ddp.py` (run ~5 times).

**Answer:**

---

## Problem (ddp_bucketed_benchmarking): 3 points

### (a)
> Benchmark bucketed DDP (1 node, 2 GPUs, XL) with max bucket sizes 1, 10, 100, 1000 MB. Compare with previous experiments — do results align with expectations? If not, why? What setup changes would make results align?
>
> Deliverable: Time per iteration for various bucket sizes; 3-4 sentence commentary.

| Bucket size (MB) | Time / iter (ms) |
|-----------------:|-----------------:|
| 1 |  |
| 10 |  |
| 100 |  |
| 1000 |  |

**Answer:**

### (b)
> Assuming per-bucket compute time equals per-bucket communication time, model DDP communication overhead (time after backward) as a function of total parameter bytes `s`, all-reduce bandwidth `w`, per-call overhead `o`, and number of buckets `n_b`. Then derive the optimal bucket size.
>
> Deliverable: Equation for DDP overhead, and an equation for the optimal bucket size.

**Answer:**

---

## Problem (communication_accounting): 10 points

> XXL model: d_model = 16384, d_ff = 53248, num_blocks = 126; each block = two linear layers (d_model→d_ff, d_ff→d_model). No activation checkpointing; activations and gradient communication in BF16; accumulated gradients, master weights, and optimizer state in FP32.

### (a)
> Memory for FP32 master weights, accumulated gradients, and optimizer states on a single device? Memory saved for backward (BF16)? How many H100 80GB GPUs worth is this?
>
> Deliverable: Your calculations and a one-sentence response.

**Answer:**

### (b)
> With master weights, optimizer state, gradients, and half of activations sharded across N_FSDP devices, write an expression for per-device memory. What N_FSDP is needed to fit within 1 TPU v5p (95 GB)?
>
> Deliverable: Your calculations and a one-sentence response.

**Answer:**

### (c)
> Forward pass only. W_ici = 2 · 9 · 10¹⁰, C = 4.6 · 10¹⁴ FLOP/s, M_X = 2, M_Y = 1, X = 16 (FSDP), Y = 4 (TP). At what per-device batch size is the model compute bound? What is the overall batch size?
>
> Deliverable: Your calculations and a one-sentence response.

**Answer:**

### (d)
> What other tricks can reduce the overall batch size while retaining high throughput?
>
> Deliverable: A one-paragraph response, backed by references and/or equations.

**Answer:**

---

# 3. Optimizer State Sharding

## Problem (optimizer_state_sharding): 15 points

> Implement a sharded optimizer wrapper (`__init__(params, optimizer_cls, **kwargs)`, `step(closure, **kwargs)`, `add_param_group(param_group)`) where each rank updates its shard and broadcasts updated parameters.
>
> Deliverable: Implement `adapters.get_sharded_optimizer`; pass `uv run pytest tests/test_sharded_optimizer.py` (run ~5 times).

**Answer:**

---

## Problem (optimizer_state_sharding_accounting): 5 points

### (a)
> Profile peak memory with and without optimizer state sharding (1 node, 2 GPUs, XL): after model init, directly before the optimizer step, and directly after. Do results align with expectations? Break down memory by component.
>
> Deliverable: 2-3 sentence response with peak memory results and a breakdown.

| Setting | After init (GB) | Before opt step (GB) | After opt step (GB) |
|---------|----------------:|---------------------:|--------------------:|
| no sharding |  |  |  |
| sharded |  |  |  |

**Answer:**

### (b)
> Time per iteration with and without optimizer state sharding (1 node, 2 GPUs, XL).
>
> Deliverable: 2-3 sentence response with your timings.

| Setting | Time / iter (ms) |
|---------|-----------------:|
| no sharding |  |
| sharded |  |

**Answer:**

### (c)
> How does this approach differ from ZeRO stage 1 (ZeRO-DP P_os, Rajbhandari et al., 2020)?
>
> Deliverable: 2-3 sentence summary of differences, especially memory and communication volume.

**Answer:**
