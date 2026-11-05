#!/bin/bash
# Remaining §1 experiments on the GPU server (run from assignment2-systems/ with env.sh sourced):
#   timing group (GPU $GPU_TIME, sequential so timings aren't disturbed):
#     benchmarking_mixed_precision (c), torch_compile (b), pytorch_attention + torch_compile (a)
#   memory group (GPU $GPU_MEM, in parallel; timings irrelevant): memory_profiling
# Results: results/*.log, results/attention.md, memory/*.pickle + *.png, results/memory.md
set -uo pipefail
GPU_TIME=${GPU_TIME:-0}
GPU_MEM=${GPU_MEM:-1}
mkdir -p results memory

timing_group() {
    export CUDA_VISIBLE_DEVICES=$GPU_TIME
    # benchmarking_mixed_precision (c): FP32 vs BF16 autocast, forward + backward, all sizes
    for size in small medium large xl 2.7B; do
        python benchmark.py --size "$size" > "results/mp_${size}_fp32.log" 2>&1
        python benchmark.py --size "$size" --mixed_precision > "results/mp_${size}_bf16.log" 2>&1
    done
    # torch_compile (b): vanilla vs compiled model
    for size in small medium large; do
        for mode in forward forward_backward train; do
            python benchmark.py --size "$size" --mode "$mode" > "results/compile_${size}_${mode}_vanilla.log" 2>&1
            python benchmark.py --size "$size" --mode "$mode" --compile > "results/compile_${size}_${mode}_compiled.log" 2>&1
        done
    done
    # pytorch_attention + torch_compile (a)
    python cs336_systems/benchmark_pytorch_attention.py --compile > results/attention.md 2>results/attention.err
    echo "timing group done"
}

memory_group() {
    export CUDA_VISIBLE_DEVICES=$GPU_MEM
    # memory_profiling (a)(b): 2.7B, forward-only and full training step, ctx 128/256/512
    for ctx in 128 256 512; do
        for mode in forward train; do
            python benchmark.py --size 2.7B --context_length "$ctx" --mode "$mode" --warmup_step 1 --exec_step 1 \
                --memory_profile "memory/2.7B_ctx${ctx}_${mode}.pickle" > "results/mem_2.7B_ctx${ctx}_${mode}.log" 2>&1
        done
    done
    # memory_profiling (c): BF16 autocast
    for mode in forward train; do
        python benchmark.py --size 2.7B --context_length 512 --mode "$mode" --warmup_step 1 --exec_step 1 --mixed_precision \
            --memory_profile "memory/2.7B_ctx512_${mode}_bf16.pickle" > "results/mem_2.7B_ctx512_${mode}_bf16.log" 2>&1
    done
    # A full training step that fits in 24 GB, for a complete forward/backward/optimizer timeline
    for ctx in 128 256 512; do
        python benchmark.py --size medium --context_length "$ctx" --mode train --warmup_step 1 --exec_step 1 \
            --memory_profile "memory/medium_ctx${ctx}_train.pickle" > "results/mem_medium_ctx${ctx}_train.log" 2>&1
    done
    python scripts/memory_analyze.py memory/*.pickle > results/memory.md 2>&1
    echo "memory group done"
}

timing_group &
memory_group &
wait
echo "all done"
