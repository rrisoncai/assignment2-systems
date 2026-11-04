#!/bin/bash
# nsys_profile sweep: every model size x context length.
# For each config, profile a full training step (--mode train); if it OOMs, fall back to
# forward_backward, then forward, and record which mode fit.
# Only the measured steps are captured (benchmark.py calls cudaProfilerStart/Stop around them).
# Runs sequentially on one GPU: parallel runs contend for CPU and distort the timings.
# Run from assignment2-systems/ with env.sh sourced.
set -uo pipefail

OUT=nsys
mkdir -p "$OUT"
SUMMARY="$OUT/sweep_summary.tsv"
: > "$SUMMARY"

profile_size() {
    local size=$1 gpu=$2
    for ctx in 128 256 512 1024; do
        for mode in train forward_backward forward; do
            local name="$OUT/${size}_ctx${ctx}_${mode}"
            CUDA_VISIBLE_DEVICES=$gpu nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -o "$name" --force-overwrite true \
                python benchmark.py --size "$size" --context_length "$ctx" --mode "$mode" > "$name.log" 2>&1
            if grep -q "^OOM" "$name.log"; then
                rm -f "$name.nsys-rep" "$name.qdstrm"
                printf "%s\t%s\t%s\tOOM\n" "$size" "$ctx" "$mode" >> "$SUMMARY"
                continue
            fi
            printf "%s\t%s\t%s\tok\n" "$size" "$ctx" "$mode" >> "$SUMMARY"
            break
        done
    done
}

GPU=${GPU:-0}
for size in small medium large xl 2.7B; do
    profile_size "$size" "$GPU"
done
echo "sweep done"
sort "$SUMMARY"
