"""memory_profiling: summarize CUDA memory snapshots written by `benchmark.py --memory_profile`.

For each snapshot pickle:
  - reconstruct the active-memory timeline (like pytorch.org/memory_viz "Active Memory Timeline")
    and save it as a PNG next to the pickle;
  - print peak active memory and the largest allocations with the model code that made them.

Usage: python scripts/memory_analyze.py memory/*.pickle [--top 8]
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def user_frame(frames):
    """First stack frame inside our model/benchmark code (skip torch internals)."""
    for f in frames or []:
        fn = f.get("filename", "")
        if "cs336_basics" in fn or "benchmark" in fn or "cs336_systems" in fn:
            return f"{Path(fn).name}:{f.get('line')} {f.get('name')}"
    return "?"


def timeline(snapshot):
    trace = snapshot["device_traces"][0]
    events = [e for e in trace if e["action"] in ("alloc", "free_completed")]
    # Blocks allocated before recording started (e.g. weights) are still active at dump time;
    # recover them as a constant baseline so the curve shows total allocated memory.
    active_at_dump = sum(
        b["size"] for seg in snapshot["segments"] for b in seg["blocks"] if b["state"] == "active_allocated"
    )
    net = sum(e["size"] if e["action"] == "alloc" else -e["size"] for e in events)
    baseline = active_at_dump - net

    t0 = events[0]["time_us"] if events else 0
    xs, ys, cur = [], [], baseline
    for e in events:
        cur += e["size"] if e["action"] == "alloc" else -e["size"]
        xs.append((e["time_us"] - t0) / 1e3)
        ys.append(cur / 2**30)
    return xs, ys, baseline, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickles", nargs="+")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    for path in map(Path, args.pickles):
        with open(path, "rb") as f:
            snapshot = pickle.load(f)
        xs, ys, baseline, events = timeline(snapshot)
        oom = any(e["action"] == "oom" for e in snapshot["device_traces"][0])

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(xs, ys, lw=1, drawstyle="steps-post")
        ax.fill_between(xs, ys, step="post", alpha=0.3)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("active memory (GiB)")
        ax.set_title(f"Active memory timeline: {path.stem}" + (" (OOM)" if oom else ""))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png = path.with_suffix(".png")
        fig.savefig(png, dpi=120)
        plt.close(fig)

        print(f"\n## {path.stem}{'  (OOM during run)' if oom else ''}")
        print(f"timeline: {png}")
        print(f"baseline (allocated before recording): {baseline / 2**30:.2f} GiB, peak active: {max(ys, default=0):.2f} GiB")

        allocs = [e for e in events if e["action"] == "alloc"]
        by_size = Counter()
        where = {}
        for e in allocs:
            key = (e["size"], user_frame(e.get("frames")))
            by_size[key] += 1
            where[key] = key[1]
        print(f"| size (MiB) | count | allocated at |")
        print("|---:|---:|---|")
        for (size, frame), cnt in sorted(by_size.items(), key=lambda kv: -kv[0][0])[: args.top]:
            print(f"| {size / 2**20:.1f} | {cnt} | `{frame}` |")


if __name__ == "__main__":
    main()
