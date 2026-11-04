"""Summarize nsys profiles from scripts/nsys_sweep.sh for the nsys_profile writeup.

For each nsys/*.nsys-rep: export to SQLite, attribute every GPU kernel to the NVTX ranges that
were open on the CPU when it was launched, and report (measured steps only, i.e. inside step_N):
  - per-phase GPU kernel time and GPU span vs. the Python timers   -> (a)
  - top kernels in the forward pass and in the full step           -> (b), (c)
  - matmul share of GPU time, forward vs. full step                -> (d)
  - softmax vs. QK^T / PV matmul time inside attention             -> (e)

Usage: python scripts/nsys_analyze.py [nsys_dir] [--top N]
"""

import argparse
import bisect
import re
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path

MATMUL_RE = re.compile(r"gemm|cutlass|xmma|cublas", re.IGNORECASE)
PHASES = ["forward", "backward", "optimizer"]
ATTN_RANGES = ["qk_matmul", "softmax", "pv_matmul", "apply_mask"]


class Ranges:
    """Non-overlapping instances of one NVTX range name, for fast containment checks."""

    def __init__(self, intervals):
        self.intervals = sorted(intervals)
        self.starts = [s for s, _ in self.intervals]

    def index_of(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        if i >= 0 and t <= self.intervals[i][1]:
            return i
        return None


def export_sqlite(rep: Path) -> Path:
    db = rep.with_suffix(".sqlite")
    if not db.exists() or db.stat().st_mtime < rep.stat().st_mtime:
        subprocess.run(
            ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(db), str(rep)],
            check=True,
            capture_output=True,
        )
    return db


def load(db: Path):
    con = sqlite3.connect(db)
    strings = dict(con.execute("SELECT id, value FROM StringIds"))

    nvtx_cols = {r[1] for r in con.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text_expr = "COALESCE(text, '')" if "textId" not in nvtx_cols else "text, textId"
    by_name = defaultdict(list)
    for row in con.execute(f"SELECT start, end, {text_expr} FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        start, end = row[0], row[1]
        name = row[2] if row[2] else (strings.get(row[3], "") if len(row) > 3 else "")
        by_name[name].append((start, end))

    kernels = con.execute(
        """
        SELECT r.start, k.start, k.end, k.shortName, k.demangledName
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        """
    ).fetchall()
    kernels = [(launch, ks, ke, strings.get(short, "?"), strings.get(dem, "")) for launch, ks, ke, short, dem in kernels]
    con.close()
    return by_name, kernels


def parse_log(log: Path):
    timers = {}
    for line in log.read_text().splitlines():
        m = re.match(r"(Forward|Backward|Optimizer) average=([\d.]+) ms, std=([\d.]+)", line)
        if m:
            timers[m.group(1).lower()] = (float(m.group(2)), float(m.group(3)))
        m = re.match(r"Peak memory: ([\d.]+) GiB", line)
        if m:
            timers["peak_gib"] = float(m.group(1))
    return timers


def analyze(rep: Path, top: int):
    by_name, kernels = load(export_sqlite(rep))
    steps = Ranges([iv for name, ivs in by_name.items() if re.fullmatch(r"step_\d+", name) for iv in ivs])
    n_steps = len(steps.intervals)
    phases = {p: Ranges(by_name.get(p, [])) for p in PHASES}
    attn = {a: Ranges(by_name.get(a, [])) for a in ATTN_RANGES}

    phase_time = defaultdict(float)  # summed kernel ms over measured steps
    phase_span = defaultdict(lambda: defaultdict(lambda: [float("inf"), 0]))  # phase -> step -> [first start, last end]
    kern_fwd = defaultdict(lambda: [0.0, 0])  # name -> [ms, count]
    kern_all = defaultdict(lambda: [0.0, 0])
    attn_time = defaultdict(float)
    matmul = {"forward": 0.0, "all": 0.0}
    total = {"forward": 0.0, "all": 0.0}

    for launch, ks, ke, short, dem in kernels:
        step = steps.index_of(launch)
        if step is None:
            continue
        ms = (ke - ks) / 1e6
        is_mm = bool(MATMUL_RE.search(dem or short))
        kern_all[short][0] += ms
        kern_all[short][1] += 1
        total["all"] += ms
        matmul["all"] += ms if is_mm else 0.0
        for p, rng in phases.items():
            if rng.index_of(launch) is not None:
                phase_time[p] += ms
                span = phase_span[p][step]
                span[0], span[1] = min(span[0], ks), max(span[1], ke)
                if p == "forward":
                    kern_fwd[short][0] += ms
                    kern_fwd[short][1] += 1
                    total["forward"] += ms
                    matmul["forward"] += ms if is_mm else 0.0
                    for a, arng in attn.items():
                        if arng.index_of(launch) is not None:
                            attn_time[a] += ms
                break

    timers = parse_log(rep.with_suffix(".log"))
    res = {"name": rep.stem, "n_steps": n_steps, "timers": timers}
    for p in PHASES:
        spans = [e - s for s, e in phase_span[p].values() if e > 0]
        res[p] = {
            "kernel_ms": phase_time[p] / n_steps if n_steps else 0.0,
            "span_ms": (sum(spans) / len(spans) / 1e6) if spans else 0.0,
        }
    res["top_fwd"] = sorted(kern_fwd.items(), key=lambda kv: -kv[1][0])[:top]
    res["top_all"] = sorted(kern_all.items(), key=lambda kv: -kv[1][0])[:top]
    res["fwd_total_ms"] = total["forward"]
    res["all_total_ms"] = total["all"]
    res["matmul_frac"] = {k: (matmul[k] / total[k] if total[k] else 0.0) for k in total}
    res["attn_ms"] = {a: attn_time[a] / n_steps if n_steps else 0.0 for a in ATTN_RANGES}
    return res


def print_report(res, top):
    n = res["n_steps"] or 1
    t = res["timers"]
    print(f"\n## {res['name']}  ({res['n_steps']} measured steps, peak {t.get('peak_gib', '?')} GiB)")
    print("| phase | python timer (ms) | nsys GPU span (ms) | nsys kernel sum (ms) |")
    print("|---|---:|---:|---:|")
    for p in PHASES:
        if p in t:
            print(f"| {p} | {t[p][0]:.2f} ± {t[p][1]:.2f} | {res[p]['span_ms']:.2f} | {res[p]['kernel_ms']:.2f} |")

    for label, key, denom in [("forward", "top_fwd", res["fwd_total_ms"]), ("full step", "top_all", res["all_total_ms"])]:
        if not res[key]:
            continue
        print(f"\nTop {top} kernels, {label} (per step):")
        print("| kernel | ms/step | calls/step | % of GPU time |")
        print("|---|---:|---:|---:|")
        for name, (ms, cnt) in res[key]:
            short = name if len(name) < 70 else name[:67] + "..."
            print(f"| `{short}` | {ms / n:.2f} | {cnt / n:.0f} | {100 * ms / denom:.1f}% |")

    mf = res["matmul_frac"]
    print(f"\nMatmul share of GPU time: forward {100 * mf['forward']:.1f}%, full step {100 * mf['all']:.1f}%")
    a = res["attn_ms"]
    if any(a.values()):
        print(
            "Attention per forward (ms): "
            + ", ".join(f"{k}={v:.2f}" for k, v in a.items())
            + f"  | softmax / (qk+pv) = {a['softmax'] / max(a['qk_matmul'] + a['pv_matmul'], 1e-9):.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nsys_dir", nargs="?", default="nsys")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--only", default="", help="substring filter on profile names")
    args = ap.parse_args()

    for rep in sorted(Path(args.nsys_dir).glob("*.nsys-rep")):
        if args.only and args.only not in rep.stem:
            continue
        print_report(analyze(rep, args.top), args.top)


if __name__ == "__main__":
    main()
