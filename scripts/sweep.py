"""Automate native benchmark runs and preserve raw samples with provenance.

This script does NOT generate gem5 cycle results. It launches a native binary,
randomises configuration ordering, and records one row per timed repeat.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Identify the exact executable, not just its source revision."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    """Run a reproducible, paired density/shape/representation sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=Path("build/snn_bench"))
    parser.add_argument("--output", type=Path, default=Path("results/native_sweep.csv"))
    parser.add_argument("--density", type=float, nargs="+", default=[0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--shapes", nargs="+", default=["128x128", "256x256"])
    parser.add_argument("--group", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--modes", nargs="+", default=["dense", "sparse"])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    benchmark = args.bench.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        parser.error("Refusing to overwrite existing experimental results")
    jobs = []
    for shape in args.shapes:
        n_in, n_out = (int(x) for x in shape.split("x"))
        for density in args.density:
            for group in args.group:
                for mode in args.modes:
                    jobs.append((n_in, n_out, density, group, mode))
    random.Random(args.seed).shuffle(jobs)
    records = []
    for index, (n_in, n_out, density, group, mode) in enumerate(jobs, 1):
        cmd = [str(benchmark), "--inputs", str(n_in), "--outputs", str(n_out),
               "--density", str(density), "--group", str(group),
               "--steps", str(args.steps), "--seed", str(args.seed),
               "--repeats", str(args.repeats), "--warmup", str(args.warmup),
               "--mode", mode]
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        if len(rows) != args.repeats:
            raise RuntimeError(f"Expected {args.repeats} timed trials, got {len(rows)}")
        records.extend(rows)
        print(f"[{index}/{len(jobs)}] {shape} d={density:g} g={group} {mode}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "measurement": "native_wall_clock_us; NOT gem5 cycles",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": "scripts/sweep.py",
        "arguments": vars(args) | {"bench": str(benchmark), "output": str(output)},
        "host": {"platform": platform.platform(), "processor": platform.processor(),
                 "python": platform.python_version(), "machine": platform.machine()},
        "benchmark_sha256": sha256_file(benchmark),
        "experiment_count": len(jobs),
        "raw_sample_count": len(records),
        "environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "caveats": ["No CPU affinity or frequency governor enforcement",
                    "Preprocessing of grouped weights excluded from timed region",
                    "Full timestep includes spike-index construction, state updates, and allocations"],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    grouped = {}
    for row in records:
        key = (row["inputs"], row["outputs"], row["group"], row["requested_density"], row["mode"])
        grouped.setdefault(key, []).append(float(row["elapsed_us"]))
    print("Sample medians (us), not scientific conclusions:")
    for key, values in list(sorted(grouped.items()))[:12]:
        print(key, round(statistics.median(values), 3))
    print(f"Raw data: {output}")


if __name__ == "__main__":
    main()
