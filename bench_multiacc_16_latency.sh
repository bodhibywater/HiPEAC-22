#!/usr/bin/env bash
# Compare 1 vs 2 RVV accumulators on identical 16-image MNIST traces.
# Only generic SIMD execution latency changes between the two gem5 models.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gem5_root="${GEM5_ROOT:-$HOME/gem5-sim}"
gem5="$gem5_root/build/RISCV/gem5.opt"
runner="$repo/run_o3_vector_sensitivity.py"
binary="$repo/build-riscv-m5/snn_mnist"
python="$repo/.venv/bin/python"
trace="$repo/artifacts/mnist_trace_16_im.bin"
results="$repo/results/multiacc_16_latency"
summary="$results/summary.csv"

# Fail before any simulations if a required executable or file is missing.
for f in "$gem5" "$runner" "$binary" "$python" "$repo/artifacts/mnist_snn.pt"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    exit 2
  fi
done
mkdir -p "$results" "$repo/artifacts"
cd "$repo"

# Export the same trained network and deterministic input spike traces once.
"$python" -m python.export_mnist_trace \
  --checkpoint artifacts/mnist_snn.pt \
  --samples 16 --seed 20261115 --input-major \
  --output "$trace"

# Read exactly the first gem5 ROI statistics section, not the exit dump.
read_roi_stat() {
  local file="$1" key="$2"
  awk -v key="$key" '
    /Begin Simulation Statistics/ { section++; next }
    /End Simulation Statistics/ { if (section == 1) exit }
    section == 1 && $1 == key { print $2; exit }
  ' "$file"
}

printf 'simd_units,simd_latency,mode,cycles,instructions,branch_mispredicts,lsq_full_events\n' > "$summary"

# Vary arithmetic latency alone: keep four generic SIMD units in both models.
for latency in 1 4; do
  for mode in sparse_rvv_fused sparse_rvv_fused_acc2; do
    name="units4_lat${latency}_${mode}"
    output_dir="$results/$name"
    mkdir -p "$output_dir"
    echo "Running: 16 MNIST images, 4 SIMD units, latency=$latency, mode=$mode"

    if ! "$gem5" --outdir="$output_dir" "$runner" \
      --binary "$binary" --vlen 256 \
      --simd-units 4 --simd-latency "$latency" \
      -- --trace "$trace" --mode "$mode" \
      > "$output_dir/run.log" 2>&1; then
      echo "gem5 failed: $name" >&2
      tail -40 "$output_dir/run.log" >&2
      exit 3
    fi
    if ! grep -q '^reference_check=PASS ' "$output_dir/run.log"; then
      echo "Spike-level correctness check failed: $name" >&2
      tail -40 "$output_dir/run.log" >&2
      exit 4
    fi
    if ! grep -q "SIMD_MODEL units=4 operation_latency=$latency cycles" "$output_dir/run.log"; then
      echo "WARNING: could not confirm model settings in $output_dir/run.log" >&2
      exit 5
    fi

    stats="$output_dir/stats.txt"
    if [[ ! -s "$stats" ]]; then
      echo "Missing gem5 stats: $stats" >&2
      exit 6
    fi
    cycles="$(read_roi_stat "$stats" board.processor.cores.core.numCycles)"
    insts="$(read_roi_stat "$stats" simInsts)"
    branches="$(read_roi_stat "$stats" board.processor.cores.core.commit.branchMispredicts)"
    lsq="$(read_roi_stat "$stats" board.processor.cores.core.iew.lsqFullEvents)"
    if [[ -z "$cycles" || -z "$insts" ]]; then
      echo "Missing ROI cycles/instructions for $name" >&2
      exit 7
    fi
    printf '4,%s,%s,%s,%s,%s,%s\n' \
      "$latency" "$mode" "$cycles" "$insts" "$branches" "$lsq" >> "$summary"
    echo "  Correctness PASS; cycles=$cycles; instructions=$insts"
  done
done

# Report effect of the accumulator change within each timing configuration.
"$python" - "$summary" <<'PY'
"""Print ROI results and per-model speedups without mixing simulation models."""
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

print("\nResults:")
for row in rows:
    print(f"  latency={row['simd_latency']}, mode={row['mode']}: {int(row['cycles']):,} cycles")
for latency in ("1", "4"):
    results = {row["mode"]: int(row["cycles"]) for row in rows if row["simd_latency"] == latency}
    baseline = results["sparse_rvv_fused"]
    candidate = results["sparse_rvv_fused_acc2"]
    print(f"  Latency {latency}: acc2 improvement = {(baseline - candidate) / baseline * 100:.2f}%")
print(f"\nCSV: {sys.argv[1]}")
PY
