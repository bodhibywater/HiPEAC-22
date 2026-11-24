#!/usr/bin/env bash
# Reproducible gem5 handoff: run all inference kernels on the SAME trained MNIST
# images, timesteps, model and spike traces; fail if spike-level checks differ.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gem5_root="${GEM5_ROOT:-$HOME/gem5-sim}"
gem5="${GEM5_BIN:-$gem5_root/build/RISCV/gem5.opt}"
runner="${GEM5_RUNNER:-$repo/gem5/run_o3.py}"
binary="${SNN_BINARY:-$repo/build-riscv-m5/snn_mnist}"
python="${SNN_PYTHON:-$repo/.venv/bin/python}"
samples="${SAMPLES:-16}"
seed="${SEED:-20261115}"
vlen="${VLEN:-256}"
results="${RESULTS_DIR:-$repo/results/kernel_handoff_${samples}}"
trace_om="$repo/artifacts/kernel_handoff_${samples}_om.bin"
trace_im="$repo/artifacts/kernel_handoff_${samples}_im.bin"
summary="$results/summary.csv"

# Only use the mutually comparable subset in the main plot; the old RVV and
# unfused RVV variants remain available by overriding MODES for diagnostics.
read -r -a modes <<< "${MODES:-dense sparse sparse_scalar_im dense_rvv_fused sparse_rvv_fused sparse_rvv_fused_acc2 sparse_rvv_fused_acc4}"

# Fail before generating traces if any infrastructure prerequisite is absent.
for path in "$gem5" "$runner" "$binary" "$python" "$repo/artifacts/mnist_snn.pt"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing prerequisite: $path" >&2
    exit 2
  fi
done
mkdir -p "$results" "$repo/artifacts"
cd "$repo"

# Export identical samples and RNG-driven input spikes in both weight layouts.
# The two formats only differ in how trained weight matrices are stored.
"$python" -m python.export_mnist_trace \
  --checkpoint artifacts/mnist_snn.pt --samples "$samples" --seed "$seed" \
  --output "$trace_om" > "$results/export_om.json"
"$python" -m python.export_mnist_trace \
  --checkpoint artifacts/mnist_snn.pt --samples "$samples" --seed "$seed" \
  --input-major --output "$trace_im" > "$results/export_im.json"

# Parse only the FIRST statistics block bounded by m5 ROI markers.
read_roi_stat() {
  local file="$1" key="$2"
  awk -v key="$key" '
    /Begin Simulation Statistics/ { section++; next }
    /End Simulation Statistics/ { if (section == 1) exit }
    section == 1 && $1 == key { print $2; exit }
  ' "$file"
}

revision="$(git rev-parse --short HEAD 2>/dev/null || printf unknown)"
printf 'git_revision,mode,weight_layout,samples,vlen,cycles,instructions,branch_mispredicts,lsq_full_events\n' > "$summary"

# Each kernel gets its own gem5 config.ini, run.log and first-ROI stats.
for mode in "${modes[@]}"; do
  case "$mode" in
    dense|sparse|rvv) trace="$trace_om"; layout=output_major ;;
    sparse_scalar_im|dense_rvv_fused|sparse_rvv|sparse_rvv_fused|sparse_rvv_fused_acc2|sparse_rvv_fused_acc4)
      trace="$trace_im"; layout=input_major ;;
    sparse_rvv_temporal4_weights|sparse_rvv_temporal4)
      trace="$trace_im"; layout=input_major ;;
    *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
  esac

  out="$results/$mode"
  mkdir -p "$out"
  echo "Running mode=$mode layout=$layout samples=$samples"
  if ! "$gem5" --outdir="$out" "$runner" \
    --binary "$binary" --vlen "$vlen" \
    -- --trace "$trace" --mode "$mode" > "$out/run.log" 2>&1; then
    echo "gem5 failed for $mode:" >&2
    tail -35 "$out/run.log" >&2
    exit 3
  fi
  if ! grep -q '^reference_check=PASS ' "$out/run.log"; then
    echo "PyTorch spike-level check failed for $mode:" >&2
    tail -35 "$out/run.log" >&2
    exit 4
  fi
  if [[ ! -s "$out/stats.txt" || ! -s "$out/config.ini" ]]; then
    echo "Missing stats/config for $mode" >&2
    exit 5
  fi

  cycles="$(read_roi_stat "$out/stats.txt" board.processor.cores.core.numCycles)"
  instructions="$(read_roi_stat "$out/stats.txt" simInsts)"
  branch="$(read_roi_stat "$out/stats.txt" board.processor.cores.core.commit.branchMispredicts)"
  lsq="$(read_roi_stat "$out/stats.txt" board.processor.cores.core.iew.lsqFullEvents)"
  if [[ -z "$cycles" || -z "$instructions" ]]; then
    echo "Missing first-ROI cycles/instructions for $mode" >&2
    exit 6
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$revision" "$mode" "$layout" "$samples" "$vlen" \
    "$cycles" "$instructions" "$branch" "$lsq" >> "$summary"
  echo "  PASS: cycles=$cycles instructions=$instructions"
done

echo "Completed all requested modes. Summary: $summary"
cat "$summary"
