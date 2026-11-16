#!/usr/bin/env bash
# Reproduce a small trained-MNIST experiment across single/multi-accumulator
# kernels and old/new weight formats. Requires exported traces and gem5.
set -euo pipefail

repo="${REPO:-$HOME/HiPEAC-22}"
gem5="${GEM5_ROOT:-$HOME/gem5-sim}"
runner="${GEM5_RUNNER:-$repo/gem5/run_o3.py}"
bin="$repo/build-riscv-m5/snn_mnist"
trace_old="$repo/artifacts/mnist_trace_2_om.bin"
trace_new="$repo/artifacts/mnist_trace_2_im.bin"
root="$repo/results/${RESULTS_DIR:-multiacc_initial}"
mkdir -p "$root"

for path in "$gem5/build/RISCV/gem5.opt" "$runner" "$bin" "$trace_old" "$trace_new"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing $path; follow README.md to build/export traces" >&2
    exit 2
  fi
done


csv="$root/summary.csv"
printf 'trace_layout,mode,cycles,instructions,branch_mispredicts,l1d_misses,l2_data_misses,lsq_full_events\n' > "$csv"

# Same trained network and identical deterministic traces; only the kernel
# or the pre-transposition representation changes between the four runs.
for spec in \
  'output_major sparse_rvv_fused' \
  'input_major sparse_rvv_fused' \
  'input_major sparse_rvv_fused_acc2' \
  'input_major sparse_rvv_fused_acc4'; do
  read -r layout mode <<< "$spec"
  if [[ "$layout" == output_major ]]; then trace="$trace_old"; else trace="$trace_new"; fi
  out="$root/${layout}_${mode}"
  mkdir -p "$out"
  echo "gem5: $layout, $mode"

  runner_args=()
  if [[ -n "${SIMD_UNITS:-}" || -n "${SIMD_LATENCY:-}" ]]; then
    if [[ -z "${SIMD_UNITS:-}" || -z "${SIMD_LATENCY:-}" ]]; then
      echo 'Set both SIMD_UNITS and SIMD_LATENCY or neither.' >&2; exit 2
    fi
    runner_args+=(--simd-units "$SIMD_UNITS" --simd-latency "$SIMD_LATENCY")
  fi

  if ! "$gem5/build/RISCV/gem5.opt" \
    --outdir="$out" "$runner" \
    --binary "$bin" "${runner_args[@]}" \
    -- --trace "$trace" --mode "$mode" \
    > "$out/run.log" 2>&1; then
    tail -35 "$out/run.log"
    exit 4
  fi
  if ! grep -q '^reference_check=PASS ' "$out/run.log"; then
    tail -35 "$out/run.log"
    echo "Reference failure; this run is not valid." >&2
    exit 5
  fi

  # Restrict everything to the first gem5 ROI section, not the exit dump.
  stats="$out/stats.txt"
  if [[ ! -s "$stats" ]]; then echo "Missing $stats" >&2; exit 6; fi
  extract() {
    awk -v key="$1" '
      /Begin Simulation Statistics/ {section++;next}
      /End Simulation Statistics/ {if(section==1) exit}
      section==1 && $1==key {print $2;exit}
    ' "$stats"
  }
  cycles="$(extract board.processor.cores.core.numCycles)"
  instr="$(extract simInsts)"
  branches="$(extract board.processor.cores.core.commit.branchMispredicts)"
  l1d="$(extract 'board.cache_hierarchy.l1d-cache-0.overallMisses::total')"
  l2d="$(extract 'board.cache_hierarchy.l2-cache-0.demandMisses::processor.cores.core.data')"
  lsq="$(extract board.processor.cores.core.iew.lsqFullEvents)"
  if [[ -z "$cycles" || -z "$instr" ]]; then
    echo "Could not parse first ROI in $stats" >&2
    exit 7
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$layout" "$mode" "$cycles" "$instr" "$branches" "$l1d" "$l2d" "$lsq" >> "$csv"
  echo "  cycles=$cycles instructions=$instr"
done

echo
echo "Saved $csv"
cat "$csv"
