# Kernel handoff for microarchitectural experiments (22 November 2026)

## Current state and validation boundary

The baseline implementations are committed, but the **new** input-major
scalar control and dense-fused RVV control have **not yet passed RISC-V build,
QEMU vector tests, or gem5/PyTorch spike-level verification** in the author's
environment. Do not use their cycle numbers in the report until these checks
pass. This handoff deliberately retains all original modes for comparisons.

## Main kernels

| Mode | Sparse inputs | RVV | Layout | Fused LIF | Role |
| --- | --- | --- | --- | --- | --- |
| `dense` | No | No | Output-major | No | Original scalar control |
| `sparse` | Yes | No | Output-major | No | Original sparse scalar baseline |
| `sparse_scalar_im` | Yes | No | Input-major | Yes | Layout-matched sparse scalar baseline (NEW) |
| `dense_rvv_fused` | No | Yes | Input-major | Yes | Fair output-parallel dense vector control (NEW) |
| `sparse_rvv_fused` | Yes | Yes | Input-major | Yes | Main sparse vector kernel, acc1 |
| `sparse_rvv_fused_acc2` | Yes | Yes | Input-major | Yes | Two independent vector accumulators |
| `sparse_rvv_fused_acc4` | Yes | Yes | Input-major | Yes | Four independent vector accumulators |

**Legacy diagnostic modes:** `rvv` uses output-major weights, vector
dot products, reductions and separate LIF; `sparse_rvv` uses input-major
weights but stores an intermediate current array. They remain supported, but
should not be described as optimised primary baselines.

The fairer factorial comparison is `sparse_scalar_im` versus
`sparse_rvv_fused` (vectorisation, holding sparse layout and fusion fixed)
and `dense_rvv_fused` versus `sparse_rvv_fused` (sparsity, holding
vector layout and fusion fixed). A dense scalar input-major fused control
has **not** been implemented, so the full 2-by-2 interaction is not isolated.

## One-time setup (RISC-V environment)

From the repository root, with your previously working RISC-V cross-compiler,
gem5 v25.1.0.1 build, libm5.a, Python venv and checkpoint:

```bash
git pull --ff-only
cmake -S . -B build-riscv-m5 -DCMAKE_BUILD_TYPE=Release -DSNN_GEM5=ON -DGEM5_ROOT="$HOME/gem5-sim"
cmake --build build-riscv-m5 --target snn_test snn_mnist snn_bench -j2
qemu-riscv64 -cpu max,vlen=256 ./build-riscv-m5/snn_test
```

**Do not run `snn_mnist` under QEMU when `SNN_GEM5=ON`**:
its ROI m5 instructions are intended for gem5 and will fault under QEMU.

## Same-workload gem5 correctness and initial performance screen

```bash
bash bench_kernel_handoff.sh
```

This exports 16 identical MNIST samples and spike traces in both weight
layouts, runs the seven primary kernels under a single default gem5 model,
requires `reference_check=PASS` for each, and saves
`results/kernel_handoff_16/summary.csv` plus per-mode `config.ini`,
`stats.txt`, and `run.log`. It includes index extraction, activation
conversion and allocations inside the ROI; checkpoint loading and weight
transposition/export are outside it.

For a quick validation use `SAMPLES=2 bash bench_kernel_handoff.sh`.
For more representative results use
`SAMPLES=64 RESULTS_DIR=results/kernel_handoff_64 bash bench_kernel_handoff.sh`.
For old diagnostics, `MODES="rvv sparse_rvv" bash bench_kernel_handoff.sh`.

## What teammates can now investigate

**Memory/cache:** L1D/L2 misses, LSQ occupancy, memory latency and layer
weight footprints. The first-layer FP32 matrix is 784×256 = 784 KiB,
larger than our default 512 KiB L2. Compare cache capacity and access order.
Changing cache size alone is a hardware sensitivity experiment, not a
software-level cache optimisation.

**Branches:** active-index extraction and its predictor behaviour under
different spike densities; log mispredictions *and cycles*, and distinguish
a branch predictor sensitivity sweep from an actual branchless implementation.

**Execution:** acc1/acc2/acc4 across SIMD operation latencies and available
pipelined execution units. The prior 1-cycle-vs-4-cycle experiment modified
the generic SIMD unit; it is NOT a calibrated hardware model.

**Ablations to hand off later:** main 7-mode comparison at 16/64+ images,
scalar-vs-RVV with identical input-major weights, dense-vs-sparse fused RVV,
acc1/acc2/acc4, per-layer runs, and spike-density sweeps. Keep traces,
compiler settings and gem5 reference configuration fixed within comparisons.

## Experiment integrity

- Always archive `config.ini`, `stats.txt`, `run.log`, trace metadata,
  source revision and compiler flags.
- Record the **first** ROI stats block, not the exit dump.
- Full MNIST accuracy of the checkpoint was 95.09% over 10k test images;
  a 16-image subset's accuracy is not an estimate of that figure.
- Exact PyTorch spike checks are mandatory on each inference mode.
- Any added cache-/branch-optimised variant should get a NEW mode and a
  separate correctness comparison rather than silently replacing a baseline.
- There is no claim of genuine hardware calibration or of sparsity novelty.
