# Mechanism A: fixed temporal execution (prototype)

Base: main at 3db62c51 (22 November 2026). No changes to the existing
kernel implementations or default seven-mode handoff. New modes:

- `sparse_rvv_temporal4_weights`: four timesteps, shared weight loads,
  independent currents, per-timestep membrane and bias memory operations.
- `sparse_rvv_temporal4`: same scheduling and weight sharing, with membrane
  and bias kept in vector registers across LIF updates.

Both use e32m1 (the existing vector width). No B/LMUL sweep, new hardware,
precision changes, frontier policy, or simulator changes. A short final
window is supported. Inference uses actual hidden spikes, not the oracle.
The full network still checks every hidden/output spike and prediction
against PyTorch before timing; all mask building and buffering are timed.

The weight-only variant uses an empty compiler memory barrier to keep its
per-step state accesses from being optimised away. It emits no hardware
fence, but may affect compiler scheduling; check final target assembly.
The A1-to-A2 comparison isolates residency approximately; baseline-to-A1
also changes scheduling and event encoding (not only weight loads).

## Validation on the existing RISC-V Linux/gem5 environment

Use the same compiler/toolchain and gem5 configuration as the baseline.

```bash
cmake -S . -B build-riscv-m5 -DCMAKE_BUILD_TYPE=Release \
  -DSNN_GEM5=ON -DGEM5_ROOT="$HOME/gem5-sim"
cmake --build build-riscv-m5 --target snn_test snn_mnist -j2
qemu-riscv64 -cpu max,vlen=256 ./build-riscv-m5/snn_test
```

Do not run the m5-instrumented snn_mnist binary in QEMU. For a quick
end-to-end smoke test (not final paper results):

```bash
SAMPLES=2 MODES="sparse_rvv_fused sparse_rvv_temporal4_weights sparse_rvv_temporal4" \
  RESULTS_DIR=results/temporal_a_smoke bash bench_kernel_handoff.sh
```

After all modes pass, use a matched 16-image screen:

```bash
SAMPLES=16 MODES="sparse_rvv_fused sparse_rvv_fused_acc2 sparse_rvv_fused_acc4 sparse_rvv_temporal4_weights sparse_rvv_temporal4" \
  RESULTS_DIR=results/temporal_a_16 bash bench_kernel_handoff.sh
```

Compare against the fastest validated conventional mode. Archive summaries,
raw stats/config/logs, trace hash, compiler flags, and committed source SHA.
The handoff regenerates the traces for each sample count; 2/16/64-image
exports are not guaranteed nested subsets. Compare only within a run.
Inspect emitted assembly for actual vector loads, stores and spills before
claiming that the theoretical operation savings are realised.
