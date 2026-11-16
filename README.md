# End-to-end MNIST SNN integration

Adds two new files and a CMake target without touching existing kernel implementations:

- `python/export_mnist_trace.py`: loads `artifacts/mnist_snn.pt` and held-out MNIST images, generates deterministic binary input spikes, and serialises trained weights plus **all per-timestep hidden/output spike outputs** from PyTorch.
- `src/mnist_inference.cpp`: executes the full two-layer model for each image and timestep, verifies spike sequences/predictions against the exported PyTorch reference **outside** the measured region, then measures the entire network using `m5_reset_stats` / `m5_dump_stats`.
- `setup_mnist_integration.py`: adds `snn_mnist` to existing CMake only once.

## From HiPEAC-22 repo root

```bash
python3 setup_mnist_integration.py
source .venv/bin/activate
python -m python.export_mnist_trace --checkpoint artifacts/mnist_snn.pt --samples 64 --output artifacts/mnist_trace.bin
cmake -S . -B build-riscv-m5 -DSNN_GEM5=ON -DGEM5_ROOT="$HOME/gem5-sim"
cmake --build build-riscv-m5 --target snn_mnist -j2
```

In `~/gem5-sim` run:

```bash
./build/RISCV/gem5.opt \
  --outdir="$HOME/HiPEAC-22/results/o3_mnist_fused_64" \
  "$HOME/HiPEAC-22/gem5/run_o3.py" \
  --binary "$HOME/HiPEAC-22/build-riscv-m5/snn_mnist" \
  -- --trace "$HOME/HiPEAC-22/artifacts/mnist_trace.bin" \
  --mode sparse_rvv_fused
```

Run another mode (`sparse`, `rvv`, etc) with another `--outdir`. `--first-mode` and `--second-mode` allow different kernels per layer.

### Important controls

- Verification is strict and performed before gem5 ROI. Do not report speedups if reference_check is not PASS.
- Input spikes and trained weights are fixed. No dataset I/O or weight transposition inside the ROI.
- ROI measures complete two-layer inference over the selected sample count and timesteps, including activation conversion, membrane resets between images, allocations and classification counts.
- The exported reference reports accuracy *on this subset*, not the full 10k-image test accuracy.
- Simulated cycles reflect the configured O3CPU model, not measured RISC-V hardware throughput.
- Transposition happens once per model load; for cold-start latency/short inference sessions, account for it separately.
- If a PyTorch-vs-C++ mismatch appears, diagnose numerical ordering/threshold behaviour rather than disabling correctness checks.
