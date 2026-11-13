# HiPEAC-22 — adaptive SNN inference on RISC-V

Research foundation for the HiPEAC SNN/RISC-V proposal. This is an **initial,
reviewable implementation**, not a completed competition result.

## What works now

- Portable C++ reference and sparse LIF kernels with identical dynamics.
- Initial adjacent-group mean-state approximation (with explicit semantics).
- Independent C++ and NumPy correctness tests.
- Explicit RVV dense kernel guarded by `__riscv_vector` (not yet run on RISC-V).
- Native wall-clock benchmarking and metadata-producing parameter sweeps.
- Optional PyTorch MNIST surrogate-gradient training prototype.
- gem5 RISC-V SE configuration **draft** (not locally executed).

## Host quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python -m pip install numpy pytest
python -m pytest -q python/test_reference.py
./build/snn_bench --inputs 128 --outputs 128 --steps 64 --density 0.1 --mode sparse
python scripts/sweep.py --bench build/snn_bench --output results/first_sweep.csv \
  --shapes 128x128 --group 1 2 --density 0.01 0.1 0.5 --repeats 5
```

The command writes **native wall-clock time**, not gem5 cycles. Do not use it
as a processor architecture result.

## Optional SNN training

```bash
python -m pip install torch torchvision
python -m python.train_mnist --epochs 2 --steps 16 --seed 42
```

This downloads MNIST. The training model uses Bernoulli rate coding, hard
spikes with sigmoid-surrogate backward gradients, and reset-to-zero LIF.
Training hasn't been validated for useful test accuracy in CI; record the
result you actually observe rather than assuming a target score.

## RISC-V / gem5 bring-up

The optional RVV path is in `src/kernels.cpp` and uses RVV 1.0 intrinsics. It
must be compiled for `rv64gcv` using a RISC-V C++ cross compiler and linked
against a matching RISC-V userspace. Cross-compiling **and running** the test
suite on a RISC-V environment is a necessary follow-up.

`gem5/run_se.py` is a starter configuration targeting the gem5 v25.1 stdlib
API. It has not been executed here because no gem5 binary / RISC-V sysroot is
available. Avoid presenting its results until verified end-to-end. Record the
exact gem5 release/commit and `m5out/config.ini` to confirm RVV/VLEN settings.

**Scientific protocol:** [docs/experimental_protocol.md](docs/experimental_protocol.md)

**Project definition:** [docs/project.md](docs/project.md)  
**Original roadmap:** [docs/roadmap.md](docs/roadmap.md)

## Next research milestones

1. Cross-compile and validate RVV results against scalar outputs in gem5 and,
   if available, on the physical RISC-V board.
2. Export a trained SNN's real weight matrices/spike traces; compare the C++
   kernels against PyTorch step-by-step, including grouped-state accuracy.
3. Add gem5 ROI markers and verify performance-stat extraction.
4. Only then generate regime maps and train/calibrate the adaptive router;
   compare against best static and hindsight-oracle baselines.

No empirical speedups, accuracy guarantees, or energy savings are claimed yet.
