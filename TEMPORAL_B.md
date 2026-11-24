# Mechanism B: resource-aware temporal and output tiling

Extends the frozen A patch (four timesteps, LMUL=1) without replacing its
kernels or any conventional baseline. This branch adds one configurable
kernel with fixed compiled variants: windows 1/2/4/8 and LMUL 1/2 (at 256-bit
VLEN, up to 8/16 FP32 outputs per vector). LMUL4 is deliberately deferred.
No new hardware, frontier policy, precision changes or automatic selector.

New mode naming: `sparse_rvv_tile4_m2_shared_reg`, for example:

- `tile4`: four timesteps (1/2/4/8 supported).
- `m2`: vector register grouping 2 (1/2 supported).
- `shared` or `separate`: coalesced events/weight loads or separate active
  lists and weight traversal for each timestep.
- `reg` or `mem`: retain membrane and bias across LIF updates or explicitly
  access them at each timestep.

The new 4x1 variants are separate implementations from A; compare them with
A1/A2 to detect implementation differences before attributing tile effects.
Separate-list controls also change event-list construction/traversal, so
shared-vs-separate captures the whole coalescing mechanism, not a pure load
instruction substitution. `mem` uses the same compiler-only barrier as A1.
The 1-step controls use the windowed driver and are not replacements for the
original conventional baseline. All variants preserve ascending summation
order and chronological LIF updates. Hidden inputs are computed, not oracle.
Mixed layer modes are allowed if their nonzero window sizes match; this is
not a measurement of isolated layer cycles.

## Run on the team's verified RISC-V Linux environment

Configure the existing working CMake build with SNN_GEM5=ON and the same
compiler/flags as A. Activate the existing Python environment.

```bash
python run_temporal_study.py --suite full --list
python run_temporal_study.py --suite full --samples 2 --out results/team_ab_2
python run_temporal_study.py --suite full --samples 16 --out results/team_ab_16
```

Prefer `--trace results/team_a_16/trace_im.bin` instead of a new export for
matched cross-branch experiments. `--suite a` runs 7 A/control modes;
`--suite core` runs 23 modes; `--suite full` runs 33 (24 factorial cells,
2 one-step width controls, 5 conventional baselines, 2 original A modes).
`--suite custom --modes ...` allows selected 64-image confirmation. The
runner rebuilds the configured CMake targets and runs QEMU by default; it
archives raw configs, stats, logs, trace, hashes, flags and disassembly if
available. Missing optional counters stay blank. Failed modes get no speedup.
The original handoff shell script still supports the new mode names.

This is a prototype, not a validated B result. Verify all target tests and
exact PyTorch hidden/output spike traces, inspect GCC/Clang output for
actual vector width and spills, and compare only consistent simulator
versions/configurations. The full factorial is about end-to-end execution;
per-layer instrumentation and synthetic recurrence controls are later work.
