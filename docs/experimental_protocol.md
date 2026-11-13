# Experimental protocol: preregister before collecting paper results

## 1. Definitions and validity

One feedforward LIF layer: for output neuron `i` at timestep `t`,

- `current_i(t) = sum_j W_ij * s_j(t)` where `s_j(t) ∈ {0,1}`;
- `u_i(t) = beta * v_i(t-1) + current_i(t) + bias_i`;
- `s_i(t) = 1[u_i(t) >= threshold]`;
- `v_i(t) = 0` if `s_i(t)=1`, otherwise `v_i(t)=u_i(t)`.

State is reset **at the beginning of each sample**, not between timesteps.
Float32 arithmetic, row-major weights `[out, in]`, no recurrent synapses,
no refractory periods, no multi-spike emission, zero-reset convention.

**Grouping (initial approximation):** adjacent output rows of `W` and `bias`
are averaged into one aggregate neuron per group. Its *single* membrane state
produces a spike broadcast to group members. Last partial group uses its actual
size. This changes both representation and effective synaptic weights; any
claimed savings must report changed computation AND measured accuracy. In
particular it is **not** a proven drop-in replacement for a trained model.
The initial implementation fixes grouping per inference; raw/grouped dynamic
switching is not yet supported, since discarded individual state cannot be
recovered exactly.

## 2. Correctness gates (before interpreting speedup)

- Dense vs sparse must emit the same spikes and preserve state within an
  explicitly reported float32 tolerance; include zero-density, full-density,
  threshold equality, non-multiple group sizes, and multi-timestep histories.
- `group_size=1` must exactly reproduce the raw semantics.
- RVV output must be checked against a scalar reference on **RISC-V hardware or
  a functioning RVV simulator**; compiler acceptance alone is insufficient.
- End-to-end classification accuracy must be evaluated on a held-out dataset
  before choosing acceptable grouped modes.
- Do not tune thresholds using the test set. Fix classifier weights across
  compared inference modes unless retraining is explicitly labelled.

## 3. Benchmark design

- Pair runs: same weights, input spike traces, seed and timestep budget for
  each sparse/dense comparison. Record **realised** density.
- Time the complete per-timestep computation (index creation, synaptic work,
  state updates and firing); distinguish it from pre-aggregation of weights.
- Use native benchmark for bring-up only; native `elapsed_us` is **NOT** a
  RISC-V cycle measurement and must never be labelled as a gem5 result.
- In gem5, record and pin simulator commit, exact binary checksum, toolchain,
  build flags, CPU model, ISA extensions, VLEN, cache hierarchy, memory model,
  CPU clock, input parameters and complete `stats.txt`/`config.ini`.
- `simTicks` and instruction counts from a full program include setup and
  teardown. For publishable *kernel-only* gem5 cycles, add and verify gem5
  statistics reset/dump ROI markers (or an equivalent validated approach),
  rather than treating total program cycles as kernel cycles.
- TimingSimpleCPU is useful for basic functional/timing bring-up, but is not
  calibrated to any specific physical CPU. Do not claim hardware-realistic
  RVV speedup without sensitivity analysis and/or physical-board validation.
- Sweep densities and shapes in advance, randomise native experiment order,
  use multiple timing trials, retain **all** raw samples, and report spread
  (e.g. bootstrap confidence intervals across independent runs).
- Include warmup, avoid noisy simultaneous jobs on a shared CPU, control
  affinity/governor when possible and do not cherry-pick fastest samples.
- Report overhead of routing and state-format conversion if added.

## 4. Separate metrics

- **Algorithmic:** useful synaptic additions, total scalar/vector operations,
  neuron-state updates, parameter bytes and membrane-state bytes.
- **Native runtime:** wall-clock microseconds per workload, hardware counters
  only where explicitly verified (and labelled with processor and kernel).
- **gem5 architectural:** simulated cycles, system-specific cache/memory
  counters, vector instruction statistics where reliable, and configuration.
- **Accuracy:** top-1 accuracy change in percentage points, with confidence
  intervals or repeated seeds where feasible. Coherence cannot be assumed to
  predict *classification* error without calibration and held-out testing.

No latency or energy improvements are asserted by this starter codebase.
"""
