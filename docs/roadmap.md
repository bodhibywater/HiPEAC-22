# High-Level Roadmap

Estimates are approximate **team-hours**.

## Phase 0 — Environment and experiment harness

**15–25 hours**

- choose SNN framework and baseline implementation
- establish reproducible experiment structure
- set up RISC-V toolchain
- compare QEMU, gem5 and available hardware
- decide how performance counters / cache / cycle measurements will be taken
- establish C/C++ and RVV kernel path

**Output:** reproducible development and benchmarking environment.

---

## Phase 1 — Baseline SNN

**20–25 hours**

- reproduce a conventional LIF SNN
- sanity-check on MNIST/Fashion-MNIST
- establish CIFAR-10 baseline
- record baseline accuracy, spike rates and state requirements

**Output:** trusted reference implementation.

---

## Phase 2 — Grouped-state characterisation

**30–40 hours**

- implement shared/grouped membrane state
- sweep group size and timestep count
- measure accuracy and state reduction
- identify a useful coherence/error signal

**Output:** representation accuracy/compute frontier.

---

## Phase 3 — Sparse and dense kernels

**30–40 hours**

- implement equivalent sparse and dense execution paths
- support both raw and grouped state
- add correctness tests between kernel variants

**Output:** four comparable execution modes.

---

## Phase 4 — Profiling and regime maps

**30–40 hours**

- sweep spike density and layer shapes
- collect cycle, operation and memory measurements
- identify sparse/dense crossover regions
- explain the causes of the crossovers

**Output:** main crossover/regime maps.

---

## Phase 5 — Adaptive runtime

**20–30 hours**

- implement threshold/lookup routing
- compare against static modes and oracle
- measure routing overhead and stability

**Output:** adaptive four-mode runtime.

---

## Phase 6 — RISC-V/RVV demonstration
