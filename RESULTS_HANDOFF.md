# HiPEAC gem5 results: team handoff (24 November 2026)

> **Provenance:** The numbers below are transcribed from terminal output shared in the project chat. This document is NOT a replacement for raw `stats.txt`, `config.ini`, logs, exact binary, and input trace. The `results/` directory is gitignored, so raw simulation artifacts remain on the machines where the runs were made unless separately shared. The runs below use different inputs/settings across groups; **do not compare cycle counts across groups as if they were one controlled sweep.**

See [KERNEL_HANDOFF.md](KERNEL_HANDOFF.md) for kernel definitions, workloads and reproducibility instructions.

## A. Latest seven-kernel integration screen — 2 MNIST images, 16 timesteps/image

**Source:** `SAMPLES=2 bash bench_kernel_handoff.sh` terminal output, revision `3db62c5`; default gem5 O3/RVV VLEN=256 runner; all seven modes reported `reference_check=PASS`. Original/output-major and new/input-major modes used the same sample selection and fixed spike traces, with different weight storage layouts.

| Mode | Cycles | Instructions | Branch mispredicts | LSQ full events |
|---|---:|---:|---:|---:|
| `dense` | 33,222,532 | 39,629,825 | 15,119 | 90,814 |
| `sparse` | 5,643,539 | 8,898,533 | 18,938 | 453,885 |
| `sparse_scalar_im` | 3,962,990 | 9,796,643 | 18,659 | 15,430 |
| `dense_rvv_fused` | 13,372,068 | 10,150,847 | 6,087 | 39,892 |
| `sparse_rvv_fused` | 890,548 | 1,800,099 | 10,072 | 24,179 |
| `sparse_rvv_fused_acc2` | 838,314 | 1,804,355 | 10,352 | 14,428 |
| `sparse_rvv_fused_acc4` | 825,216 | 1,690,967 | 10,454 | 17,379 |

Expected local artifacts: `results/kernel_handoff_2/summary.csv` and per-mode directories containing `stats.txt`, `config.ini` and `run.log`. **This was a two-image correctness/integration screen, not a final performance evaluation.**

## B. Accumulator vs generic SIMD latency — 16 MNIST images

**Source:** `bench_multiacc_16_latency.sh` terminal output after fixing the functional-unit model on the O3 instruction queue. All four runs passed strict spike-level reference checks; input-major weights, same 16-image trace, four generic SIMD units.

| SIMD op latency assumption | Kernel | Cycles | Instructions |
|---|---|---:|---:|
| 1 cycle | `sparse_rvv_fused` | 6,614,269 | 13,129,111 |
| 1 cycle | `sparse_rvv_fused_acc2` | 6,369,282 | 13,215,189 |
| 4 cycles | `sparse_rvv_fused` | 6,881,082 | 13,129,111 |
| 4 cycles | `sparse_rvv_fused_acc2` | 6,484,568 | 13,215,189 |

Acc2 reduced cycles by **3.70%** at the default latency and **5.76%** when all generic SIMD arithmetic op classes were assigned four cycles. **This changes all generic SIMD op classes, not solely addition, and does not reproduce a calibrated physical core.**

Expected local artifacts: `results/multiacc_16_latency/summary.csv` and `results/multiacc_16_latency/units4_lat*_*/`.

## C. Earlier original 16-image baseline (separate trace/configuration)

| Kernel | Cycles | Instructions | Branch mispredicts | L1D misses | L2 data-demand misses | LSQ full |
|---|---:|---:|---:|---:|---:|---:|
| Sparse scalar | 43,715,571 | 62,582,118 | 152,299 | 4,927,611 | 221,764 | 2,352,172 |
| Original dense RVV (dot-product + reductions) | 52,206,278 | 107,365,503 | 106,710 | 8,492,010 | 63,476 | 26,980 |
| Sparse fused RVV | 6,443,517 | 13,086,907 | 80,391 | 1,411,189 | 4,347 | 192,796 |

These numbers were logged before the new fair `dense_rvv_fused` control was implemented. **Do not claim the difference between original dense RVV and sparse RVV comes from sparsity alone.** The original dense RVV also differs in layout, reduction direction and fusion.

## D. Earlier two-image layout/accumulator screen

| Weight trace layout | Mode | Cycles | Instructions | Branch mispredicts | L1D misses | L2 data-demand misses | LSQ full |
|---|---|---:|---:|---:|---:|---:|---:|
| Output-major | `sparse_rvv_fused` | 891,095 | 1,793,598 | 9,962 | 202,389 | 52 | 24,239 |
| Input-major | `sparse_rvv_fused` | 890,030 | 1,793,891 | 9,956 | 202,095 | 43 | 25,305 |
| Input-major | `sparse_rvv_fused_acc2` | 855,885 | 1,804,227 | 9,992 | 203,912 | 8 | 14,204 |
| Input-major | `sparse_rvv_fused_acc4` | 879,408 | 1,690,839 | 10,257 | 202,104 | 3 | 14,429 |

Expected default script output: `results/multiacc_initial/summary.csv` (unless a custom `RESULTS_DIR` was used). Pre-transposition primarily reduces setup work and duplicated weight memory; that setup is outside measured ROI.

## E. Synthetic single-layer spike-density sweep

An earlier 128×128 single-layer, 64-timestep run swept requested spike densities **0%, 1%, 10%, 30%, 50%, 75%, 100%**. In the logs reported in chat, fused sparse RVV outperformed original dense RVV for every sampled density, including at 100% (615,895 versus 664,014 cycles, respectively). The full per-density raw measurements were **not provided in this handoff** and should be recovered from the original local result folders before making a detailed regime/crossover claim. Moreover, the dense RVV comparator was the older, less-optimised kernel.

## What teammates should do

1. **Obtain the original `results/` artifacts from the machine that ran gem5**, not just this summary. Each run's `stats.txt` and `config.ini` supports deeper investigation of miss types, pipeline stalls, branch prediction and the precise simulation model.
2. For the current main comparison use `results/kernel_handoff_2/`, and rerun `SAMPLES=16 bash bench_kernel_handoff.sh` or larger if time permits; two images are insufficient for a final paper result.
3. For latency hiding, use `results/multiacc_16_latency/`; verify `core.instQueues[0].fuPool` is the pool being changed before drawing conclusions.
4. Branch/cache studies should start from an explicitly recorded kernel revision and keep inputs, cache layout and timing model constant for software comparisons.
5. Keep historical experiments separate from the final freeze; `docs/project.md` describes an early project proposal involving dynamic grouped neurons and is **not** the current locked research direction.

## Local raw-results transfer

On the machine with `~/HiPEAC-22/results/`, this GNU tar command packages the key analysis files across all run directories while avoiding unnecessary simulator binaries and traces:

```bash
cd ~/HiPEAC-22
find results -type f \( -name stats.txt -o -name config.ini -o -name run.log -o -name '*.csv' \) -print0 |
  tar --null -czf "$HOME/hipeac-gem5-results.tar.gz" -T -
```

Share that archive directly (or publish as a clearly labelled raw-results attachment), along with the code revision and input-trace metadata. **The archive has not been created or uploaded by this document.**
