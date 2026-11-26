# Temporal A+B evaluation: verified results and initial interpretation

Source: uploaded experiment archive at commit `c3e2d62`, under [results/temporal_study](../results/temporal_study). Analysis uses the archived summary CSVs and spot-checks the first complete ROI block and correctness logs for the 64-image acc4 and winning tile.

## Provenance and controls

All 88 runs across A_2 (7), A_16 (7), B_2 (33), B_16 (33) and B_64 (8) are recorded PASS. All sets report gem5 25.1.0.1, simulator commit `c8222cc67a399bfc01e8658dd14b30d5bfd634f9`, identical simulator SHA256 and model fingerprint. GCC 13.3.0; Release, RV64GCV, LP64D, -O3, -fno-fast-math; FP32 LIF model 784→256→10, 16 steps; O3, 2 GHz, VLEN256, L1I/L1D 32 KiB each, L2 512 KiB. A and B have different program binaries and revisions (e0f2698 and 15c4b2f), but exact matching trace hashes for each 2/16-image pair. All B comparisons below are within the same B executable and workload. 64-image reference correctness is 62/64 classifications (96.875%) with exact hidden/output spike parity; this is subset accuracy, not full MNIST accuracy.

## Headline confirmation (64 images)

| Mode | Cycles | Instructions | Speedup vs acc4 |
|---|---:|---:|---:|
| acc1 | 25,843,071 | 53,474,323 | 0.963 |
| acc2 | 25,753,901 | 53,639,059 | 0.967 |
| acc4 | 24,894,399 | 50,477,805 | 1.000 |
| A1 weights | 13,192,418 | 34,260,478 | 1.887 |
| A2 registers | 13,685,423 | 33,866,692 | 1.819 |
| B 4×2 shared/reg | 10,369,484 | 23,358,641 | 2.401 |
| B 8×2 shared/mem | 10,362,963 | 22,064,611 | 2.402 |
| B 8×2 shared/reg | 10,144,048 | 21,879,444 | 2.454 |

The winning tile saves 59.25% of simulated cycles relative to the best conventional baseline and 23.11% relative to A1. At 16 images the corresponding speedup is 2.399×; at 2 images it is 2.727×. These are independent workload sizes, not repeated random trials or necessarily nested subsets.

## Architectural counters (64 images; absolute counts)

| Counter | acc4 | A1 | A2 | B 8×2 shared/reg | Change vs acc4 |
|---|---:|---:|---:|---:|---:|
| Instructions | 50,477,805 | 34,260,478 | 33,866,692 | 21,879,444 | −56.66% |
| L1D accesses | 16,534,544 | 7,894,794 | 7,349,701 | 5,293,386 | −67.99% |
| L1D misses | 5,735,944 | 1,889,377 | 1,857,778 | 991,584 | −82.71% |
| L2 accesses | 5,452,744 | 1,914,972 | 1,914,565 | 1,144,485 | −79.01% |
| L2 misses | 25,694 | 37,471 | 37,483 | 32,951 | +28.24% |
| Branch mispredictions | 326,439 | 510,358 | 512,439 | 312,112 | −4.39% |
| LSQ full events | 570,297 | 103,215 | 3,156 | 5,698 | −99.00% |

The raw first-ROI stats additionally show rename LQ-full blocking events dropping from 9,405,928 to 190,504 (about 98%). These counts are not directly convertible into saved stall cycles. Conditional prediction errors / predictions are 322,183/11,410,194 (2.82%) and 307,204/8,737,116 (3.52%) respectively: fewer errors in absolute terms does not mean improved prediction rate. I-cache stall cycles increase from 40,419 to 199,099; rename ROB-full events increase from 2,093 to 64,823. This supports a shift in bottlenecks rather than universal improvement in every counter. Memory-access and queue-pressure reductions are consistent with the intended reuse mechanism; do not infer a complete causal cycle breakdown.

## Factorial screen (16 images)

Shared, register-resident tiles (cycles):

| Temporal window | LMUL1 | LMUL2 |
|---|---:|---:|
| 2 | 5,555,797 | 3,935,240 |
| 4 | 3,569,425 | 2,569,297 |
| 8 | 3,198,812 | 2,556,493 |

The matched 24-cell factorial shows coalescing improves cycles in all 12 shared-vs-separate comparisons (23.79–55.58% reduction). Residency has heterogeneous effects: it improves 11 of 12 mem-vs-reg pairs, but worsens window4/LMUL1/shared by 3.56%; original A2 also loses to A1 at all three workload sizes on the team machine. At LMUL2, increasing the shared/resident window from 4 to 8 saves only 0.50% (16 images) and 2.17% (64 images). Larger temporal windows have diminishing returns despite additional analytical reuse. Spills, register pressure, control overhead and instruction scheduling are hypotheses, not established causes. The separate-vs-shared variants also differ in event-list construction; the mem variants include a compiler-only barrier.

## Real trace recurrence (64 images)

| Window | L1 ideal weight-vector load reduction | L2 ideal weight-vector load reduction |
|---|---:|---:|
| 1 | 0% | 0% |
| 2 | 43.20% | 32.47% |
| 4 | 68.94% | 62.00% |
| 8 | 83.51% | 79.96% |
| 16 | 91.42% | 89.67% |

The archived recurrence CSV reproduces the handoff's counts exactly: L1 density 13.3357%, 107,061 spike events; L2 density 28.1059%, 73,678 spike events. These reductions are analytical ideal explicit-load opportunities at LMUL1, not measured memory accesses or cache misses. Window16 is not an implemented tile.

## Paper figures and remaining work

Use the 64-image progression, 16-image full factorial/heatmap, recurrence curve, and normalized architectural-counter chart. Retain negative cases; avoid using older weaker dense RVV as headline baseline. All claims concern simulated cycles in one documented architecture on a single feedforward MNIST network. Do not claim physical-hardware speedups, general recurrent/online applicability, or energy improvement. A complete assembly-level spill analysis remains pending: the archived full disassembly is large (~24 MB for B_64) and the connector did not return its contents. Extract relevant functions locally or obtain a smaller symbol-specific listing before making spill claims. No additional large simulation sweep is required to describe the core study; targeted architecture/workload sensitivity would strengthen generalization if time permits.
