# Project Definition

## Research question

Can an SNN runtime dynamically choose neuron-state granularity and
sparse/dense execution to reduce computation and memory traffic while staying
within a controlled accuracy budget?

## Motivation

SNN workloads can vary significantly in spike density.

Sparse execution avoids inactive work but introduces indexing and control
overhead. Dense/vector execution performs more work but may become faster once
activity is sufficiently high.

Separately, membrane-potential state can potentially be shared between groups
of similar neurons, reducing state storage and updates at the cost of some
approximation error.

We want to study these two decisions jointly.

## Execution modes

| State | Sparse | Dense |
| --- | --- | --- |
| Raw | per-neuron state, skip inactive work | per-neuron state, vector kernel |
| Grouped | shared group state, sparse updates | shared group state, vector/block kernel |

The eventual runtime should select between these modes using cheap observable
signals such as:

- spike density
- layer shape
- group coherence / estimated approximation error

## Core experiments

### 1. Representation

Measure how grouped membrane state affects:

- classification accuracy
- state bytes
- neuron-state updates
- memory traffic

Initial group sizes: `1, 2, 4, 8, 16`.

Start with simple adjacent/channel-wise grouping. More complex grouping should
only be added if useful.

### 2. Sparse vs dense execution

Implement equivalent sparse and dense kernels and sweep:

- spike density
- layer dimensions
- raw vs grouped state

Measure:

- cycles / latency
- useful operations
- bytes moved
- cache behaviour where available
- vector utilisation where available

The main result should be a crossover/regime map rather than one universal
sparsity threshold.

### 3. Adaptive routing

Build a deliberately small threshold/lookup router.

Compare it against:

- each static execution mode
- the best single static mode
- an oracle that chooses the cheapest valid mode with hindsight

Measure routing overhead, stability and regret versus the oracle.

## Workloads

Initial plan:

- MNIST or Fashion-MNIST for bring-up
- CIFAR-10 compact SNN as the primary workload
- second model/dataset only after the main experiments are stable

An event-based dataset such as DVS-CIFAR10 is optional.

## RISC-V direction

Start with normal RISC-V scalar/vector execution.

Do not design custom instructions upfront.

Profile first. If a repeated hot primitive is clearly important, evaluate one
or two targeted custom operations against the RVV baseline.

Possible examples include:

