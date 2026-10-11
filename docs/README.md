# HiPEAC SNN RISC-V Project

Research project investigating adaptive execution of Spiking Neural Networks
(SNNs) on RISC-V.

The core idea is to dynamically choose both:

- neuron-state representation: **raw** or **grouped**
- execution strategy: **sparse/event-driven** or **dense/vector**

This gives four execution modes:

1. raw + sparse
2. raw + dense
3. grouped + sparse
4. grouped + dense

See [project.md](project.md) for the research definition and
