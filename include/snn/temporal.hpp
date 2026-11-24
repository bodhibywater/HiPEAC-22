#pragma once
#include "snn/kernels.hpp"

namespace snn {

// Mechanism A: a fixed four-timestep, LMUL=1 window (a short final window
// of 1..3 steps is also supported). Input vectors must be binary. Returns
// time-major spikes [time * outputs + neuron]; state survives across windows.
// resident=false is the weight-sharing-only ablation, with state and bias
// explicitly accessed per timestep. No silent scalar fallback.
std::vector<std::uint8_t> step_sparse_rvv_temporal4(
    const Layer& layer, const std::vector<float>& input_major_weights,
    const std::vector<float>* inputs_by_time, std::size_t steps,
    State& state, bool resident);

}  // namespace snn
