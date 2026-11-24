#pragma once
#include "snn/kernels.hpp"

namespace snn {

// Mechanism B: fixed compile-time temporal tiles (1/2/4/8), LMUL 1/2.
// A short final tile is allowed. Each step retains its original input order.
// coalesce=false: separate active lists and weight traversal per timestep.
// resident=false: explicitly load/store membrane and reload bias each step.
// Returns time-major spikes; no scalar fallback.
std::vector<std::uint8_t> step_sparse_rvv_tile(
    const Layer& layer, const std::vector<float>& input_major_weights,
    const std::vector<float>* inputs_by_time, std::size_t steps,
    State& state, std::size_t window, unsigned lmul,
    bool coalesce, bool resident);

}  // namespace snn
