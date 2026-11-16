#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace snn {

// Dense, row-major weight matrix and LIF hyperparameters for one layer.
// W[i * inputs + j] connects presynaptic neuron j to postsynaptic neuron i.
struct Layer {
  std::size_t inputs = 0;
  std::size_t outputs = 0;
  std::vector<float> weights;
  std::vector<float> bias;
  float beta = 0.9F;
  float threshold = 1.0F;

  void validate() const;
};

// After each timestep, voltage is reset to zero if the neuron fires.
// No synaptic delays, refractory periods, or recurrent synapses are modelled.
struct State {
  std::vector<float> voltage;
  explicit State(std::size_t neurons) : voltage(neurons, 0.0F) {}
};

enum class Execution { Dense, Sparse };

// Each input is required to be exactly 0.0f or 1.0f.
// Returns a binary spike vector while modifying State in place.
std::vector<std::uint8_t> step(const Layer& layer,
                               const std::vector<float>& input_spikes,
                               State& state, Execution execution);

// Mean-weight/mean-bias approximation within adjacent output-neuron groups.
// One voltage is retained per group, and its spike is broadcast to members.
// For group_size=1, it is equivalent to the ungrouped layer.
// For larger groups, the model's function changes: accuracy MUST be evaluated.
struct GroupedLayer {
  Layer aggregated;
  std::size_t original_outputs = 0;
  std::size_t group_size = 1;
};

GroupedLayer group_adjacent_mean(const Layer& layer, std::size_t group_size);

std::vector<std::uint8_t> grouped_step(const GroupedLayer& grouped,
                                       const std::vector<float>& input_spikes,
                                       State& group_state, Execution execution);

// Used to confirm that the build actually supports RISC-V vector instructions.
// Transpose once during setup so active inputs have contiguous output weights.
// Layout: input_major[j * layer.outputs + i] = W[i,j].
std::vector<float> prepare_input_major_weights(const Layer& layer);

// Compute only active input connections using RVV operations across outputs.
std::vector<std::uint8_t> step_sparse_rvv(
    const Layer& layer, const std::vector<float>& input_major_weights,
    const std::vector<float>& input_spikes, State& state);

// Like step_sparse_rvv, but accumulate and update each output block in
// registers without allocating/storing/reloading an intermediate current array.
std::vector<std::uint8_t> step_sparse_rvv_fused(
    const Layer& layer, const std::vector<float>& input_major_weights,
    const std::vector<float>& input_spikes, State& state);

bool rvv_available();

// Equivalent dense implementation when compiled with RVV support. On non-RVV
// hosts this throws, rather than silently treating scalar work as vectorised.
std::vector<std::uint8_t> step_rvv(const Layer& layer,
                                   const std::vector<float>& input_spikes,
                                   State& state);

}  // namespace snn
