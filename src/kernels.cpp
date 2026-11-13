#include "snn/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace snn {
namespace {

// Reject malformed binary inputs rather than benchmarking undefined workloads.
void validate_input(const Layer& layer, const std::vector<float>& spikes,
                    const State& state) {
  if (spikes.size() != layer.inputs || state.voltage.size() != layer.outputs ||
      layer.inputs == 0 || layer.outputs == 0 ||
      layer.outputs > std::numeric_limits<std::size_t>::max() / layer.inputs ||
      layer.weights.size() != layer.inputs * layer.outputs ||
      layer.bias.size() != layer.outputs) {
    throw std::invalid_argument("SNN input/state shape mismatch");
  }
  for (float spike : spikes) {
    if (spike != 0.0F && spike != 1.0F) {
      throw std::invalid_argument("SNN inputs must contain binary spikes");
    }
  }
}

// Apply leak, integrate the current, emit a binary spike, and reset after firing.
std::vector<std::uint8_t> advance(const Layer& layer,
                                  const std::vector<float>& current,
                                  State& state) {
  std::vector<std::uint8_t> output(layer.outputs);
  for (std::size_t i = 0; i < layer.outputs; ++i) {
    const float voltage = layer.beta * state.voltage[i] + current[i] + layer.bias[i];
    const bool fire = voltage >= layer.threshold;
    output[i] = static_cast<std::uint8_t>(fire);
    state.voltage[i] = fire ? 0.0F : voltage;
  }
  return output;
}

// The dense version processes every weight, including zero-valued spikes.
std::vector<float> dense_current(const Layer& layer,
                                 const std::vector<float>& spikes) {
  std::vector<float> current(layer.outputs, 0.0F);
  for (std::size_t i = 0; i < layer.outputs; ++i) {
    float sum = 0.0F;
    for (std::size_t j = 0; j < layer.inputs; ++j) {
      sum += layer.weights[i * layer.inputs + j] * spikes[j];
    }
    current[i] = sum;
  }
  return current;
}

// Sparse version first identifies active input indices, then visits only their
// weights. Index extraction is INCLUDED in benchmark timing.
std::vector<float> sparse_current(const Layer& layer,
                                  const std::vector<float>& spikes) {
  std::vector<std::size_t> active;
  active.reserve(layer.inputs);
  for (std::size_t j = 0; j < layer.inputs; ++j) {
    if (spikes[j] == 1.0F) active.push_back(j);
  }
  std::vector<float> current(layer.outputs, 0.0F);
  for (std::size_t i = 0; i < layer.outputs; ++i) {
    float sum = 0.0F;
    for (std::size_t j : active) {
      sum += layer.weights[i * layer.inputs + j];
    }
    current[i] = sum;
  }
  return current;
}

#if defined(__riscv_vector)
// Explicit RVV 1.0 vector intrinsic path. The reduction changes floating-point
// addition order, so results are expected to match scalar within a tolerance.
// Architecture-level performance still requires a validated gem5/board run.
std::vector<float> rvv_current(const Layer& layer,
                               const std::vector<float>& spikes) {
  std::vector<float> current(layer.outputs, 0.0F);
  for (std::size_t i = 0; i < layer.outputs; ++i) {
    float sum = 0.0F;
    for (std::size_t j = 0; j < layer.inputs;) {
      const std::size_t vl = __riscv_vsetvl_e32m1(layer.inputs - j);
      const vfloat32m1_t w = __riscv_vle32_v_f32m1(
          layer.weights.data() + i * layer.inputs + j, vl);
      const vfloat32m1_t s = __riscv_vle32_v_f32m1(spikes.data() + j, vl);
      const vfloat32m1_t product = __riscv_vfmul_vv_f32m1(w, s, vl);
      const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0F, vl);
      const vfloat32m1_t reduced =
          __riscv_vfredusum_vs_f32m1_f32m1(product, zero, vl);
      sum += __riscv_vfmv_f_s_f32m1_f32(reduced);
      j += vl;
    }
    current[i] = sum;
  }
  return current;
}
#endif

}  // namespace

// Check all dimensions, finite weights/parameters, and LIF parameter ranges.
void Layer::validate() const {
  if (inputs == 0 || outputs == 0 ||
      outputs > std::numeric_limits<std::size_t>::max() / inputs ||
      weights.size() != inputs * outputs || bias.size() != outputs) {
    throw std::invalid_argument("Invalid SNN layer dimensions");
  }
  if (!std::isfinite(beta) || beta < 0.0F || beta > 1.0F ||
      !std::isfinite(threshold) || threshold <= 0.0F) {
    throw std::invalid_argument("Invalid LIF beta/threshold");
  }
  for (float weight : weights) {
    if (!std::isfinite(weight)) throw std::invalid_argument("Nonfinite weight");
  }
  for (float b : bias) {
    if (!std::isfinite(b)) throw std::invalid_argument("Nonfinite bias");
  }
}

// Evaluate one timestep; both paths implement the SAME LIF dynamics.
std::vector<std::uint8_t> step(const Layer& layer,
                               const std::vector<float>& input_spikes,
                               State& state, Execution execution) {
  validate_input(layer, input_spikes, state);
  const auto current = (execution == Execution::Dense)
                           ? dense_current(layer, input_spikes)
                           : sparse_current(layer, input_spikes);
  return advance(layer, current, state);
}

// Form adjacent groups and average input weights and biases within each group.
GroupedLayer group_adjacent_mean(const Layer& layer, std::size_t group_size) {
  layer.validate();
  if (group_size == 0) throw std::invalid_argument("group_size must be positive");
  const std::size_t count = 1 + (layer.outputs - 1) / group_size;
  Layer aggregate;
  aggregate.inputs = layer.inputs;
  aggregate.outputs = count;
  aggregate.beta = layer.beta;
  aggregate.threshold = layer.threshold;
  aggregate.weights.assign(count * layer.inputs, 0.0F);
  aggregate.bias.assign(count, 0.0F);
  for (std::size_t g = 0; g < count; ++g) {
    const std::size_t begin = g * group_size;
    const std::size_t end = std::min(begin + group_size, layer.outputs);
    const float denominator = static_cast<float>(end - begin);
    for (std::size_t i = begin; i < end; ++i) {
      aggregate.bias[g] += layer.bias[i] / denominator;
      for (std::size_t j = 0; j < layer.inputs; ++j) {
        aggregate.weights[g * layer.inputs + j] +=
            layer.weights[i * layer.inputs + j] / denominator;
      }
    }
  }
  return {std::move(aggregate), layer.outputs, group_size};
}

// The group emits one spike; each member receives that same emitted event.
std::vector<std::uint8_t> grouped_step(const GroupedLayer& grouped,
                                       const std::vector<float>& input_spikes,
                                       State& group_state, Execution execution) {
  const auto group_spikes = step(grouped.aggregated, input_spikes, group_state, execution);
  std::vector<std::uint8_t> output(grouped.original_outputs);
  for (std::size_t i = 0; i < output.size(); ++i) {
    output[i] = group_spikes[i / grouped.group_size];
  }
  return output;
}

// Report RVV availability at compile time: do not silently fake vector results.
bool rvv_available() {
#if defined(__riscv_vector)
  return true;
#else
  return false;
#endif
}

// Run the explicit vector kernel only on an RVV-capable build.
std::vector<std::uint8_t> step_rvv(const Layer& layer,
                                   const std::vector<float>& input_spikes,
                                   State& state) {
#if defined(__riscv_vector)
  validate_input(layer, input_spikes, state);
  return advance(layer, rvv_current(layer, input_spikes), state);
#else
  (void)layer;
  (void)input_spikes;
  (void)state;
  throw std::runtime_error("RVV unavailable: build for a RISC-V V target");
#endif
}

}  // namespace snn
