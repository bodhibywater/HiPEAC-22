#include "snn/temporal.hpp"

#include <limits>
#include <stdexcept>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace snn {
namespace {
#if defined(__riscv_vector)
struct Event {
  std::size_t input;
  unsigned mask;
};

// Same arithmetic order, threshold and hard reset as the acc1 fused baseline.
inline vfloat32m1_t lif(vfloat32m1_t potential, vfloat32m1_t current,
                       vfloat32m1_t bias, float beta, float threshold,
                       std::uint8_t* output, std::size_t vl) {
  auto voltage = __riscv_vfmul_vf_f32m1(potential, beta, vl);
  voltage = __riscv_vfadd_vv_f32m1(voltage, current, vl);
  voltage = __riscv_vfadd_vv_f32m1(voltage, bias, vl);
  const auto fired = __riscv_vmfge_vf_f32m1_b32(voltage, threshold, vl);
  const auto zero = __riscv_vfmv_v_f_f32m1(0.0F, vl);
  const auto next = __riscv_vmerge_vvm_f32m1(voltage, zero, fired, vl);
  const auto zeros = __riscv_vmv_v_x_u32m1(0, vl);
  const auto ones = __riscv_vmv_v_x_u32m1(1, vl);
  const auto spikes32 = __riscv_vmerge_vvm_u32m1(zeros, ones, fired, vl);
  const auto spikes16 = __riscv_vnsrl_wx_u16mf2(spikes32, 0, vl);
  const auto spikes8 = __riscv_vnsrl_wx_u8mf4(spikes16, 0, vl);
  __riscv_vse8_v_u8mf4(output, spikes8, vl);
  return next;
}

inline void lif_memory(const Layer& layer, State& state, std::size_t i,
                       vfloat32m1_t current, std::uint8_t* output,
                       std::size_t vl) {
  const auto potential = __riscv_vle32_v_f32m1(state.voltage.data() + i, vl);
  const auto bias = __riscv_vle32_v_f32m1(layer.bias.data() + i, vl);
  const auto next = lif(potential, current, bias, layer.beta, layer.threshold,
                        output, vl);
  __riscv_vse32_v_f32m1(state.voltage.data() + i, next, vl);
  // Compiler-only barrier: prevent the ablation from silently becoming the
  // resident implementation. Emits no hardware fence or instruction.
  asm volatile("" ::: "memory");
}

template <bool Resident>
std::vector<std::uint8_t> temporal_impl(
    const Layer& layer, const std::vector<float>& weights,
    const std::vector<float>* inputs, std::size_t steps, State& state) {
  if (!inputs || steps < 1 || steps > 4 || !layer.inputs || !layer.outputs ||
      layer.outputs > std::numeric_limits<std::size_t>::max() / layer.inputs ||
      layer.outputs > std::numeric_limits<std::size_t>::max() / steps ||
      layer.weights.size() != layer.inputs * layer.outputs ||
      weights.size() != layer.weights.size() ||
      layer.bias.size() != layer.outputs || state.voltage.size() != layer.outputs) {
    throw std::invalid_argument("Temporal SNN input/state shape mismatch");
  }
  for (std::size_t t = 0; t < steps; ++t) {
    if (inputs[t].size() != layer.inputs) {
      throw std::invalid_argument("Temporal SNN input shape mismatch");
    }
  }

  // Encode active input rows once per window, in ascending input order.
  // Cache the pointers instead of repeatedly indexing vector objects.
  const float* x0 = inputs[0].data();
  const float* x1 = steps > 1 ? inputs[1].data() : nullptr;
  const float* x2 = steps > 2 ? inputs[2].data() : nullptr;
  const float* x3 = steps > 3 ? inputs[3].data() : nullptr;
  std::vector<Event> active;
  active.reserve(layer.inputs);
  for (std::size_t j = 0; j < layer.inputs; ++j) {
    const float a = x0[j], b = x1 ? x1[j] : 0.0F;
    const float c = x2 ? x2[j] : 0.0F, d = x3 ? x3[j] : 0.0F;
    if ((a != 0.0F && a != 1.0F) || (b != 0.0F && b != 1.0F) ||
        (c != 0.0F && c != 1.0F) || (d != 0.0F && d != 1.0F)) {
      throw std::invalid_argument("SNN inputs must contain binary spikes");
    }
    const unsigned mask = unsigned(a == 1.0F) | (unsigned(b == 1.0F) << 1) |
                          (unsigned(c == 1.0F) << 2) | (unsigned(d == 1.0F) << 3);
    if (mask) active.push_back({j, mask});
  }

  const std::size_t n = layer.outputs;
  std::vector<std::uint8_t> output(steps * n);
  for (std::size_t i = 0; i < n;) {
    const std::size_t vl = __riscv_vsetvl_e32m1(n - i);
    auto c0 = __riscv_vfmv_v_f_f32m1(0.0F, vl);
    auto c1 = c0, c2 = c0, c3 = c0;
    for (const auto& e : active) {
      // The one loaded weight vector is shared across every active timestep.
      // Each accumulator sees the same ordered additions as the baseline.
      const auto w = __riscv_vle32_v_f32m1(weights.data() + e.input * n + i, vl);
      if (e.mask & 1U) c0 = __riscv_vfadd_vv_f32m1(c0, w, vl);
      if (e.mask & 2U) c1 = __riscv_vfadd_vv_f32m1(c1, w, vl);
      if (e.mask & 4U) c2 = __riscv_vfadd_vv_f32m1(c2, w, vl);
      if (e.mask & 8U) c3 = __riscv_vfadd_vv_f32m1(c3, w, vl);
    }
    if constexpr (Resident) {
      auto v = __riscv_vle32_v_f32m1(state.voltage.data() + i, vl);
      const auto bias = __riscv_vle32_v_f32m1(layer.bias.data() + i, vl);
      v = lif(v, c0, bias, layer.beta, layer.threshold, output.data() + i, vl);
      if (steps > 1) v = lif(v, c1, bias, layer.beta, layer.threshold, output.data() + n + i, vl);
      if (steps > 2) v = lif(v, c2, bias, layer.beta, layer.threshold, output.data() + 2*n + i, vl);
      if (steps > 3) v = lif(v, c3, bias, layer.beta, layer.threshold, output.data() + 3*n + i, vl);
      __riscv_vse32_v_f32m1(state.voltage.data() + i, v, vl);
    } else {
      lif_memory(layer, state, i, c0, output.data() + i, vl);
      if (steps > 1) lif_memory(layer, state, i, c1, output.data() + n + i, vl);
      if (steps > 2) lif_memory(layer, state, i, c2, output.data() + 2*n + i, vl);
      if (steps > 3) lif_memory(layer, state, i, c3, output.data() + 3*n + i, vl);
    }
    i += vl;
  }
  return output;
}
#endif
}  // namespace

std::vector<std::uint8_t> step_sparse_rvv_temporal4(
    const Layer& layer, const std::vector<float>& input_major_weights,
    const std::vector<float>* inputs_by_time, std::size_t steps,
    State& state, bool resident) {
#if defined(__riscv_vector)
  if (resident) return temporal_impl<true>(layer, input_major_weights, inputs_by_time, steps, state);
  return temporal_impl<false>(layer, input_major_weights, inputs_by_time, steps, state);
#else
  (void)layer; (void)input_major_weights; (void)inputs_by_time;
  (void)steps; (void)state; (void)resident;
  throw std::runtime_error("Temporal sparse RVV requires a RISC-V vector target");
#endif
}
}  // namespace snn
