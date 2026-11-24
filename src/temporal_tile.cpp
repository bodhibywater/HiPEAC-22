#include "snn/temporal_tile.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>

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

// Keep the RVV types local to the operations and explicit accumulators;
// sizeless RVV values cannot be stored in C++ arrays or std::vector.
struct M1 {
  using Vec = vfloat32m1_t;
  static inline std::size_t vl(std::size_t n) { return __riscv_vsetvl_e32m1(n); }
  static inline Vec zero(std::size_t vl) { return __riscv_vfmv_v_f_f32m1(0.0F, vl); }
  static inline Vec load(const float* p, std::size_t vl) { return __riscv_vle32_v_f32m1(p, vl); }
  static inline Vec add(Vec a, Vec b, std::size_t vl) { return __riscv_vfadd_vv_f32m1(a, b, vl); }
  static inline void store(float* p, Vec a, std::size_t vl) { __riscv_vse32_v_f32m1(p, a, vl); }
  static inline Vec lif(Vec potential, Vec current, Vec bias, float beta,
                        float threshold, std::uint8_t* output, std::size_t vl) {
    auto v = __riscv_vfmul_vf_f32m1(potential, beta, vl);
    v = add(v, current, vl);
    v = add(v, bias, vl);
    const auto fired = __riscv_vmfge_vf_f32m1_b32(v, threshold, vl);
    const auto next = __riscv_vmerge_vvm_f32m1(v, zero(vl), fired, vl);
    const auto zeros = __riscv_vmv_v_x_u32m1(0, vl);
    const auto ones = __riscv_vmv_v_x_u32m1(1, vl);
    const auto spikes32 = __riscv_vmerge_vvm_u32m1(zeros, ones, fired, vl);
    const auto spikes16 = __riscv_vnsrl_wx_u16mf2(spikes32, 0, vl);
    const auto spikes8 = __riscv_vnsrl_wx_u8mf4(spikes16, 0, vl);
    __riscv_vse8_v_u8mf4(output, spikes8, vl);
    return next;
  }
};

struct M2 {
  using Vec = vfloat32m2_t;
  static inline std::size_t vl(std::size_t n) { return __riscv_vsetvl_e32m2(n); }
  static inline Vec zero(std::size_t vl) { return __riscv_vfmv_v_f_f32m2(0.0F, vl); }
  static inline Vec load(const float* p, std::size_t vl) { return __riscv_vle32_v_f32m2(p, vl); }
  static inline Vec add(Vec a, Vec b, std::size_t vl) { return __riscv_vfadd_vv_f32m2(a, b, vl); }
  static inline void store(float* p, Vec a, std::size_t vl) { __riscv_vse32_v_f32m2(p, a, vl); }
  static inline Vec lif(Vec potential, Vec current, Vec bias, float beta,
                        float threshold, std::uint8_t* output, std::size_t vl) {
    auto v = __riscv_vfmul_vf_f32m2(potential, beta, vl);
    v = add(v, current, vl);
    v = add(v, bias, vl);
    const auto fired = __riscv_vmfge_vf_f32m2_b16(v, threshold, vl);
    const auto next = __riscv_vmerge_vvm_f32m2(v, zero(vl), fired, vl);
    const auto zeros = __riscv_vmv_v_x_u32m2(0, vl);
    const auto ones = __riscv_vmv_v_x_u32m2(1, vl);
    const auto spikes32 = __riscv_vmerge_vvm_u32m2(zeros, ones, fired, vl);
    const auto spikes16 = __riscv_vnsrl_wx_u16m1(spikes32, 0, vl);
    const auto spikes8 = __riscv_vnsrl_wx_u8mf2(spikes16, 0, vl);
    __riscv_vse8_v_u8mf2(output, spikes8, vl);
    return next;
  }
};

// The non-resident ablation matches A1: compiler barrier, no hardware fence.
template <typename Ops>
inline void lif_memory(const Layer& layer, State& state, std::size_t i,
                       typename Ops::Vec current, std::uint8_t* output,
                       std::size_t vl) {
  const auto v = Ops::load(state.voltage.data() + i, vl);
  const auto bias = Ops::load(layer.bias.data() + i, vl);
  const auto next = Ops::lif(v, current, bias, layer.beta, layer.threshold, output, vl);
  Ops::store(state.voltage.data() + i, next, vl);
  asm volatile("" ::: "memory");
}

template <std::size_t K, std::size_t Window>
inline unsigned spike_bit(const std::array<const float*, Window>& inputs,
                          std::size_t j) {
  if constexpr (K < Window) {
    const float x = inputs[K] ? inputs[K][j] : 0.0F;
    if (x != 0.0F && x != 1.0F) {
      throw std::invalid_argument("SNN inputs must contain binary spikes");
    }
    return unsigned(x == 1.0F) << K;
  } else {
    return 0;
  }
}

template <typename Ops, std::size_t Window, bool Coalesce, bool Resident>
std::vector<std::uint8_t> tile_impl(
    const Layer& layer, const std::vector<float>& weights,
    const std::vector<float>* inputs, std::size_t steps, State& state) {
  if (!inputs || steps < 1 || steps > Window || !layer.inputs || !layer.outputs ||
      layer.outputs > std::numeric_limits<std::size_t>::max() / layer.inputs ||
      layer.outputs > std::numeric_limits<std::size_t>::max() / steps ||
      layer.weights.size() != layer.inputs * layer.outputs ||
      weights.size() != layer.weights.size() ||
      layer.bias.size() != layer.outputs || state.voltage.size() != layer.outputs) {
    throw std::invalid_argument("Temporal tile input/state shape mismatch");
  }
  std::array<const float*, Window> x{};
  for (std::size_t t = 0; t < steps; ++t) {
    if (inputs[t].size() != layer.inputs) {
      throw std::invalid_argument("Temporal tile input shape mismatch");
    }
    x[t] = inputs[t].data();
  }

  // Both schedules use the same binary validation and input-major ordering.
  // The separate-list control does not deliberately reload via volatile or
  // extra barriers: it genuinely traverses one list per timestep.
  using Lists = std::conditional_t<Coalesce, std::vector<Event>,
                                 std::array<std::vector<std::size_t>, Window>>;
  Lists events;
  if constexpr (Coalesce) {
    events.reserve(layer.inputs);
  } else {
    for (std::size_t t = 0; t < steps; ++t) events[t].reserve(layer.inputs);
  }
  for (std::size_t j = 0; j < layer.inputs; ++j) {
    const unsigned mask = spike_bit<0>(x, j) | spike_bit<1>(x, j) |
                          spike_bit<2>(x, j) | spike_bit<3>(x, j) |
                          spike_bit<4>(x, j) | spike_bit<5>(x, j) |
                          spike_bit<6>(x, j) | spike_bit<7>(x, j);
    if constexpr (Coalesce) {
      if (mask) events.push_back({j, mask});
    } else {
      for (std::size_t t = 0; t < steps; ++t) {
        if (mask & (1U << t)) events[t].push_back(j);
      }
    }
  }

  const std::size_t n = layer.outputs;
  std::vector<std::uint8_t> output(steps * n);
  for (std::size_t i = 0; i < n;) {
    const std::size_t vl = Ops::vl(n - i);
    auto c0 = Ops::zero(vl);
    auto c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    if constexpr (Coalesce) {
      for (const auto& e : events) {
        const auto w = Ops::load(weights.data() + e.input * n + i, vl);
        if constexpr (Window > 0) { if (e.mask & 1U) c0 = Ops::add(c0, w, vl); }
        if constexpr (Window > 1) { if (e.mask & 2U) c1 = Ops::add(c1, w, vl); }
        if constexpr (Window > 2) { if (e.mask & 4U) c2 = Ops::add(c2, w, vl); }
        if constexpr (Window > 3) { if (e.mask & 8U) c3 = Ops::add(c3, w, vl); }
        if constexpr (Window > 4) { if (e.mask & 16U) c4 = Ops::add(c4, w, vl); }
        if constexpr (Window > 5) { if (e.mask & 32U) c5 = Ops::add(c5, w, vl); }
        if constexpr (Window > 6) { if (e.mask & 64U) c6 = Ops::add(c6, w, vl); }
        if constexpr (Window > 7) { if (e.mask & 128U) c7 = Ops::add(c7, w, vl); }
      }
    } else {
      // Separate ordered weight passes, with the same output tile and LIF order.
      if constexpr (Window > 0) {
        for (std::size_t j : events[0]) c0 = Ops::add(c0, Ops::load(weights.data() + j * n + i, vl), vl);
      }
      if constexpr (Window > 1) {
        for (std::size_t j : events[1]) c1 = Ops::add(c1, Ops::load(weights.data() + j * n + i, vl), vl);
      }
      if constexpr (Window > 2) {
        for (std::size_t j : events[2]) c2 = Ops::add(c2, Ops::load(weights.data() + j * n + i, vl), vl);
      }
      if constexpr (Window > 3) {
        for (std::size_t j : events[3]) c3 = Ops::add(c3, Ops::load(weights.data() + j * n + i, vl), vl);
      }
      if constexpr (Window > 4) {
        for (std::size_t j : events[4]) c4 = Ops::add(c4, Ops::load(weights.data() + j * n + i, vl), vl);
      }
      if constexpr (Window > 5) {
        for (std::size_t j : events[5]) c5 = Ops::add(c5, Ops::load(weights.data() + j * n + i, vl), vl);
      }
      if constexpr (Window > 6) {
        for (std::size_t j : events[6]) c6 = Ops::add(c6, Ops::load(weights.data() + j * n + i, vl), vl);
      }
      if constexpr (Window > 7) {
        for (std::size_t j : events[7]) c7 = Ops::add(c7, Ops::load(weights.data() + j * n + i, vl), vl);
      }
    }
    if constexpr (Resident) {
      auto v = Ops::load(state.voltage.data() + i, vl);
      const auto bias = Ops::load(layer.bias.data() + i, vl);
      v = Ops::lif(v, c0, bias, layer.beta, layer.threshold, output.data() + i, vl);
      if constexpr (Window > 1) { if (steps > 1) v = Ops::lif(v, c1, bias, layer.beta, layer.threshold, output.data() + 1*n + i, vl); }
      if constexpr (Window > 2) { if (steps > 2) v = Ops::lif(v, c2, bias, layer.beta, layer.threshold, output.data() + 2*n + i, vl); }
      if constexpr (Window > 3) { if (steps > 3) v = Ops::lif(v, c3, bias, layer.beta, layer.threshold, output.data() + 3*n + i, vl); }
      if constexpr (Window > 4) { if (steps > 4) v = Ops::lif(v, c4, bias, layer.beta, layer.threshold, output.data() + 4*n + i, vl); }
      if constexpr (Window > 5) { if (steps > 5) v = Ops::lif(v, c5, bias, layer.beta, layer.threshold, output.data() + 5*n + i, vl); }
      if constexpr (Window > 6) { if (steps > 6) v = Ops::lif(v, c6, bias, layer.beta, layer.threshold, output.data() + 6*n + i, vl); }
      if constexpr (Window > 7) { if (steps > 7) v = Ops::lif(v, c7, bias, layer.beta, layer.threshold, output.data() + 7*n + i, vl); }
      Ops::store(state.voltage.data() + i, v, vl);
    } else {
      lif_memory<Ops>(layer, state, i, c0, output.data() + i, vl);
      if constexpr (Window > 1) { if (steps > 1) lif_memory<Ops>(layer, state, i, c1, output.data() + 1*n + i, vl); }
      if constexpr (Window > 2) { if (steps > 2) lif_memory<Ops>(layer, state, i, c2, output.data() + 2*n + i, vl); }
      if constexpr (Window > 3) { if (steps > 3) lif_memory<Ops>(layer, state, i, c3, output.data() + 3*n + i, vl); }
      if constexpr (Window > 4) { if (steps > 4) lif_memory<Ops>(layer, state, i, c4, output.data() + 4*n + i, vl); }
      if constexpr (Window > 5) { if (steps > 5) lif_memory<Ops>(layer, state, i, c5, output.data() + 5*n + i, vl); }
      if constexpr (Window > 6) { if (steps > 6) lif_memory<Ops>(layer, state, i, c6, output.data() + 6*n + i, vl); }
      if constexpr (Window > 7) { if (steps > 7) lif_memory<Ops>(layer, state, i, c7, output.data() + 7*n + i, vl); }
    }
    i += vl;
  }
  return output;
}

template <typename Ops, std::size_t Window>
std::vector<std::uint8_t> dispatch_flags(
    const Layer& layer, const std::vector<float>& weights,
    const std::vector<float>* inputs, std::size_t steps, State& state,
    bool coalesce, bool resident) {
  if (coalesce) {
    if (resident) return tile_impl<Ops, Window, true, true>(layer, weights, inputs, steps, state);
    return tile_impl<Ops, Window, true, false>(layer, weights, inputs, steps, state);
  }
  if (resident) return tile_impl<Ops, Window, false, true>(layer, weights, inputs, steps, state);
  return tile_impl<Ops, Window, false, false>(layer, weights, inputs, steps, state);
}

template <typename Ops>
std::vector<std::uint8_t> dispatch_window(
    const Layer& layer, const std::vector<float>& weights,
    const std::vector<float>* inputs, std::size_t steps, State& state,
    std::size_t window, bool coalesce, bool resident) {
  switch (window) {
    case 1: return dispatch_flags<Ops, 1>(layer, weights, inputs, steps, state, coalesce, resident);
    case 2: return dispatch_flags<Ops, 2>(layer, weights, inputs, steps, state, coalesce, resident);
    case 4: return dispatch_flags<Ops, 4>(layer, weights, inputs, steps, state, coalesce, resident);
    case 8: return dispatch_flags<Ops, 8>(layer, weights, inputs, steps, state, coalesce, resident);
    default: throw std::invalid_argument("Unsupported temporal tile size");
  }
}
#endif
}  // namespace

std::vector<std::uint8_t> step_sparse_rvv_tile(
    const Layer& layer, const std::vector<float>& weights,
    const std::vector<float>* inputs, std::size_t steps, State& state,
    std::size_t window, unsigned lmul, bool coalesce, bool resident) {
#if defined(__riscv_vector)
  if (lmul == 1) return dispatch_window<M1>(layer, weights, inputs, steps, state, window, coalesce, resident);
  if (lmul == 2) return dispatch_window<M2>(layer, weights, inputs, steps, state, window, coalesce, resident);
  throw std::invalid_argument("Unsupported temporal tile LMUL");
#else
  (void)layer; (void)weights; (void)inputs; (void)steps; (void)state;
  (void)window; (void)lmul; (void)coalesce; (void)resident;
  throw std::runtime_error("Temporal tile requires a RISC-V vector target");
#endif
}
}  // namespace snn
