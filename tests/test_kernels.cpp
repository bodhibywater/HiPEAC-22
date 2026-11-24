#include "snn/kernels.hpp"
#include "snn/temporal.hpp"
#include "snn/temporal_tile.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

// Turn an observed assertion failure into a non-zero CI exit status.
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// Independently implement the LIF formula without calling production kernels.
std::vector<std::uint8_t> independent_step(
    const snn::Layer& layer, const std::vector<float>& input,
    std::vector<float>& potential) {
  std::vector<std::uint8_t> spikes(layer.outputs);
  for (std::size_t i = 0; i < layer.outputs; ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < layer.inputs; ++j) {
      sum += static_cast<double>(layer.weights[i * layer.inputs + j]) * input[j];
    }
    const float u = layer.beta * potential[i] +
                    static_cast<float>(sum) + layer.bias[i];
    spikes[i] = static_cast<std::uint8_t>(u >= layer.threshold);
    potential[i] = spikes[i] ? 0.0F : u;
  }
  return spikes;
}

// Check dense/sparse parity, independent reference, and g=1 grouping on many
// seeded timesteps. Tolerance accounts for different floating-point evaluation.
void random_parity() {
  std::mt19937 rng(20261113);
  std::uniform_real_distribution<float> values(-0.4F, 0.7F);
  std::bernoulli_distribution activity(0.24);
  for (std::size_t inputs : {1U, 3U, 17U, 65U}) {
    for (std::size_t outputs : {1U, 5U, 13U, 129U}) {
      snn::Layer layer;
      layer.inputs = inputs;
      layer.outputs = outputs;
      layer.weights.resize(inputs * outputs);
      layer.bias.resize(outputs);
      layer.beta = 0.83F;
      layer.threshold = 1.25F;
      for (auto& v : layer.weights) v = values(rng);
      for (auto& v : layer.bias) v = values(rng) * 0.1F;
      layer.validate();
      snn::State dense(outputs), sparse(outputs), group_state(outputs);
      snn::State sparse_rvv_state(outputs), dense_rvv_state(outputs);
      snn::State sparse_scalar_im_state(outputs);
      snn::State dense_fused_rvv_state(outputs);
      snn::State fused_rvv_state(outputs);
      snn::State acc2_rvv_state(outputs), acc4_rvv_state(outputs);
      const auto input_major = snn::prepare_input_major_weights(layer);
      std::vector<float> independent(outputs, 0.0F);
      const auto group1 = snn::group_adjacent_mean(layer, 1);
      for (int t = 0; t < 37; ++t) {
        std::vector<float> in(inputs);
        for (float& v : in) v = static_cast<float>(activity(rng));
        const auto expected = independent_step(layer, in, independent);
        const auto d = snn::step(layer, in, dense, snn::Execution::Dense);
        const auto s = snn::step(layer, in, sparse, snn::Execution::Sparse);
        const auto g = snn::grouped_step(group1, in, group_state, snn::Execution::Sparse);
        const auto si = snn::step_sparse_scalar_input_major(
            layer, input_major, in, sparse_scalar_im_state);
        require(si == expected, "Input-major sparse scalar spikes mismatch");
        for (std::size_t i = 0; i < outputs; ++i) {
          require(std::fabs(sparse_scalar_im_state.voltage[i] - independent[i]) <= 1e-5F,
                  "Input-major sparse scalar voltage mismatch");
        }
        if (snn::rvv_available()) {
          const auto sr = snn::step_sparse_rvv(
              layer, input_major, in, sparse_rvv_state);
          const auto dr = snn::step_rvv(layer, in, dense_rvv_state);
          const auto df = snn::step_dense_rvv_fused(
              layer, input_major, in, dense_fused_rvv_state);
          const auto fused = snn::step_sparse_rvv_fused(
              layer, input_major, in, fused_rvv_state);
          const auto acc2 = snn::step_sparse_rvv_fused_multi(
              layer, input_major, in, acc2_rvv_state, 2);
          const auto acc4 = snn::step_sparse_rvv_fused_multi(
              layer, input_major, in, acc4_rvv_state, 4);
          require(sr == expected, "Sparse RVV emitted different spikes");
          require(fused == expected, "Fused sparse RVV emitted different spikes");
          require(fused == sr, "Fused/unfused sparse RVV mismatch");
          require(acc2 == expected, "2-accumulator RVV emitted different spikes");
          require(acc4 == expected, "4-accumulator RVV emitted different spikes");
          require(dr == expected, "Dense RVV emitted different spikes");
          require(df == expected, "Fused dense RVV emitted different spikes");
          for (std::size_t i = 0; i < outputs; ++i) {
            require(std::fabs(sparse_rvv_state.voltage[i] - independent[i]) <= 1e-5F,
                    "Sparse RVV voltage mismatch");
            require(std::fabs(fused_rvv_state.voltage[i] - sparse_rvv_state.voltage[i]) <= 1e-5F,
                    "Fused/unfused RVV voltage mismatch");
            require(std::fabs(acc2_rvv_state.voltage[i] - independent[i]) <= 5e-5F,
                    "2-accumulator RVV voltage mismatch");
            require(std::fabs(acc4_rvv_state.voltage[i] - independent[i]) <= 5e-5F,
                    "4-accumulator RVV voltage mismatch");
            require(std::fabs(dense_rvv_state.voltage[i] - independent[i]) <= 1e-5F,
                    "Dense RVV voltage mismatch");
            require(std::fabs(dense_fused_rvv_state.voltage[i] - independent[i]) <= 1e-5F,
                    "Fused dense RVV voltage mismatch");
          }
        }
        require(d == expected && s == expected && g == expected,
                "Incorrect emitted spikes");
        for (std::size_t i = 0; i < outputs; ++i) {
          require(std::fabs(dense.voltage[i] - sparse.voltage[i]) <= 1e-5F,
                  "Dense/sparse voltages diverged");
          require(std::fabs(dense.voltage[i] - independent[i]) <= 1e-5F,
                  "Reference voltage mismatch");
          require(std::fabs(dense.voltage[i] - group_state.voltage[i]) <= 1e-5F,
                  "g=1 is not equivalent to raw");
        }
      }
    }
  }
}

// Both A variants must match the original acc1 RVV kernel exactly, including
// membrane values at every window boundary and short final windows.
void temporal_parity() {
  if (!snn::rvv_available()) {
    snn::Layer layer{1, 1, {1.0F}, {0.0F}, 0.5F, 1.0F};
    snn::State state(1);
    const std::vector<float> input{1.0F};
    for (bool resident : {false, true}) {
      bool rejected = false;
      try { (void)snn::step_sparse_rvv_temporal4(layer, layer.weights, &input, 1, state, resident); }
      catch (const std::runtime_error&) { rejected = true; }
      require(rejected, "Temporal RVV must not fall back silently");
    }
    return;
  }
  std::mt19937 rng(20261124);
  std::uniform_real_distribution<float> weight(-0.3F, 0.5F);
  std::bernoulli_distribution active(0.28);
  for (std::size_t ni : {1U, 17U, 65U}) {
    for (std::size_t no : {1U, 10U, 13U, 129U}) {
      snn::Layer layer{ni, no, std::vector<float>(ni*no),
                       std::vector<float>(no), 0.83F, 1.25F};
      for (auto& x : layer.weights) x = weight(rng);
      for (auto& x : layer.bias) x = weight(rng) * 0.1F;
      const auto weights = snn::prepare_input_major_weights(layer);
      snn::State ref(no), a1(no), a2(no);
      for (std::size_t i = 0; i < no; ++i) {
        ref.voltage[i] = a1.voltage[i] = a2.voltage[i] = weight(rng);
      }
      for (unsigned w = 0; w < 20; ++w) {
        const std::size_t len = w < 17 ? 4 : w - 16;
        std::vector<std::vector<float>> inputs(len, std::vector<float>(ni));
        for (std::size_t t = 0; t < len; ++t) {
          for (std::size_t j = 0; j < ni; ++j) {
            // Silent, all-active, repeated, disjoint and random activity.
            inputs[t][j] = w == 0 ? 0.0F : w == 1 ? 1.0F :
                w == 2 ? float(j % 2 == 0) : w == 3 ? float(j % 4 == t) :
                float(active(rng));
          }
        }
        std::vector<std::uint8_t> expected;
        for (const auto& input : inputs) {
          const auto spikes = snn::step_sparse_rvv_fused(layer, weights, input, ref);
          expected.insert(expected.end(), spikes.begin(), spikes.end());
        }
        require(snn::step_sparse_rvv_temporal4(layer, weights, inputs.data(), len, a1, false)
                    == expected, "A1 spike mismatch");
        require(snn::step_sparse_rvv_temporal4(layer, weights, inputs.data(), len, a2, true)
                    == expected, "A2 spike mismatch");
        require(a1.voltage == ref.voltage && a2.voltage == ref.voltage,
                "Temporal membrane state mismatch");
      }
      const std::vector<float> malformed(ni, 0.5F);
      bool rejected = false;
      try { (void)snn::step_sparse_rvv_temporal4(layer, weights, &malformed, 1, a2, true); }
      catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "Temporal nonbinary input must be rejected");
    }
  }
}

// B: test the full factorial, including short windows, output tails, all
// activity extremes, nonzero initial state and state continuity across tiles.
void tile_parity() {
  if (!snn::rvv_available()) {
    snn::Layer layer{1, 1, {1.0F}, {0.0F}, 0.5F, 1.0F};
    snn::State state(1);
    const std::vector<float> input{1.0F};
    bool rejected = false;
    try { (void)snn::step_sparse_rvv_tile(layer, layer.weights, &input, 1, state, 1, 1, true, true); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Temporal tile must not fall back silently");
    return;
  }
  std::mt19937 rng(20261125);
  std::uniform_real_distribution<float> weight(-0.3F, 0.5F);
  std::bernoulli_distribution active(0.28);
  for (std::size_t ni : {1U, 17U, 65U}) {
    for (std::size_t no : {1U, 10U, 13U, 129U, 256U}) {
      snn::Layer layer{ni, no, std::vector<float>(ni*no),
                       std::vector<float>(no), 0.83F, 1.25F};
      for (auto& x : layer.weights) x = weight(rng);
      for (auto& x : layer.bias) x = weight(rng) * 0.1F;
      const auto weights = snn::prepare_input_major_weights(layer);
      for (std::size_t window : {1U, 2U, 4U, 8U}) {
        for (unsigned lmul : {1U, 2U}) {
          for (bool shared : {false, true}) {
            for (bool resident : {false, true}) {
              snn::State ref(no), tiled(no);
              for (std::size_t i = 0; i < no; ++i) ref.voltage[i] = tiled.voltage[i] = weight(rng);
              for (std::size_t w = 0; w < 8 + window; ++w) {
                const std::size_t len = w < 8 ? window : w - 7;
                std::vector<std::vector<float>> inputs(len, std::vector<float>(ni));
                for (std::size_t t = 0; t < len; ++t) {
                  for (std::size_t j = 0; j < ni; ++j) {
                    inputs[t][j] = w == 0 ? 0.0F : w == 1 ? 1.0F :
                        w == 2 ? float(j % 2 == 0) : w == 3 ? float(j % window == t) :
                        float(active(rng));
                  }
                }
                std::vector<std::uint8_t> expected;
                for (const auto& input : inputs) {
                  const auto spikes = snn::step_sparse_rvv_fused(layer, weights, input, ref);
                  expected.insert(expected.end(), spikes.begin(), spikes.end());
                }
                const auto actual = snn::step_sparse_rvv_tile(layer, weights, inputs.data(), len,
                    tiled, window, lmul, shared, resident);
                require(actual == expected, "B tile spike mismatch");
                require(tiled.voltage == ref.voltage, "B tile membrane mismatch");
              }
              const std::vector<float> malformed(ni, 0.5F);
              bool rejected = false;
              try { (void)snn::step_sparse_rvv_tile(layer, weights, &malformed, 1,
                  tiled, window, lmul, shared, resident); }
              catch (const std::invalid_argument&) { rejected = true; }
              require(rejected, "B tile must reject nonbinary input");
            }
          }
        }
      }
    }
  }
}

// Explicit edge cases: empty activity must still leak state, threshold must
// fire on equality, and grouped dense/sparse must agree including partial groups.
void edge_cases() {
  snn::Layer layer{3, 5,
                   {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
                   {0, 0, 0, 0, 0}, 0.5F, 1.0F};
  layer.validate();
  snn::State s(5);
  const auto fired = snn::step(layer, {1, 0, 0}, s, snn::Execution::Sparse);
  require(fired == std::vector<std::uint8_t>(5, 1),
          "Threshold equality must fire");
  for (float v : s.voltage) require(v == 0.0F, "Reset must be zero");
  s.voltage[0] = 0.8F;
  const auto empty = snn::step(layer, {0, 0, 0}, s, snn::Execution::Sparse);
  require(empty[0] == 0 && std::fabs(s.voltage[0] - 0.4F) < 1e-6F,
          "No incoming spikes must still leak voltage");

  const auto grouped = snn::group_adjacent_mean(layer, 2);
  require(grouped.aggregated.outputs == 3, "Final partial group missing");
  snn::State a(3), b(3);
  const auto da = snn::grouped_step(grouped, {1, 0, 0}, a, snn::Execution::Dense);
  const auto sb = snn::grouped_step(grouped, {1, 0, 0}, b, snn::Execution::Sparse);
  require(da == sb && a.voltage == b.voltage, "Grouped mode parity failed");

  bool rejected = false;
  try { (void)snn::step(layer, {0, 0.5F, 1}, s, snn::Execution::Sparse); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "Nonbinary input must be rejected");
  require(!snn::rvv_available() || sizeof(float) == 4, "Unexpected float size");
}

// Validate reproducibility and correctness before making performance claims.
}  // namespace

int main() {
  try {
    random_parity();
    edge_cases();
    temporal_parity();
    tile_parity();
    std::cout << "All C++ LIF kernel tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "TEST FAILURE: " << e.what() << '\n';
    return 1;
  }
}
