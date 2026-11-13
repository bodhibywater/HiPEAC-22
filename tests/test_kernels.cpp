#include "snn/kernels.hpp"

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
    for (std::size_t outputs : {1U, 5U, 13U}) {
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
      std::vector<float> independent(outputs, 0.0F);
      const auto group1 = snn::group_adjacent_mean(layer, 1);
      for (int t = 0; t < 37; ++t) {
        std::vector<float> in(inputs);
        for (float& v : in) v = static_cast<float>(activity(rng));
        const auto expected = independent_step(layer, in, independent);
        const auto d = snn::step(layer, in, dense, snn::Execution::Dense);
        const auto s = snn::step(layer, in, sparse, snn::Execution::Sparse);
        const auto g = snn::grouped_step(group1, in, group_state, snn::Execution::Sparse);
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
    std::cout << "All C++ LIF kernel tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "TEST FAILURE: " << e.what() << '\n';
    return 1;
  }
}
