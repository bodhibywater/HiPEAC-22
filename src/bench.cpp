#include "snn/kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
  std::size_t inputs = 256;
  std::size_t outputs = 256;
  std::size_t steps = 64;
  std::size_t group = 1;
  int repeats = 9;
  int warmup = 3;
  double density = 0.1;
  std::uint32_t seed = 42;
  std::string mode = "dense";
};

// Parse self-contained command-line flags so every experiment is reproducible.
Config parse(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 >= argc) throw std::invalid_argument("flag missing value");
    const std::string name = argv[i++];
    const std::string value = argv[i];
    if (name == "--inputs") c.inputs = std::stoull(value);
    else if (name == "--outputs") c.outputs = std::stoull(value);
    else if (name == "--steps") c.steps = std::stoull(value);
    else if (name == "--group") c.group = std::stoull(value);
    else if (name == "--density") c.density = std::stod(value);
    else if (name == "--seed") c.seed = static_cast<std::uint32_t>(std::stoul(value));
    else if (name == "--repeats") c.repeats = std::stoi(value);
    else if (name == "--warmup") c.warmup = std::stoi(value);
    else if (name == "--mode") c.mode = value;
    else throw std::invalid_argument("unknown argument: " + name);
  }
  if (c.inputs == 0 || c.outputs == 0 || c.steps == 0 || c.group == 0 ||
      c.repeats < 1 || c.warmup < 0 || !std::isfinite(c.density) ||
      c.density < 0 || c.density > 1 ||
      (c.mode != "dense" && c.mode != "sparse" && c.mode != "rvv")) {
    throw std::invalid_argument("invalid benchmark configuration");
  }
  if (c.mode == "rvv" && !snn::rvv_available()) {
    throw std::runtime_error("RVV mode requires an RVV-compiled target");
  }
  return c;
}

// Fix the actual spike patterns and weights before measuring any kernel.
struct Workload {
  snn::Layer layer;
  std::vector<std::vector<float>> spikes;
  double realised_density = 0.0;
};

Workload make_workload(const Config& c) {
  std::mt19937 rng(c.seed);
  std::normal_distribution<float> weight(0.0F, 0.2F);
  std::bernoulli_distribution fire(c.density);
  Workload w;
  w.layer.inputs = c.inputs;
  w.layer.outputs = c.outputs;
  w.layer.beta = 0.9F;
  w.layer.threshold = 1.0F;
  w.layer.weights.resize(c.inputs * c.outputs);
  w.layer.bias.assign(c.outputs, 0.0F);
  for (float& v : w.layer.weights) v = weight(rng);
  std::size_t active_count = 0;
  for (std::size_t t = 0; t < c.steps; ++t) {
    auto& spikes = w.spikes.emplace_back(c.inputs);
    for (float& v : spikes) {
      v = static_cast<float>(fire(rng));
      active_count += static_cast<std::size_t>(v);
    }
  }
  w.realised_density = static_cast<double>(active_count) /
                       static_cast<double>(c.steps * c.inputs);
  return w;
}

// Time T consecutive timesteps with reset state; report per-repeat samples
// rather than only the smallest or a selectively chosen timing.
double run_once(const Config& c, const Workload& w, const snn::GroupedLayer& g,
                std::uint64_t& checksum) {
  const bool grouped = c.group != 1;
  snn::State state(grouped ? g.aggregated.outputs : w.layer.outputs);
  checksum = 0;  // Independent checksum for each repeat.
  const auto start = std::chrono::steady_clock::now();
  for (const auto& spikes : w.spikes) {
    std::vector<std::uint8_t> output;
    if (grouped) {
      if (c.mode == "rvv") {
        const auto gs = snn::step_rvv(g.aggregated, spikes, state);
        output.resize(w.layer.outputs);
        for (std::size_t i = 0; i < output.size(); ++i) {
          output[i] = gs[i / c.group];
        }
      } else {
        output = snn::grouped_step(g, spikes, state,
            c.mode == "dense" ? snn::Execution::Dense : snn::Execution::Sparse);
      }
    } else if (c.mode == "rvv") {
      output = snn::step_rvv(w.layer, spikes, state);
    } else {
      output = snn::step(w.layer, spikes, state,
          c.mode == "dense" ? snn::Execution::Dense : snn::Execution::Sparse);
    }
    for (std::uint8_t s : output) checksum += s;
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - start).count();
}

}  // namespace

// Output all native wall-clock samples as CSV; simulated cycles are separate.
int main(int argc, char** argv) {
  try {
    const Config c = parse(argc, argv);
    const auto workload = make_workload(c);
    const auto grouped = snn::group_adjacent_mean(workload.layer, c.group);
    std::uint64_t checksum = 0;
    std::cout << "mode,inputs,outputs,group,steps,requested_density,realised_density,seed,repeat,elapsed_us,spike_checksum,measurement\n";
    for (int i = -c.warmup; i < c.repeats; ++i) {
      const double elapsed = run_once(c, workload, grouped, checksum);
      if (i < 0) continue;
      std::cout << c.mode << ',' << c.inputs << ',' << c.outputs << ','
                << c.group << ',' << c.steps << ',' << c.density << ','
                << std::setprecision(9) << workload.realised_density << ','
                << c.seed << ',' << i << ',' << std::setprecision(12)
                << elapsed << ',' << checksum << ",native_wall_clock_us\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "BENCH ERROR: " << e.what() << '\n';
    return 2;
  }
}
