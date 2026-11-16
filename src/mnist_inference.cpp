#include "snn/kernels.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SNN_GEM5
#include <gem5/m5ops.h>
#endif

namespace {

// Each record is an MNIST image with exactly reproducible input spikes and
// precomputed PyTorch reference spikes for every hidden and output neuron.
struct Sample {
  std::uint8_t label = 0, expected_prediction = 0;
  std::vector<std::vector<float>> inputs_by_time;
  std::vector<std::uint8_t> expected_hidden, expected_output;
};

struct Workload {
  snn::Layer first, second;
  std::vector<float> first_input_major, second_input_major;
  std::vector<Sample> samples;
  std::uint32_t steps = 0;
};

struct Result {
  std::uint64_t hidden_spikes = 0, output_spikes = 0;
  std::uint64_t hidden_mismatches = 0, output_mismatches = 0, prediction_mismatches = 0;
  std::uint64_t correct_predictions = 0;
};

// A deliberately tiny binary reader: the export format is explicitly
// little-endian, and our supported x86-64/RISC-V Linux targets are little-endian.
template <typename T>
T read_scalar(std::ifstream& file) {
  T value{};
  if (!file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
    throw std::runtime_error("Truncated MNIST trace header or sample");
  }
  return value;
}

// Read a tensor of exactly the declared dimensions and fail on a short file.
template <typename T>
std::vector<T> read_tensor(std::ifstream& file, std::size_t n) {
  if (n > std::numeric_limits<std::streamsize>::max() / sizeof(T)) {
    throw std::runtime_error("MNIST trace tensor exceeds supported size");
  }
  std::vector<T> values(n);
  if (!file.read(reinterpret_cast<char*>(values.data()),
                 static_cast<std::streamsize>(n * sizeof(T)))) {
    throw std::runtime_error("Truncated MNIST trace tensor");
  }
  return values;
}

// Load all weights and input traces before entering the gem5 measurement ROI.
Workload read_workload(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("Cannot open MNIST trace: " + path);
  std::array<char, 8> magic{};
  if (!file.read(magic.data(), magic.size()) ||
      std::memcmp(magic.data(), "SNNMNST1", 8) != 0) {
    throw std::runtime_error("Not an SNNMNST1 trace");
  }
  const auto n = read_scalar<std::uint32_t>(file);
  const auto steps = read_scalar<std::uint32_t>(file);
  const auto inputs = read_scalar<std::uint32_t>(file);
  const auto hidden = read_scalar<std::uint32_t>(file);
  const auto outputs = read_scalar<std::uint32_t>(file);
  const auto beta = read_scalar<float>(file);
  const auto threshold = read_scalar<float>(file);
  if (n == 0 || n > 10000 || steps == 0 || steps > 1024 || inputs != 784 ||
      hidden == 0 || hidden > 8192 || outputs != 10) {
    throw std::runtime_error("Invalid MNIST trace dimensions");
  }

  Workload workload;
  workload.steps = steps;
  workload.first = snn::Layer{inputs, hidden,
      read_tensor<float>(file, std::size_t(inputs) * hidden),
      read_tensor<float>(file, hidden), beta, threshold};
  workload.second = snn::Layer{hidden, outputs,
      read_tensor<float>(file, std::size_t(hidden) * outputs),
      read_tensor<float>(file, outputs), beta, threshold};
  workload.first.validate();
  workload.second.validate();

  for (std::uint32_t image = 0; image < n; ++image) {
    Sample sample;
    sample.label = read_scalar<std::uint8_t>(file);
    sample.expected_prediction = read_scalar<std::uint8_t>(file);
    if (sample.label >= outputs || sample.expected_prediction >= outputs) {
      throw std::runtime_error("Invalid MNIST target/prediction");
    }
    const auto bits = read_tensor<std::uint8_t>(file, std::size_t(steps) * inputs);
    sample.expected_hidden = read_tensor<std::uint8_t>(file, std::size_t(steps) * hidden);
    sample.expected_output = read_tensor<std::uint8_t>(file, std::size_t(steps) * outputs);
    sample.inputs_by_time.resize(steps, std::vector<float>(inputs));
    for (std::uint32_t t = 0; t < steps; ++t) {
      for (std::uint32_t j = 0; j < inputs; ++j) {
        const std::uint8_t bit = bits[std::size_t(t) * inputs + j];
        if (bit > 1) throw std::runtime_error("Non-binary input spike in trace");
        sample.inputs_by_time[t][j] = static_cast<float>(bit);
      }
    }
    workload.samples.push_back(std::move(sample));
  }
  return workload;
}

// Dispatch one layer to an existing kernel, without changing trained weights.
std::vector<std::uint8_t> run_layer(
    const std::string& mode, const snn::Layer& layer,
    const std::vector<float>& transposed, const std::vector<float>& spikes,
    snn::State& state) {
  if (mode == "dense") return snn::step(layer, spikes, state, snn::Execution::Dense);
  if (mode == "sparse") return snn::step(layer, spikes, state, snn::Execution::Sparse);
  if (mode == "rvv") return snn::step_rvv(layer, spikes, state);
  if (mode == "sparse_rvv") return snn::step_sparse_rvv(layer, transposed, spikes, state);
  if (mode == "sparse_rvv_fused") {
    return snn::step_sparse_rvv_fused(layer, transposed, spikes, state);
  }
  throw std::invalid_argument("Unknown inference mode: " + mode);
}

// Run complete image inference. Membrane states persist across timesteps and
// reset at the next image. Optional verification compares EVERY emitted spike.
Result infer(const Workload& w, const std::string& first_mode,
             const std::string& second_mode, bool verify) {
  Result result;
  for (const Sample& sample : w.samples) {
    snn::State first_state(w.first.outputs), second_state(w.second.outputs);
    std::vector<std::uint32_t> counts(w.second.outputs, 0);
    std::vector<float> hidden_input(w.first.outputs);

    for (std::uint32_t t = 0; t < w.steps; ++t) {
      const auto hidden = run_layer(first_mode, w.first, w.first_input_major,
                                    sample.inputs_by_time[t], first_state);
      for (std::size_t i = 0; i < hidden.size(); ++i) {
        result.hidden_spikes += hidden[i];
        if (verify && hidden[i] != sample.expected_hidden[t * hidden.size() + i]) {
          ++result.hidden_mismatches;
        }
        hidden_input[i] = static_cast<float>(hidden[i]);
      }
      const auto output = run_layer(second_mode, w.second, w.second_input_major,
                                    hidden_input, second_state);
      for (std::size_t i = 0; i < output.size(); ++i) {
        result.output_spikes += output[i];
        counts[i] += output[i];
        if (verify && output[i] != sample.expected_output[t * output.size() + i]) {
          ++result.output_mismatches;
        }
      }
    }

    const auto prediction = static_cast<std::uint8_t>(
        std::distance(counts.begin(), std::max_element(counts.begin(), counts.end())));
    if (prediction == sample.label) ++result.correct_predictions;
    if (verify && prediction != sample.expected_prediction) {
      ++result.prediction_mismatches;
    }
  }
  return result;
}

// Restrict benchmark modes explicitly to avoid accidental scalar fallback.
void validate_mode(const std::string& mode) {
  if (mode != "dense" && mode != "sparse" && mode != "rvv" &&
      mode != "sparse_rvv" && mode != "sparse_rvv_fused") {
    throw std::invalid_argument("Unsupported mode: " + mode);
  }
  if (mode != "dense" && mode != "sparse" && !snn::rvv_available()) {
    throw std::runtime_error("Selected kernel requires a RISC-V RVV build");
  }
}

}  // namespace

// Verify outside the ROI, then measure end-to-end inference with both layers.
int main(int argc, char** argv) {
  try {
    std::string trace, first_mode = "sparse_rvv_fused", second_mode = "sparse_rvv_fused";
    for (int i = 1; i < argc; ++i) {
      if (i + 1 >= argc) throw std::invalid_argument("Missing option value");
      const std::string key = argv[i++];
      const std::string value = argv[i];
      if (key == "--trace") trace = value;
      else if (key == "--mode") first_mode = second_mode = value;
      else if (key == "--first-mode") first_mode = value;
      else if (key == "--second-mode") second_mode = value;
      else throw std::invalid_argument("Unknown option: " + key);
    }
    if (trace.empty()) throw std::invalid_argument("Usage: snn_mnist --trace FILE [--mode MODE]");
    validate_mode(first_mode);
    validate_mode(second_mode);
    Workload work = read_workload(trace);

    // Transpose once per model load. Future measurements must account for
    // preparation costs if weights are frequently swapped or loaded anew.
    if (first_mode == "sparse_rvv" || first_mode == "sparse_rvv_fused") {
      work.first_input_major = snn::prepare_input_major_weights(work.first);
    }
    if (second_mode == "sparse_rvv" || second_mode == "sparse_rvv_fused") {
      work.second_input_major = snn::prepare_input_major_weights(work.second);
    }

    const Result checked = infer(work, first_mode, second_mode, true);
    std::cout << "reference_check="
              << (checked.hidden_mismatches == 0 && checked.output_mismatches == 0 &&
                  checked.prediction_mismatches == 0 ? "PASS" : "FAIL")
              << " hidden_spike_mismatches=" << checked.hidden_mismatches
              << " output_spike_mismatches=" << checked.output_mismatches
              << " prediction_mismatches=" << checked.prediction_mismatches << '\n';
    if (checked.hidden_mismatches || checked.output_mismatches || checked.prediction_mismatches) {
      return 3;
    }

#ifdef SNN_GEM5
    m5_reset_stats(0, 0);
#endif
    const Result measured = infer(work, first_mode, second_mode, false);
#ifdef SNN_GEM5
    m5_dump_stats(0, 0);
#endif

    std::cout << "first_mode=" << first_mode << " second_mode=" << second_mode
              << " samples=" << work.samples.size() << " timesteps=" << work.steps
              << " images_correct=" << measured.correct_predictions
              << " accuracy_on_subset=" << std::fixed << std::setprecision(6)
              << static_cast<double>(measured.correct_predictions) / work.samples.size()
              << " hidden_spike_checksum=" << measured.hidden_spikes
              << " output_spike_checksum=" << measured.output_spikes << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "SNN MNIST ERROR: " << error.what() << '\n';
    return 2;
  }
}
