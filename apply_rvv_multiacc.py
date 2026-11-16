#!/usr/bin/env python3
"""Add optional 2/4-chain RVV sparse accumulation and input-major MNIST traces.

Run at the root of the HiPEAC-22 checkout from commit d40dea5 or later.
Existing sparse_rvv_fused and SNNMNST1 traces remain supported unchanged.
Refuses ambiguous patches and never partially writes on failed pattern checks.
"""

from pathlib import Path

FILES = [
    Path("include/snn/kernels.hpp"),
    Path("src/kernels.cpp"),
    Path("src/mnist_inference.cpp"),
    Path("python/export_mnist_trace.py"),
    Path("tests/test_kernels.cpp"),
]


def once(source: str, old: str, new: str, filename: str) -> str:
    """Replace one exact anchor so a stale checkout cannot be patched silently."""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{filename}: expected one instance of {old[:85]!r}, found {count}")
    return source.replace(old, new, 1)


def patch() -> None:
    """Stage all modifications in memory, validate them, then write files."""
    old = {str(path): path.read_text() for path in FILES}
    if "step_sparse_rvv_fused_multi(" in old["include/snn/kernels.hpp"]:
        raise RuntimeError("Multi-accumulator patch already installed; no files changed")
    new = dict(old)

    h = "include/snn/kernels.hpp"
    new[h] = once(new[h],
        "bool rvv_available();",
        """// Independent accumulation chains reduce the dependency depth of spike-driven
// synaptic integration. Only 2 or 4 are valid; 1 uses the original baseline.
// Floating-point association differs from the single-accumulator baseline.
std::vector<std::uint8_t> step_sparse_rvv_fused_multi(
    const Layer& layer, const std::vector<float>& input_major_weights,
    const std::vector<float>& input_spikes, State& state, int accumulators);

bool rvv_available();""", h)

    src = "src/kernels.cpp"
    kernel = r'''// Use K independent accumulators instead of one serial vector-add chain.
// Existing single-accumulator fused kernel remains unchanged as the baseline.
// Each accumulator processes a different active input's outgoing weights.
#if defined(__riscv_vector)
template <int K>
std::vector<std::uint8_t> fused_multi_impl(
    const Layer& layer, const std::vector<float>& input_major_weights,
    const std::vector<float>& input_spikes, State& state) {
  static_assert(K == 2 || K == 4);
  validate_input(layer, input_spikes, state);
  if (input_major_weights.size() != layer.weights.size()) {
    throw std::invalid_argument("Input-major weight shape mismatch");
  }

  // Identical spike extraction and LIF dynamics to the fused baseline.
  std::vector<std::size_t> active;
  active.reserve(layer.inputs);
  for (std::size_t j = 0; j < layer.inputs; ++j) {
    if (input_spikes[j] == 1.0F) active.push_back(j);
  }

  std::vector<std::uint8_t> output(layer.outputs);
  for (std::size_t i = 0; i < layer.outputs;) {
    const std::size_t vl = __riscv_vsetvl_e32m1(layer.outputs - i);
    vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0F, vl);
    vfloat32m1_t a1 = __riscv_vfmv_v_f_f32m1(0.0F, vl);
    vfloat32m1_t a2 = __riscv_vfmv_v_f_f32m1(0.0F, vl);
    vfloat32m1_t a3 = __riscv_vfmv_v_f_f32m1(0.0F, vl);

    // Unroll by K; each add depends only on its own accumulator.
    std::size_t a = 0;
    for (; a + K <= active.size(); a += K) {
      const vfloat32m1_t w0 = __riscv_vle32_v_f32m1(
          input_major_weights.data() + active[a] * layer.outputs + i, vl);
      a0 = __riscv_vfadd_vv_f32m1(a0, w0, vl);
      const vfloat32m1_t w1 = __riscv_vle32_v_f32m1(
          input_major_weights.data() + active[a + 1] * layer.outputs + i, vl);
      a1 = __riscv_vfadd_vv_f32m1(a1, w1, vl);
      if constexpr (K == 4) {
        const vfloat32m1_t w2 = __riscv_vle32_v_f32m1(
            input_major_weights.data() + active[a + 2] * layer.outputs + i, vl);
        a2 = __riscv_vfadd_vv_f32m1(a2, w2, vl);
        const vfloat32m1_t w3 = __riscv_vle32_v_f32m1(
            input_major_weights.data() + active[a + 3] * layer.outputs + i, vl);
        a3 = __riscv_vfadd_vv_f32m1(a3, w3, vl);
      }
    }
    // Handle remaining 0..K-1 active inputs; no contributions are discarded.
    for (; a < active.size(); ++a) {
      const vfloat32m1_t w = __riscv_vle32_v_f32m1(
          input_major_weights.data() + active[a] * layer.outputs + i, vl);
      a0 = __riscv_vfadd_vv_f32m1(a0, w, vl);
    }

    // Combining chains changes FP summation order; verify spikes externally.
    vfloat32m1_t current = __riscv_vfadd_vv_f32m1(a0, a1, vl);
    if constexpr (K == 4) {
      const vfloat32m1_t a23 = __riscv_vfadd_vv_f32m1(a2, a3, vl);
      current = __riscv_vfadd_vv_f32m1(current, a23, vl);
    }

    // Same fused LIF update, threshold, hard reset, and uint8 output encoding.
    const vfloat32m1_t potential = __riscv_vle32_v_f32m1(
        state.voltage.data() + i, vl);
    const vfloat32m1_t bias = __riscv_vle32_v_f32m1(
        layer.bias.data() + i, vl);
    vfloat32m1_t voltage = __riscv_vfmul_vf_f32m1(
        potential, layer.beta, vl);
    voltage = __riscv_vfadd_vv_f32m1(voltage, current, vl);
    voltage = __riscv_vfadd_vv_f32m1(voltage, bias, vl);

    const vbool32_t fired = __riscv_vmfge_vf_f32m1_b32(
        voltage, layer.threshold, vl);
    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0F, vl);
    const vfloat32m1_t next = __riscv_vmerge_vvm_f32m1(
        voltage, zero, fired, vl);
    __riscv_vse32_v_f32m1(state.voltage.data() + i, next, vl);

    const vuint32m1_t zeros = __riscv_vmv_v_x_u32m1(0, vl);
    const vuint32m1_t ones = __riscv_vmv_v_x_u32m1(1, vl);
    const vuint32m1_t spikes32 = __riscv_vmerge_vvm_u32m1(
        zeros, ones, fired, vl);
    const vuint16mf2_t spikes16 = __riscv_vnsrl_wx_u16mf2(
        spikes32, 0, vl);
    const vuint8mf4_t spikes8 = __riscv_vnsrl_wx_u8mf4(
        spikes16, 0, vl);
    __riscv_vse8_v_u8mf4(output.data() + i, spikes8, vl);
    i += vl;
  }
  return output;
}
#endif

// Public entry point; 1 defers to the original unmodified fused baseline.
std::vector<std::uint8_t> step_sparse_rvv_fused_multi(
    const Layer& layer, const std::vector<float>& input_major_weights,
    const std::vector<float>& input_spikes, State& state, int accumulators) {
  if (accumulators == 1) {
    return step_sparse_rvv_fused(layer, input_major_weights, input_spikes, state);
  }
#if defined(__riscv_vector)
  if (accumulators == 2) {
    return fused_multi_impl<2>(layer, input_major_weights, input_spikes, state);
  }
  if (accumulators == 4) {
    return fused_multi_impl<4>(layer, input_major_weights, input_spikes, state);
  }
#else
  (void)layer;
  (void)input_major_weights;
  (void)input_spikes;
  (void)state;
#endif
  throw std::invalid_argument("RVV multi-accumulator count must be 1, 2 or 4");
}

'''
    new[src] = once(new[src], "bool rvv_available() {", kernel + "bool rvv_available() {", src)

    cpp = "src/mnist_inference.cpp"
    new[cpp] = once(new[cpp],
        "  std::uint32_t steps = 0;\n};",
        "  std::uint32_t steps = 0;\n  bool stored_input_major = false; // SNNMNST2 traces contain input-major weights.\n};", cpp)
    new[cpp] = once(new[cpp],
        '''  if (!file.read(magic.data(), magic.size()) ||
      std::memcmp(magic.data(), "SNNMNST1", 8) != 0) {
    throw std::runtime_error("Not an SNNMNST1 trace");
  }''',
        '''  if (!file.read(magic.data(), magic.size())) {
    throw std::runtime_error("Cannot read MNIST trace magic");
  }
  const bool stored_input_major = std::memcmp(magic.data(), "SNNMNST2", 8) == 0;
  if (!stored_input_major && std::memcmp(magic.data(), "SNNMNST1", 8) != 0) {
    throw std::runtime_error("Expected SNNMNST1 or SNNMNST2 trace");
  }''', cpp)
    new[cpp] = once(new[cpp],
        '''  Workload workload;
  workload.steps = steps;''',
        '''  Workload workload;
  workload.steps = steps;
  workload.stored_input_major = stored_input_major;''', cpp)
    new[cpp] = once(new[cpp],
        '''  if (mode == "sparse_rvv_fused") {
    return snn::step_sparse_rvv_fused(layer, transposed, spikes, state);
  }''',
        '''  if (mode == "sparse_rvv_fused") {
    return snn::step_sparse_rvv_fused(layer, transposed, spikes, state);
  }
  if (mode == "sparse_rvv_fused_acc2") {
    return snn::step_sparse_rvv_fused_multi(layer, transposed, spikes, state, 2);
  }
  if (mode == "sparse_rvv_fused_acc4") {
    return snn::step_sparse_rvv_fused_multi(layer, transposed, spikes, state, 4);
  }''', cpp)
    new[cpp] = once(new[cpp],
        '''    std::vector<float> hidden_input(w.first.outputs);

    for (std::uint32_t t = 0; t < w.steps; ++t) {
      const auto hidden = run_layer(first_mode, w.first, w.first_input_major,
                                    sample.inputs_by_time[t], first_state);''',
        '''    std::vector<float> hidden_input(w.first.outputs);
    // For input-major traces, use the loaded weight buffer directly: no
    // transpose and no duplicated weight matrix in memory.
    const auto& first_vector_weights = w.stored_input_major
        ? w.first.weights : w.first_input_major;
    const auto& second_vector_weights = w.stored_input_major
        ? w.second.weights : w.second_input_major;

    for (std::uint32_t t = 0; t < w.steps; ++t) {
      const auto hidden = run_layer(first_mode, w.first, first_vector_weights,
                                    sample.inputs_by_time[t], first_state);''', cpp)
    new[cpp] = once(new[cpp],
        '''      const auto output = run_layer(second_mode, w.second, w.second_input_major,
                                    hidden_input, second_state);''',
        '''      const auto output = run_layer(second_mode, w.second, second_vector_weights,
                                    hidden_input, second_state);''', cpp)
    new[cpp] = once(new[cpp],
        '''      mode != "sparse_rvv" && mode != "sparse_rvv_fused") {''',
        '''      mode != "sparse_rvv" && mode != "sparse_rvv_fused" &&
      mode != "sparse_rvv_fused_acc2" && mode != "sparse_rvv_fused_acc4") {''', cpp)
    new[cpp] = once(new[cpp],
        '''    Workload work = read_workload(trace);

    // Transpose once per model load.''',
        '''    Workload work = read_workload(trace);
    // SNNMNST2 stores weights in input-major order. Legacy scalar/dense
    // kernels expect output-major and are deliberately disallowed here.
    if (work.stored_input_major) {
      const auto supports_input_major = [](const std::string& mode) {
        return mode == "sparse_rvv" || mode == "sparse_rvv_fused" ||
               mode == "sparse_rvv_fused_acc2" ||
               mode == "sparse_rvv_fused_acc4";
      };
      if (!supports_input_major(first_mode) || !supports_input_major(second_mode)) {
        throw std::invalid_argument(
            "Input-major traces require a sparse RVV kernel; use SNNMNST1 for scalar/dense");
      }
    }

    // Transpose once per model load.''', cpp)
    new[cpp] = once(new[cpp],
        '''    if (first_mode == "sparse_rvv" || first_mode == "sparse_rvv_fused") {''',
        '''    if (!work.stored_input_major &&
        (first_mode == "sparse_rvv" || first_mode == "sparse_rvv_fused" ||
         first_mode == "sparse_rvv_fused_acc2" ||
         first_mode == "sparse_rvv_fused_acc4")) {''', cpp)
    new[cpp] = once(new[cpp],
        '''    if (second_mode == "sparse_rvv" || second_mode == "sparse_rvv_fused") {''',
        '''    if (!work.stored_input_major &&
        (second_mode == "sparse_rvv" || second_mode == "sparse_rvv_fused" ||
         second_mode == "sparse_rvv_fused_acc2" ||
         second_mode == "sparse_rvv_fused_acc4")) {''', cpp)
    new[cpp] = once(new[cpp],
        '''    std::cout << "first_mode=" << first_mode << " second_mode=" << second_mode''',
        '''    std::cout << "weight_layout=" <<
        (work.stored_input_major ? "pretransposed_input_major" : "original_output_major")
              << " first_mode=" << first_mode << " second_mode=" << second_mode''', cpp)

    py = "python/export_mnist_trace.py"
    new[py] = once(new[py],
        '''    parser.add_argument("--seed", type=int, default=20261115)''',
        '''    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--input-major", action="store_true",
                        help="Store weights pre-transposed, using SNNMNST2")''', py)
    new[py] = once(new[py],
        '''        file.write(struct.pack("<8sIIIIIff", MAGIC, args.samples, steps, 784, hidden, 10, beta, threshold))
        write_float32(file, model.fc1.weight)
        write_float32(file, model.fc1.bias)
        write_float32(file, model.fc2.weight)
        write_float32(file, model.fc2.bias)''',
        '''        magic = b"SNNMNST2" if args.input_major else MAGIC
        file.write(struct.pack("<8sIIIIIff", magic, args.samples, steps, 784, hidden, 10, beta, threshold))
        # Only the storage order changes: model weights and spike oracle are identical.
        write_float32(file, model.fc1.weight.T if args.input_major else model.fc1.weight)
        write_float32(file, model.fc1.bias)
        write_float32(file, model.fc2.weight.T if args.input_major else model.fc2.weight)
        write_float32(file, model.fc2.bias)''', py)
    new[py] = once(new[py],
        '''        "output": str(target), "samples": args.samples, "timesteps": steps,''',
        '''        "output": str(target), "samples": args.samples, "timesteps": steps,
        "weight_layout": "input_major" if args.input_major else "output_major",''', py)

    tst = "tests/test_kernels.cpp"
    new[tst] = once(new[tst],
        '''      snn::State fused_rvv_state(outputs);''',
        '''      snn::State fused_rvv_state(outputs);
      snn::State acc2_rvv_state(outputs), acc4_rvv_state(outputs);''', tst)
    new[tst] = once(new[tst],
        '''          const auto fused = snn::step_sparse_rvv_fused(
              layer, input_major, in, fused_rvv_state);''',
        '''          const auto fused = snn::step_sparse_rvv_fused(
              layer, input_major, in, fused_rvv_state);
          const auto acc2 = snn::step_sparse_rvv_fused_multi(
              layer, input_major, in, acc2_rvv_state, 2);
          const auto acc4 = snn::step_sparse_rvv_fused_multi(
              layer, input_major, in, acc4_rvv_state, 4);''', tst)
    new[tst] = once(new[tst],
        '''          require(fused == sr, "Fused/unfused sparse RVV mismatch");''',
        '''          require(fused == sr, "Fused/unfused sparse RVV mismatch");
          require(acc2 == expected, "2-accumulator RVV emitted different spikes");
          require(acc4 == expected, "4-accumulator RVV emitted different spikes");''', tst)
    new[tst] = once(new[tst],
        '''            require(std::fabs(fused_rvv_state.voltage[i] - sparse_rvv_state.voltage[i]) <= 1e-5F,
                    "Fused/unfused RVV voltage mismatch");''',
        '''            require(std::fabs(fused_rvv_state.voltage[i] - sparse_rvv_state.voltage[i]) <= 1e-5F,
                    "Fused/unfused RVV voltage mismatch");
            require(std::fabs(acc2_rvv_state.voltage[i] - independent[i]) <= 5e-5F,
                    "2-accumulator RVV voltage mismatch");
            require(std::fabs(acc4_rvv_state.voltage[i] - independent[i]) <= 5e-5F,
                    "4-accumulator RVV voltage mismatch");''', tst)

    for path in FILES:
        path.write_text(new[str(path)])
        print(f"Updated {path}")
    print("Patch complete: new MNIST modes sparse_rvv_fused_acc2/acc4; new --input-major trace flag")


if __name__ == "__main__":
    patch()
