"""Register the new MNIST inference executable without disturbing old kernels."""
from pathlib import Path

cmake = Path("CMakeLists.txt")
text = cmake.read_text()
if "add_executable(snn_mnist " in text:
    print("snn_mnist target already present, unchanged")
else:
    if "add_library(snn_kernels STATIC" not in text:
        raise RuntimeError("Run from HiPEAC-22 root; snn_kernels target is missing")
    if "option(SNN_GEM5" not in text:
        raise RuntimeError("Existing gem5 instrumented benchmark configuration is missing")
    text += '''
# Full two-layer inference on deterministic trained MNIST spike traces.
# Use an independent executable so the synthetic microbench stays unchanged.
add_executable(snn_mnist src/mnist_inference.cpp)
target_link_libraries(snn_mnist PRIVATE snn_kernels)
target_compile_options(snn_mnist PRIVATE -O3 -Wall -Wextra -Wpedantic -fno-fast-math)
if(SNN_GEM5)
  target_compile_definitions(snn_mnist PRIVATE SNN_GEM5=1)
  target_include_directories(snn_mnist PRIVATE "${GEM5_ROOT}/include")
  target_link_libraries(snn_mnist PRIVATE "${M5_LIBRARY}")
endif()
'''
    cmake.write_text(text)
    print("Registered snn_mnist with existing SNN_GEM5 instrumentation")
