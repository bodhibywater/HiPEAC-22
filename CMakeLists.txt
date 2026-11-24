cmake_minimum_required(VERSION 3.18)
project(hipeac_snn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(snn_kernels STATIC src/kernels.cpp src/temporal.cpp)
target_include_directories(snn_kernels PUBLIC include)
target_compile_options(snn_kernels PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)

add_executable(snn_test tests/test_kernels.cpp)
target_link_libraries(snn_test PRIVATE snn_kernels)
target_compile_options(snn_test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(snn_bench src/bench.cpp)
target_link_libraries(snn_bench PRIVATE snn_kernels)
target_compile_options(snn_bench PRIVATE -O3 -Wall -Wextra -Wpedantic -fno-fast-math)

include(CTest)
add_test(NAME cpp_kernel_correctness COMMAND snn_test)

option(SNN_GEM5 "Enable gem5 ROI instrumentation" OFF)

if(SNN_GEM5)
  set(GEM5_ROOT "" CACHE PATH "Path to gem5 source tree")
  set(M5_LIBRARY "${GEM5_ROOT}/util/m5/build/riscv/out/libm5.a")

  if(NOT EXISTS "${GEM5_ROOT}/include/gem5/m5ops.h"
     OR NOT EXISTS "${M5_LIBRARY}")
    message(FATAL_ERROR "gem5 headers or RISC-V libm5.a not found")
  endif()

  target_compile_definitions(snn_bench PRIVATE SNN_GEM5=1)
  target_include_directories(snn_bench PRIVATE "${GEM5_ROOT}/include")
  target_link_libraries(snn_bench PRIVATE "${M5_LIBRARY}")
endif()

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
