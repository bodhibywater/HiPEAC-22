cmake_minimum_required(VERSION 3.18)
project(hipeac_snn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(snn_kernels STATIC src/kernels.cpp)
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
