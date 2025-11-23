cmake_minimum_required(VERSION 3.15...3.30)
project(OEQ_JAX LANGUAGES CXX CUDA) # TODO: Add HIP support

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
          "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR
)

message(STATUS "XLA include directory: ${XLA_DIR}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)

find_package(nanobind CONFIG REQUIRED)

set(OEQ_JAX_SOURCES
    libjax_tp_jit.cpp
)

set(OEQ_JAX_HEADERS
    convolution.hpp
    tensorproducts.hpp
    util/backend_cuda.hpp
    util/backend_hip.hpp
    buffer.hpp
)

nanobind_add_module(OEQ_JAX NB_STATIC ${OEQ_JAX_SOURCES} ${OEQ_JAX_HEADERS})

target_include_directories(OEQ_JAX PUBLIC ${XLA_DIR})
set_target_properties(OEQ_JAX PROPERTIES CUDA_STANDARD 17 POSITION_INDEPENDENT_CODE ON)

install(TARGETS OEQ_JAX LIBRARY DESTINATION lib)