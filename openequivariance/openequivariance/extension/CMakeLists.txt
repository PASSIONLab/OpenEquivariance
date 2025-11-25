cmake_minimum_required(VERSION 3.15...3.30)
project(oeq_jax_extension LANGUAGES CXX CUDA) # TODO: Add HIP support

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
    util/buffer.hpp
)

nanobind_add_module(oeq_jax_extension NB_STATIC ${OEQ_JAX_SOURCES} ${OEQ_JAX_HEADERS})

target_include_directories(oeq_jax_extension PUBLIC ${XLA_DIR})
set_target_properties(oeq_jax_extension PROPERTIES CUDA_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_compile_options(oeq_jax_extension PRIVATE -Wno-attributes -Wno-return-type)

install(TARGETS oeq_jax_extension LIBRARY DESTINATION lib)