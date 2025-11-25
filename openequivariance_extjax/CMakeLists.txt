cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA) # TODO: Add HIP support

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
message(STATUS "nanobind cmake directory: ${nanobind_ROOT}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
          "import openequivariance; print(openequivariance.extension_source_path())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE HEADER_DIR 
)
message(STATUS "OpenEquivariance extension source directory: ${HEADER_DIR}")

find_package(nanobind CONFIG REQUIRED)

set(OEQ_JAX_SOURCES
    src/libjax_tp_jit.cpp
)

set(OEQ_JAX_HEADERS
  ${HEADER_DIR}/convolution.hpp
  ${HEADER_DIR}/tensorproducts.hpp
  ${HEADER_DIR}/util/backend_cuda.hpp
  ${HEADER_DIR}/util/backend_hip.hpp
  ${HEADER_DIR}/util/buffer.hpp
)

nanobind_add_module(openequivariance_extjax NB_STATIC ${OEQ_JAX_SOURCES} ${OEQ_JAX_HEADERS})

target_include_directories(openequivariance_extjax PUBLIC ${XLA_DIR} ${HEADER_DIR})
set_target_properties(openequivariance_extjax PROPERTIES CUDA_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_compile_options(openequivariance_extjax PRIVATE -Wno-attributes -Wno-return-type)
install(TARGETS openequivariance_extjax LIBRARY DESTINATION .)