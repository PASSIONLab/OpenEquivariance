cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX) # TODO: Add HIP support

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)

include(ExternalProject)
ExternalProject_Add(
    xla 
    PREFIX ${CMAKE_BINARY_DIR}/xla
    GIT_REPOSITORY https://github.com/openxla/xla.git
    GIT_TAG main 
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    )
ExternalProject_Get_Property(xla source_dir)
set(XLA_DIR ${source_dir})
message(STATUS "XLA include directory: ${XLA_DIR}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
message(STATUS "nanobind cmake directory: ${nanobind_ROOT}")
set(HEADER_DIR "src/extension")

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

get_target_property(CUDA_LIB_DIR CUDA::nvrtc IMPORTED_LOCATION)
get_filename_component(CUDA_LIB_DIR ${CUDA_LIB_DIR} DIRECTORY)

add_dependencies(openequivariance_extjax xla)

set_target_properties(openequivariance_extjax PROPERTIES
    BUILD_RPATH "${CUDA_LIB_DIR}"
    INSTALL_RPATH "${CUDA_LIB_DIR}"
)

target_link_libraries(openequivariance_extjax PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc)

install(TARGETS openequivariance_extjax LIBRARY DESTINATION .)

# -------------------------------------------------------------

# Uncomment to include XLA from JAX installation 
#execute_process(
#  COMMAND "${Python_EXECUTABLE}" "-c"
#          "from jax import ffi; print(ffi.include_dir())"
#  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR
#)
#message(STATUS "XLA include directory: ${XLA_DIR}")
