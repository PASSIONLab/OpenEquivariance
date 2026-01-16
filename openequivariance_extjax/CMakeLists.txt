cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# --- XLA CONFIGURATION ---
if(XLA_DIRECT_DOWNLOAD)
    message(STATUS "XLA_DIRECT_DOWNLOAD is ON. Fetching XLA source...")
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
else()
    message(STATUS "XLA_DIRECT_DOWNLOAD is OFF. Locating XLA via installed JAX...")
    execute_process(
      COMMAND "${Python_EXECUTABLE}" "-c"
              "from jax import ffi; print(ffi.include_dir())"
      OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR
    )
endif()

message(STATUS "XLA include directory: ${XLA_DIR}")
# -------------------------

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
message(STATUS "nanobind cmake directory: ${nanobind_ROOT}")

find_package(nanobind CONFIG REQUIRED)

if(JAX_HIP)
    message(STATUS "JAX_HIP is set. Building with HIP backend.")
    find_package(hip REQUIRED)
else()
    message(STATUS "JAX_HIP is not set (or zero). Building with CUDA backend.")
    find_package(CUDAToolkit REQUIRED)
endif()

set(HEADER_DIR "src/extension")
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
set_target_properties(openequivariance_extjax PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(openequivariance_extjax PRIVATE -Wno-attributes -Wno-return-type)

# Ensure the module waits for XLA download if we are in direct download mode
if(XLA_DIRECT_DOWNLOAD)
    add_dependencies(openequivariance_extjax xla)
endif()

if(JAX_HIP)
    target_link_libraries(openequivariance_extjax PRIVATE hiprtc)
    target_compile_definitions(openequivariance_extjax PRIVATE HIP_BACKEND=1)

else()
    set_target_properties(openequivariance_extjax PROPERTIES CUDA_STANDARD 17)

    get_target_property(CUDA_LIB_DIR CUDA::nvrtc IMPORTED_LOCATION)
    get_filename_component(CUDA_LIB_DIR ${CUDA_LIB_DIR} DIRECTORY)

    set_target_properties(openequivariance_extjax PROPERTIES
        BUILD_RPATH "${CUDA_LIB_DIR}"
        INSTALL_RPATH "${CUDA_LIB_DIR}"
    )

    target_link_libraries(openequivariance_extjax PRIVATE 
      CUDA::cudart
      CUDA::cuda_driver
      CUDA::nvrtc)
    target_compile_definitions(openequivariance_extjax PRIVATE CUDA_BACKEND=1)
endif()

install(TARGETS openequivariance_extjax LIBRARY DESTINATION .)