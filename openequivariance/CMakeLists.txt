cmake_minimum_required(VERSION 3.15...3.30)
project(openequivariance_stable_ext LANGUAGES CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Download LibTorch 
include(FetchContent)

FetchContent_Declare(
    libtorch
    URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip"
)

message(STATUS "Downloading LibTorch...")
FetchContent_MakeAvailable(libtorch)

set(LIBTORCH_INCLUDE_DIR "${libtorch_SOURCE_DIR}/include")
set(LIBTORCH_LIB_DIR "${libtorch_SOURCE_DIR}/lib")

message(STATUS "LibTorch Include: ${LIBTORCH_INCLUDE_DIR}")
message(STATUS "LibTorch Lib: ${LIBTORCH_LIB_DIR}")

# Setup Nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
message(STATUS "nanobind cmake directory: ${nanobind_ROOT}")

find_package(nanobind CONFIG REQUIRED)

set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openequivariance/extension")
set(EXT_UTIL_DIR "${EXT_DIR}/util")
set(EXT_JSON_DIR "${EXT_DIR}/json11")

# Source files
set(OEQ_SOURCES
    ${EXT_DIR}/libtorch_tp_jit_stable.cpp
    ${EXT_JSON_DIR}/json11.cpp
)

function(add_stable_extension target_name backend_define)
    nanobind_add_module(${target_name} NB_STATIC ${OEQ_SOURCES})

    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    # CRITICAL: Manually enforce CXX11 ABI to match the downloaded LibTorch
    target_compile_definitions(${target_name} PRIVATE 
        ${backend_define}=1
        _GLIBCXX_USE_CXX11_ABI=1 
    )
    
    target_include_directories(${target_name} PRIVATE
        ${EXT_DIR}
        ${EXT_UTIL_DIR}
        ${EXT_JSON_DIR}
        ${LIBTORCH_INCLUDE_DIR}
        ${LIBTORCH_API_INCLUDE_DIR}
    )    
endfunction()

find_package(CUDAToolkit QUIET)
find_package(hip QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "Building stable extension with CUDA backend.")
    add_stable_extension(torch_tp_jit_stable_cuda CUDA_BACKEND)
    target_link_libraries(torch_tp_jit_stable_cuda PRIVATE CUDA::cudart CUDA::cuda_driver)
    
    install(TARGETS torch_tp_jit_stable_cuda LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/openequivariance/_torch/extlib")
endif()

if(hip_FOUND)
    message(STATUS "Building stable extension with HIP backend.")
    add_stable_extension(torch_tp_jit_stable_hip HIP_BACKEND)
    target_link_libraries(torch_tp_jit_stable_hip PRIVATE hiprtc)

    install(TARGETS torch_tp_jit_stable_hip LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/openequivariance/_torch/extlib")
endif()

if(NOT CUDAToolkit_FOUND AND NOT hip_FOUND)
    message(FATAL_ERROR "Neither CUDAToolkit nor HIP was found. Cannot build the stable extension.")
endif()