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
find_library(TORCH_CPU_LIB NAMES torch_cpu PATHS "${LIBTORCH_LIB_DIR}" NO_DEFAULT_PATH)
find_library(C10_LIB NAMES c10 PATHS "${LIBTORCH_LIB_DIR}" NO_DEFAULT_PATH)

message(STATUS "LibTorch Include: ${LIBTORCH_INCLUDE_DIR}")
message(STATUS "LibTorch Lib: ${LIBTORCH_LIB_DIR}")

message(STATUS "Torch CPU Library: ${TORCH_CPU_LIB}")
message(STATUS "Torch C10 Library: ${C10_LIB}")

# Setup Nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
message(STATUS "nanobind cmake directory: ${nanobind_ROOT}")

find_package(nanobind CONFIG REQUIRED)

set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openequivariance/extension")
set(EXT_BACKEND_DIR "${EXT_DIR}/backend")
set(EXT_JSON_DIR "${EXT_DIR}/json11")

# Source files
set(OEQ_SOURCES
    ${EXT_DIR}/libtorch_tp_jit_stable.cpp
    ${EXT_JSON_DIR}/json11.cpp
)

set(OEQ_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openequivariance/_torch/extlib")

function(add_stable_extension target_name backend_define link_libraries)
    # Create nanobind extension
    nanobind_add_module(${target_name} NB_STATIC ${OEQ_SOURCES})

    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Enforce CXX11 ABI to match LibTorch
    target_compile_definitions(${target_name} PRIVATE 
        ${backend_define}=1
        _GLIBCXX_USE_CXX11_ABI=1
        INCLUDE_NB_EXTENSION
    )
    
    target_include_directories(${target_name} PRIVATE
        ${EXT_DIR}
        ${EXT_BACKEND_DIR}
        ${EXT_JSON_DIR}
        ${LIBTORCH_INCLUDE_DIR}
    )
    target_link_libraries(${target_name} PRIVATE
        ${TORCH_CPU_LIB}
        ${C10_LIB}
        ${link_libraries}
    )
    
    install(TARGETS ${target_name} LIBRARY DESTINATION "${OEQ_INSTALL_DIR}")

    # AOTI C++ library (identical except without nanobind and without INCLUDE_NB_EXTENSION)
    set(aoti_target_name ${target_name}_aoti)
    add_library(${aoti_target_name} SHARED ${OEQ_SOURCES})

    set_target_properties(${aoti_target_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    target_compile_definitions(${aoti_target_name} PRIVATE 
        ${backend_define}=1
        _GLIBCXX_USE_CXX11_ABI=1
    )
    
    target_include_directories(${aoti_target_name} PRIVATE
        ${EXT_DIR}
        ${EXT_BACKEND_DIR}
        ${EXT_JSON_DIR}
        ${LIBTORCH_INCLUDE_DIR}
    )
    target_link_libraries(${aoti_target_name} PRIVATE
        ${TORCH_CPU_LIB}
        ${C10_LIB}
        ${link_libraries}
    )
    
    install(TARGETS ${aoti_target_name} LIBRARY DESTINATION "${OEQ_INSTALL_DIR}")
endfunction()

find_package(CUDAToolkit QUIET)
find_package(hip QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "Building stable extension with CUDA backend.")

    add_library(cuda_stub_lib SHARED ${EXT_DIR}/stubs/stream.cpp)

    target_include_directories(cuda_stub_lib PRIVATE
        ${LIBTORCH_INCLUDE_DIR}
    )

    set_target_properties(cuda_stub_lib PROPERTIES 
        OUTPUT_NAME "torch_cuda"  
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )

    set(CUDA_LINK_LIBS
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        cuda_stub_lib
    )
    add_stable_extension(oeq_stable_cuda CUDA_BACKEND "${CUDA_LINK_LIBS}")
endif()

if(hip_FOUND)
    message(STATUS "Building stable extension with HIP backend.")

    add_library(hip_stub_lib SHARED ${EXT_DIR}/stubs/stream.cpp)

    target_include_directories(hip_stub_lib PRIVATE
        ${LIBTORCH_INCLUDE_DIR}
    )

    set_target_properties(hip_stub_lib PROPERTIES 
        OUTPUT_NAME "torch_hip"  
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )

    set(HIP_LINK_LIBS
        hiprtc
        hip_stub_lib
    )
    add_stable_extension(torch_stable_hip HIP_BACKEND "${HIP_LINK_LIBS}")
endif()

if(NOT CUDAToolkit_FOUND AND NOT hip_FOUND)
    message(FATAL_ERROR "Neither CUDAToolkit nor HIP was found. Cannot build the stable extension.")
endif()