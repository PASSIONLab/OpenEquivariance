cmake_minimum_required(VERSION 3.15...3.30)
project(openequivariance_stable_ext LANGUAGES CXX)

set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip")
set(LIBTORCH_ZIP "${CMAKE_BINARY_DIR}/libtorch-shared-with-deps-2.10.0+cpu.zip")
set(LIBTORCH_EXTRACT_DIR "${CMAKE_BINARY_DIR}/libtorch_extract")
set(LIBTORCH_INCLUDE_DIR "${LIBTORCH_EXTRACT_DIR}/libtorch/include")

file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ZIP} SHOW_PROGRESS)
file(MAKE_DIRECTORY ${LIBTORCH_EXTRACT_DIR})
file(ARCHIVE_EXTRACT
    INPUT ${LIBTORCH_ZIP}
    DESTINATION ${LIBTORCH_EXTRACT_DIR}
    PATTERNS "libtorch/include/*"
)

add_custom_target(libtorch_headers ALL DEPENDS ${LIBTORCH_INCLUDE_DIR})

set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openequivariance/extension")
set(EXT_UTIL_DIR "${EXT_DIR}/util")
set(EXT_JSON_DIR "${EXT_DIR}/json11")

set(OEQ_SOURCES
    ${EXT_DIR}/libtorch_tp_jit_stable.cpp
    ${EXT_JSON_DIR}/json11.cpp
)

function(add_stable_extension target_name backend_define)
    add_library(${target_name} MODULE ${OEQ_SOURCES})
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_definitions(${target_name} PRIVATE ${backend_define}=1)
    target_include_directories(${target_name} PRIVATE
        ${LIBTORCH_INCLUDE_DIR}
        ${EXT_DIR}
        ${EXT_UTIL_DIR}
        ${EXT_JSON_DIR}
    )
    add_dependencies(${target_name} libtorch_headers)
endfunction()

find_package(CUDAToolkit QUIET)
find_package(hip QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "Building stable extension with CUDA backend.")
    add_stable_extension(libtorch_tp_jit_stable_cuda CUDA_BACKEND)
    target_link_libraries(libtorch_tp_jit_stable_cuda PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

if(hip_FOUND)
    message(STATUS "Building stable extension with HIP backend.")
    add_stable_extension(libtorch_tp_jit_stable_hip HIP_BACKEND)
    target_link_libraries(libtorch_tp_jit_stable_hip PRIVATE hiprtc)
endif()

if(NOT CUDAToolkit_FOUND AND NOT hip_FOUND)
    message(FATAL_ERROR "Neither CUDAToolkit nor HIP was found. Cannot build the stable extension.")
endif()
