cmake_minimum_required(VERSION 3.15...3.30)
project(openequivariance_stable_ext)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

# Download LibTorch 
include(FetchContent)

if (AARCH64 OR CMAKE_SYSTEM_PROCESSOR MATCHES "(arm)|(ARM)|(aarch64)")
    message(STATUS "Targeting ARM architecture (AARCH64 override or processor match)")
    set(LIBTORCH_ARCH AARCH64)
else()
    message(STATUS "Did not match arm/aarch64, performing x86 install")
    set(LIBTORCH_ARCH X86_64)
endif()

set(LIBTORCH_URL "${LIBTORCH_URL_${LIBTORCH_ARCH}}")
set(LIBTORCH_SHA256 "${LIBTORCH_SHA256_${LIBTORCH_ARCH}}")
if(NOT LIBTORCH_URL OR NOT LIBTORCH_SHA256)
    message(FATAL_ERROR "LIBTORCH_URL / LIBTORCH_SHA256 not set")
endif()

FetchContent_Declare(
    libtorch
    URL "${LIBTORCH_URL}"
    URL_HASH SHA256=${LIBTORCH_SHA256}
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

set(OEQ_INSTALL_DIR "openequivariance/_torch/extlib")

function(add_stable_extension target_name backend_define link_libraries)
    nanobind_add_module(${target_name} BACKEND_MODULE nanobind_backend ${OEQ_SOURCES})

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
        EXTENSION_NAME=${target_name}
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

find_package(hip QUIET)
find_package(hiprtc QUIET)

if(NOT hip_FOUND AND DEFINED ENV{ROCM_PATH})
    find_path(HIP_INCLUDE_DIR hip/hip_runtime.h PATHS $ENV{ROCM_PATH}/include NO_DEFAULT_PATH)
    find_library(AMDHIP64_LIB NAMES amdhip64 libamdhip64.so.7 PATHS $ENV{ROCM_PATH}/lib NO_DEFAULT_PATH)
    find_library(HIPRTC_LIB NAMES hiprtc libhiprtc.so.7 PATHS $ENV{ROCM_PATH}/lib NO_DEFAULT_PATH)

    if(HIP_INCLUDE_DIR AND AMDHIP64_LIB AND HIPRTC_LIB)
        add_library(hip::host INTERFACE IMPORTED)
        target_include_directories(hip::host INTERFACE ${HIP_INCLUDE_DIR})
        target_compile_definitions(hip::host INTERFACE __HIP_PLATFORM_AMD__=1)
        target_link_libraries(hip::host INTERFACE ${AMDHIP64_LIB})

        add_library(hiprtc::hiprtc INTERFACE IMPORTED)
        target_link_libraries(hiprtc::hiprtc INTERFACE ${HIPRTC_LIB})

        set(hip_FOUND TRUE)
    endif()
endif()

set(OEQ_CUDA_ROOTS "$ENV{OEQ_CUDA_ROOTS}" CACHE STRING "CUDA toolkit roots to build against")
string(REPLACE " " ";" OEQ_CUDA_ROOTS "${OEQ_CUDA_ROOTS}")

set(OEQ_CUDA_FOUND FALSE)
if(OEQ_CUDA_ROOTS)
    set(OEQ_CUDA_FOUND TRUE)
else()
    find_package(CUDAToolkit QUIET)
    set(OEQ_CUDA_FOUND ${CUDAToolkit_FOUND})
endif()

if(OEQ_CUDA_FOUND)
    add_library(cuda_stub_lib SHARED ${EXT_DIR}/stubs/stream.cpp)

    target_include_directories(cuda_stub_lib PRIVATE
        ${LIBTORCH_INCLUDE_DIR}
    )

    set_target_properties(cuda_stub_lib PROPERTIES
        OUTPUT_NAME "torch_cuda"
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
endif()

if(OEQ_CUDA_FOUND AND NOT OEQ_CUDA_ROOTS)
    message(STATUS "Building stable extension with CUDA ${CUDAToolkit_VERSION} backend.")
    set(CUDA_LINK_LIBS
        CUDA::cudart
        CUDA::cuda_driver
        ${CUDA_nvrtc_LIBRARY}
        cuda_stub_lib
    )
    add_stable_extension(oeq_stable_cuda${CUDAToolkit_VERSION_MAJOR} CUDA_BACKEND "${CUDA_LINK_LIBS}")
endif()

foreach(cuda_root ${OEQ_CUDA_ROOTS})
    file(STRINGS "${cuda_root}/include/cuda.h" cuda_version_line REGEX "^#define CUDA_VERSION [0-9]+")
    if(NOT cuda_version_line)
        message(FATAL_ERROR "No CUDA toolkit found at ${cuda_root}")
    endif()
    string(REGEX REPLACE ".* ([0-9]+)$" "\\1" cuda_version "${cuda_version_line}")
    math(EXPR cuda_major "${cuda_version} / 1000")
    math(EXPR cuda_minor "(${cuda_version} % 1000) / 10")
    message(STATUS "Building stable extension with CUDA ${cuda_major}.${cuda_minor} backend from ${cuda_root}.")

    set(cuda_lib_dir "${cuda_root}/lib64")
    add_library(oeq_cuda${cuda_major}_toolkit INTERFACE)
    target_include_directories(oeq_cuda${cuda_major}_toolkit INTERFACE "${cuda_root}/include")
    target_link_libraries(oeq_cuda${cuda_major}_toolkit INTERFACE
        "${cuda_lib_dir}/libcudart.so"
        "${cuda_lib_dir}/libnvrtc.so"
        "${cuda_lib_dir}/stubs/libcuda.so"
    )

    set(CUDA_LINK_LIBS oeq_cuda${cuda_major}_toolkit cuda_stub_lib)
    add_stable_extension(oeq_stable_cuda${cuda_major} CUDA_BACKEND "${CUDA_LINK_LIBS}")
endforeach()

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
        hip_stub_lib
	    hip::host
        hiprtc::hiprtc
    )
    add_stable_extension(oeq_stable_hip HIP_BACKEND "${HIP_LINK_LIBS}")
endif()

set(OEQ_SYCL_FOUND FALSE)
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    set(OEQ_SYCL_FOUND TRUE)
endif()

if(OEQ_SYCL_FOUND)
    message(STATUS "Building stable extension with SYCL backend.")

    add_library(sycl_stub_lib SHARED ${EXT_DIR}/stubs/stream.cpp)

    target_include_directories(sycl_stub_lib PRIVATE
        ${LIBTORCH_INCLUDE_DIR}
    )

    set_target_properties(sycl_stub_lib PROPERTIES
        OUTPUT_NAME "torch_xpu"
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )

    target_compile_definitions(sycl_stub_lib PRIVATE SYCL_BACKEND=1)

    set(SYCL_LINK_LIBS
        sycl_stub_lib
        sycl
    )
    add_stable_extension(oeq_stable_sycl SYCL_BACKEND "${SYCL_LINK_LIBS}")

    foreach(tgt oeq_stable_sycl oeq_stable_sycl_aoti)
        target_compile_options(${tgt} PRIVATE -fsycl)
        target_link_options(${tgt} PRIVATE -fsycl)
    endforeach()
endif()

if(NOT OEQ_CUDA_FOUND AND NOT hip_FOUND AND NOT OEQ_SYCL_FOUND)
    message(FATAL_ERROR "None of CUDAToolkit, HIP or SYCL was found")
endif()
