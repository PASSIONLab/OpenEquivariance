#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/tensor_inl.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include "nanobind/nanobind.h"

#ifdef HIP_BACKEND
    #include <c10/hip/HIPStream.h>
#endif

using Tensor = torch::stable::Tensor;
using Dtype = torch::headeronly::ScalarType;

constexpr Dtype kFloat = torch::headeronly::ScalarType::Float;
constexpr Dtype kDouble = torch::headeronly::ScalarType::Double;
constexpr Dtype kInt = torch::headeronly::ScalarType::Int;
constexpr Dtype kLong = torch::headeronly::ScalarType::Long;
constexpr Dtype kByte = torch::headeronly::ScalarType::Byte;

#define TCHECK STD_TORCH_CHECK
#define BOX(x) TORCH_BOX(x)
#define REGISTER_LIBRARY_IMPL STABLE_TORCH_LIBRARY_IMPL
#define REGISTER_LIBRARY STABLE_TORCH_LIBRARY

#include "torch_core.hpp"

Tensor tensor_to_cpu_contiguous(const Tensor &tensor) {
    torch::stable::Device device(torch::headeronly::DeviceType::CPU);
    return torch::stable::contiguous(torch::stable::to(tensor, device));
}

Tensor tensor_contiguous(const Tensor &tensor) {
    return torch::stable::contiguous(tensor);
}

Tensor tensor_empty_like(const Tensor &ref, const std::vector<int64_t> &sizes) {
    auto sizes_ref = torch::headeronly::IntHeaderOnlyArrayRef(sizes.data(), sizes.size());
    return torch::stable::new_empty(ref, sizes_ref);
}

Tensor tensor_zeros_like(const Tensor &ref, const std::vector<int64_t> &sizes) {
    auto sizes_ref = torch::headeronly::IntHeaderOnlyArrayRef(sizes.data(), sizes.size());
    Tensor out = torch::stable::new_empty(ref, sizes_ref);
    torch::stable::zero_(out);
    return out;
}

void tensor_zero_(Tensor &tensor) {
    torch::stable::zero_(tensor);
}

void alert_not_deterministic(const char *name) {
    (void)name;
}

const uint8_t *tensor_data_ptr_u8(const Tensor &tensor) {
    return static_cast<const uint8_t *>(tensor.data_ptr());
}

void *data_ptr(const Tensor &tensor) {
    return tensor.data_ptr();
}

Stream get_current_stream() {
    int32_t device_index;
    StreamOpaque* stream_ptr; 

    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_device_index(&device_index))
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_stream(device_index, &stream_ptr))
    //return (Stream) stream_ptr; 
    return (Stream) 0; 
}

#ifdef CUDA_BACKEND
    #define EXTENSION_NAME torch_tp_jit_stable_cuda
#endif
#ifdef HIP_BACKEND
    #define EXTENSION_NAME torch_tp_jit_stable_hip
#endif 

namespace nb = nanobind;
NB_MODULE(EXTENSION_NAME, m) {
    nb::class_<DeviceProp>(m, "DeviceProp")
        .def(nb::init<int>())
        .def_ro("name", &DeviceProp::name)
        .def_ro("warpsize", &DeviceProp::warpsize)
        .def_ro("major", &DeviceProp::major)
        .def_ro("minor", &DeviceProp::minor)
        .def_ro("multiprocessorCount", &DeviceProp::multiprocessorCount)
        .def_ro("maxSharedMemPerBlock", &DeviceProp::maxSharedMemPerBlock); 

    nb::class_<GPUTimer>(m, "GPUTimer")
        .def(nb::init<>())
        .def("start", &GPUTimer::start)
        .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed)
        .def("clear_L2_cache", &GPUTimer::clear_L2_cache);
}