#define USE_CUDA

#include <array>
#include <cstdint>
#include <optional>
#include <vector>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/tensor_inl.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#ifdef SYCL_BACKEND
    // Declared in shim_xpu.h behind USE_XPU; declared here so the SYCL build
    // does not have to define that macro as well as SYCL_BACKEND.
    extern "C" AOTITorchError aoti_torch_get_current_sycl_queue(void** ret_queue);
#endif

#ifdef CUDA_BACKEND
    #include "backend_cuda.hpp"
#endif

#ifdef HIP_BACKEND
    #include "backend_hip.hpp"
#endif

#ifdef SYCL_BACKEND
    #include "backend_sycl.hpp"
#endif

#define TCHECK STD_TORCH_CHECK
#define BOX(x) TORCH_BOX(x)
#define REGISTER_LIBRARY_IMPL STABLE_TORCH_LIBRARY_IMPL
#define REGISTER_LIBRARY STABLE_TORCH_LIBRARY

#ifndef STABLE_TORCH_ERROR_CODE_CHECK
#define STABLE_TORCH_ERROR_CODE_CHECK(call) TORCH_ERROR_CODE_CHECK(call)
#endif

namespace oeq {

using Tensor = torch::stable::Tensor;
using Dtype = torch::headeronly::ScalarType;

constexpr Dtype kFloat = torch::headeronly::ScalarType::Float;
constexpr Dtype kDouble = torch::headeronly::ScalarType::Double;
constexpr Dtype kInt = torch::headeronly::ScalarType::Int;
constexpr Dtype kLong = torch::headeronly::ScalarType::Long;
constexpr Dtype kByte = torch::headeronly::ScalarType::Byte;

} // namespace oeq

#include "torch_core.hpp"

namespace oeq {

Tensor as_strided(const Tensor& tensor, std::array<int64_t, 3> sizes,
        std::array<int64_t, 3> strides, int64_t storage_offset) {
    auto sizes_ref = torch::headeronly::IntHeaderOnlyArrayRef(sizes.data(), sizes.size());
    auto strides_ref = torch::headeronly::IntHeaderOnlyArrayRef(strides.data(), strides.size());
    std::array<StableIValue, 4> stack{
        torch::stable::detail::from(tensor),
        torch::stable::detail::from(sizes_ref),
        torch::stable::detail::from(strides_ref),
        torch::stable::detail::from(std::optional<int64_t>(storage_offset))};
    STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
        "aten::as_strided", "", stack.data(), TORCH_ABI_VERSION));
    return torch::stable::detail::to<Tensor>(stack[0]);
}

// Adapted from PyTorch PR #197441, commit
// 646e5d175800d0cce119e9d8a92bd5d79bb926d1, torch/csrc/stable/ops.h:
// https://github.com/pytorch/pytorch/pull/197441
Tensor& bmm_out(Tensor& out, const Tensor& self, const Tensor& mat2) {
    std::array<StableIValue, 3> stack{
        torch::stable::detail::from(self),
        torch::stable::detail::from(mat2),
        torch::stable::detail::from(out)};
    STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
        "aten::bmm", "out", stack.data(), TORCH_ABI_VERSION));
    (void)torch::stable::detail::to<Tensor>(stack[0]);
    return out;
}

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
    void* stream_ptr = nullptr;

    #ifdef SYCL_BACKEND
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_sycl_queue(&stream_ptr));
    #else
        auto device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
    #endif

    return static_cast<Stream>(stream_ptr);
}

bool tensor_is_on_gpu(const Tensor &tensor) {
    #ifdef SYCL_BACKEND
        // The stable Tensor has no is_xpu().
        int32_t device_type;
        TORCH_ERROR_CODE_CHECK(
            aoti_torch_get_device_type(tensor.get(), &device_type));
        return device_type == aoti_torch_device_type_xpu();
    #else
        return tensor.is_cuda();
    #endif
}

} // namespace oeq

#ifdef CUDA_BACKEND
    #define EXTENSION_NAME oeq_stable_cuda
#endif
#ifdef HIP_BACKEND
    #define EXTENSION_NAME oeq_stable_hip
#endif
#ifdef SYCL_BACKEND
    #define EXTENSION_NAME oeq_stable_sycl
#endif 

#ifdef INCLUDE_NB_EXTENSION
    #include "nanobind/nanobind.h"
    #include "nanobind/stl/string.h"
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
#endif
