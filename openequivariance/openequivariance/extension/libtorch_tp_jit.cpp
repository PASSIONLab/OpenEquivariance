#include <cstdint>
#include <pybind11/pybind11.h>

#ifdef CUDA_BACKEND
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAGuard.h>
#endif

#ifdef HIP_BACKEND
    #include <ATen/hip/HIPContext.h>
    #include <c10/hip/HIPStream.h>
    #include <c10/hip/HIPGuard.h>
#endif

#include <ATen/Operators.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/all.h>
#include <torch/library.h>

using Tensor = torch::Tensor;
using Dtype = torch::Dtype;

constexpr Dtype kFloat = torch::kFloat;
constexpr Dtype kDouble = torch::kDouble;
constexpr Dtype kInt = torch::kInt;
constexpr Dtype kLong = torch::kLong;
constexpr Dtype kByte = torch::kByte;

#define TCHECK TORCH_CHECK
#define BOX(x) x
#define REGISTER_LIBRARY_IMPL TORCH_LIBRARY_IMPL
#define REGISTER_LIBRARY TORCH_LIBRARY

#ifdef CUDA_BACKEND
    using DeviceGuard = c10::cuda::CUDAGuard;
#endif
#ifdef HIP_BACKEND
    using DeviceGuard = c10::hip::HIPGuard;
#endif

#include "torch_core.hpp"

Tensor tensor_to_cpu_contiguous(const Tensor &tensor) {
    return tensor.to(torch::kCPU).contiguous();
}

Tensor tensor_contiguous(const Tensor &tensor) {
    return tensor.contiguous();
}

Tensor tensor_empty_like(const Tensor &ref, const std::vector<int64_t> &sizes) {
    return torch::empty(sizes, ref.options());
}

Tensor tensor_zeros_like(const Tensor &ref, const std::vector<int64_t> &sizes) {
    return torch::zeros(sizes, ref.options());
}

void tensor_zero_(Tensor &tensor) {
    tensor.zero_();
}

void alert_not_deterministic(const char *name) {
    at::globalContext().alertNotDeterministic(name);
}

const uint8_t *tensor_data_ptr_u8(const Tensor &tensor) {
    return tensor.data_ptr<uint8_t>();
}

void *data_ptr(const Tensor &tensor) {
    if (tensor.dtype() == torch::kFloat)
        return reinterpret_cast<void *>(tensor.data_ptr<float>());
    else if (tensor.dtype() == torch::kDouble)
        return reinterpret_cast<void *>(tensor.data_ptr<double>());
    else if (tensor.dtype() == torch::kLong)
        return reinterpret_cast<void *>(tensor.data_ptr<int64_t>());
    else if (tensor.dtype() == torch::kByte)
        return reinterpret_cast<void *>(tensor.data_ptr<uint8_t>());
    else if (tensor.dtype() == torch::kInt)
        return reinterpret_cast<void *>(tensor.data_ptr<int32_t>());
    else
        throw std::logic_error("Unsupported tensor datatype!");
}

Stream get_current_stream(int32_t device_index) {
#ifdef CUDA_BACKEND
    return c10::cuda::getCurrentCUDAStream(device_index);
#endif
#ifdef HIP_BACKEND
    return c10::hip::getCurrentHIPStream(device_index);
#endif
}

BlasHandleT get_op_blas_handle(int32_t device_index, BlasStream stream) {
    (void)device_index;
    (void)stream;
    return at::cuda::getCurrentCUDABlasHandle();
}

namespace py=pybind11;
PYBIND11_MODULE(libtorch_tp_jit, m) {
    py::class_<DeviceProp>(m, "DeviceProp")
        .def(py::init<int>())
        .def_readonly("name", &DeviceProp::name)
        .def_readonly("warpsize", &DeviceProp::warpsize)
        .def_readonly("major", &DeviceProp::major)
        .def_readonly("minor", &DeviceProp::minor)
        .def_readonly("multiprocessorCount", &DeviceProp::multiprocessorCount)
        .def_readonly("maxSharedMemPerBlock", &DeviceProp::maxSharedMemPerBlock); 

    py::class_<GPUTimer>(m, "GPUTimer")
        .def(py::init<>())
        .def("start", &GPUTimer::start)
        .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed)
        .def("clear_L2_cache", &GPUTimer::clear_L2_cache);
}
