#include <pybind11/pybind11.h>

#ifdef CUDA_BACKEND
    #include <ATen/cuda/CUDAContext.h>
#endif

#ifdef HIP_BACKEND
    #include <c10/hip/HIPStream.h>
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

caffe2::TypeMeta tensor_dtype(const Tensor &tensor) {
    return tensor.dtype();
}

bool tensor_is_cuda(const Tensor &tensor) {
    return tensor.device().is_cuda();
}

int64_t tensor_dim(const Tensor &tensor) {
    return tensor.dim();
}

int64_t tensor_size(const Tensor &tensor, int64_t dim) {
    return tensor.size(dim);
}

int64_t tensor_numel(const Tensor &tensor) {
    return tensor.numel();
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

Stream get_current_stream() {
#ifdef CUDA_BACKEND
    return c10::cuda::getCurrentCUDAStream();
#endif
#ifdef HIP_BACKEND
    return c10::hip::getCurrentHIPStream();
#endif
}
