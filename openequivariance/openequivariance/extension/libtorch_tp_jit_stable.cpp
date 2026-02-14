#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/ArrayRef.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef HIP_BACKEND
    #include <c10/hip/HIPStream.h>
#endif

using Tensor = torch::stable::Tensor;
using Dtype = torch::headeronly::ScalarType;

constexpr Dtype kFloat = torch::headeronly::kFloat;
constexpr Dtype kDouble = torch::headeronly::kDouble;
constexpr Dtype kInt = torch::headeronly::kInt;
constexpr Dtype kLong = torch::headeronly::kLong;
constexpr Dtype kByte = torch::headeronly::kUInt8;

#define CHECK STD_TORCH_CHECK
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

Dtype tensor_dtype(const Tensor &tensor) {
    return tensor.scalar_type();
}

bool tensor_is_cuda(const Tensor &tensor) {
    return tensor.is_cuda();
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
    (void)name;
}

const uint8_t *tensor_data_ptr_u8(const Tensor &tensor) {
    return static_cast<const uint8_t *>(tensor.data_ptr());
}

void *data_ptr(const Tensor &tensor) {
    Dtype dtype = tensor.scalar_type();
    if (dtype == kFloat || dtype == kDouble || dtype == kLong || dtype == kByte || dtype == kInt)
        return tensor.data_ptr();
    throw std::logic_error("Unsupported tensor datatype!");
}

Stream get_current_stream() {
#ifdef CUDA_BACKEND
    void *stream_ptr = nullptr;
    auto device_index = torch::stable::accelerator::getCurrentDeviceIndex();
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
    return static_cast<Stream>(stream_ptr);
#endif
#ifdef HIP_BACKEND
    return c10::hip::getCurrentHIPStream();
#endif
}

