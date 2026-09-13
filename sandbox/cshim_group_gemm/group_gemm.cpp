// Prototype for the PyTorch CUDA backend. Uses only PyTorch's stable C ABI.
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace {

void check(AOTITorchError status) {
    if (status != AOTI_TORCH_SUCCESS)
        throw std::runtime_error("PyTorch C shim failed");
}

using Tensor = std::unique_ptr<
    std::remove_pointer_t<AtenTensorHandle>,
    decltype(&aoti_torch_delete_tensor_object)>;
using DeviceGuard = std::unique_ptr<
    std::remove_pointer_t<CUDAGuardHandle>,
    decltype(&aoti_torch_delete_cuda_guard)>;

// Only tensor metadata is created; the caller retains ownership of the buffer.
Tensor view(void* data, std::array<int64_t, 3> sizes,
            std::array<int64_t, 3> strides, int32_t dtype, int32_t device) {
    AtenTensorHandle tensor = nullptr;
    check(aoti_torch_create_tensor_from_blob_v2(
        data, 3, sizes.data(), strides.data(), 0, dtype,
        aoti_torch_device_type_cuda(), device, &tensor,
        aoti_torch_layout_strided(), nullptr, 0));
    return Tensor(tensor, aoti_torch_delete_tensor_object);
}

} // namespace

template <typename T>
void group_gemm_cshim(
    T* A, T* B, T* C, const int64_t* ragged_counts, int num_groups,
    int64_t batch, int64_t m, int64_t k, int ragged_inner, int32_t device) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const int32_t dtype = std::is_same_v<T, float>
        ? aoti_torch_dtype_float32() : aoti_torch_dtype_float64();

    CUDAGuardHandle raw_guard = nullptr;
    check(aoti_torch_create_cuda_guard(device, &raw_guard));
    DeviceGuard guard(raw_guard, aoti_torch_delete_cuda_guard);

    int64_t offset = 0;
    for (int i = 0; i < num_groups; ++i) {
        const int64_t n = ragged_counts[i];
        if (n == 0) continue; // Preserve the original empty-group behavior.

        if (ragged_inner == 0) {
            // [batch, n, k] @ [batch, k, m] -> [batch, n, m]
            auto input = view(B + offset * batch * k,
                              {batch, n, k}, {k, batch * k, 1}, dtype, device);
            auto weight = view(A + i * batch * m * k,
                               {batch, k, m}, {m * k, 1, k}, dtype, device);
            auto output = view(C + offset * batch * m,
                               {batch, n, m}, {m, batch * m, 1}, dtype, device);
            check(aoti_torch_cuda_bmm_out(output.get(), input.get(), weight.get()));
        } else {
            // [batch, m, n] @ [batch, n, k] -> [batch, m, k]
            auto left = view(A + offset * batch * m,
                             {batch, m, n}, {m, 1, batch * m}, dtype, device);
            auto right = view(B + offset * batch * k,
                              {batch, n, k}, {k, batch * k, 1}, dtype, device);
            auto output = view(C + i * batch * m * k,
                               {batch, m, k}, {m * k, k, 1}, dtype, device);
            check(aoti_torch_cuda_bmm_out(output.get(), left.get(), right.get()));
        }
        offset += n;
    }
}

// Small ctypes entry point for the smoke test, not production registration code.
extern "C" int oeq_group_gemm_cshim(
    int dtype, void* A, void* B, void* C, const int64_t* counts,
    int groups, int64_t batch, int64_t m, int64_t k, int inner, int32_t device) {
    try {
        if (dtype == 0) {
            group_gemm_cshim(static_cast<float*>(A), static_cast<float*>(B),
                             static_cast<float*>(C), counts, groups,
                             batch, m, k, inner, device);
        } else if (dtype == 1) {
            group_gemm_cshim(static_cast<double*>(A), static_cast<double*>(B),
                             static_cast<double*>(C), counts, groups,
                             batch, m, k, inner, device);
        } else {
            throw std::runtime_error("Expected float32 or float64");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
