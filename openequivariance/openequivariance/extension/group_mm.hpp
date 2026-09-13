#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>

namespace oeq {

inline void check_group_mm_shim(AOTITorchError status) {
    if (status != AOTI_TORCH_SUCCESS)
        throw std::runtime_error("group_gemm: PyTorch C shim failed");
}

using GroupMMTensor = std::unique_ptr<
    std::remove_pointer_t<AtenTensorHandle>,
    decltype(&aoti_torch_delete_tensor_object)>;

inline GroupMMTensor group_mm_view(
        AtenTensorHandle tensor, std::array<int64_t, 3> sizes,
        std::array<int64_t, 3> strides, int64_t offset) {
    AtenTensorHandle view = nullptr;
    check_group_mm_shim(aoti_torch__reinterpret_tensor(
        tensor, 3, sizes.data(), strides.data(), offset, &view));
    return GroupMMTensor(view, aoti_torch_delete_tensor_object);
}

inline void group_gemm_torch(
        AtenTensorHandle A, AtenTensorHandle B, AtenTensorHandle C,
        const int64_t* ragged_counts, int64_t num_W, int64_t batch_size,
        int64_t m, int64_t k, int64_t ragged_inner) {
    if (batch_size == 0 || m == 0 || k == 0)
        return;

    int64_t offset = 0;
    for (int64_t i = 0; i < num_W; ++i) {
        const int64_t n = ragged_counts[i];
        if (n == 0)
            continue;

        if (ragged_inner == 0) {
            auto input = group_mm_view(B,
                {batch_size, n, k}, {k, batch_size * k, 1},
                offset * batch_size * k);
            auto weight = group_mm_view(A,
                {batch_size, k, m}, {m * k, 1, k},
                i * batch_size * m * k);
            auto output = group_mm_view(C,
                {batch_size, n, m}, {m, batch_size * m, 1},
                offset * batch_size * m);
            check_group_mm_shim(aoti_torch_cuda_bmm_out(
                output.get(), input.get(), weight.get()));
        } else {
            auto left = group_mm_view(A,
                {batch_size, m, n}, {m, 1, batch_size * m},
                offset * batch_size * m);
            auto right = group_mm_view(B,
                {batch_size, n, k}, {k, batch_size * k, 1},
                offset * batch_size * k);
            auto output = group_mm_view(C,
                {batch_size, m, k}, {m * k, k, 1},
                i * batch_size * m * k);
            check_group_mm_shim(aoti_torch_cuda_bmm_out(
                output.get(), left.get(), right.get()));
        }
        offset += n;
    }
}

}
