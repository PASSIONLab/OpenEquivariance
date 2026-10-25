#pragma once

#include <array>
#include <cstdint>

namespace oeq {

inline Tensor group_mm_view(
        const Tensor& tensor, std::array<int64_t, 3> sizes,
        std::array<int64_t, 3> strides, int64_t offset) {
    return oeq::as_strided(tensor, sizes, strides, tensor.storage_offset() + offset);
}

inline void grouped_gemm_via_torch_bmm(
        const Tensor& A, const Tensor& B, Tensor& C,
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
            oeq::bmm_out(output, input, weight);
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
            oeq::bmm_out(output, left, right);
        }
        offset += n;
    }
}

}
