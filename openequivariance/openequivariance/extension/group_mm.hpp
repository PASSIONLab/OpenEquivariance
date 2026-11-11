#pragma once

#include <cstdint>

namespace oeq {

inline void grouped_gemm_via_torch_bmm(
        const Tensor& A, const Tensor& B, Tensor& C,
        const int64_t* ragged_counts, int64_t num_W, int64_t ragged_inner) {
    Tensor A_t = tensor_transpose(A, 0, 1);
    Tensor B_t = tensor_transpose(B, 0, 1);
    Tensor C_t = tensor_transpose(C, 0, 1);

    int64_t offset = 0;
    for (int64_t i = 0; i < num_W; ++i) {
        const int64_t n = ragged_counts[i];
        if (n == 0)
            continue;

        Tensor B_i = tensor_narrow(B_t, 1, offset, n);
        if (ragged_inner == 0) {
            Tensor C_i = tensor_narrow(C_t, 1, offset, n);
            Tensor A_i = tensor_select(A_t, 1, i);
            oeq::bmm_out(C_i, B_i, tensor_transpose(A_i, 1, 2));
        } else {
            Tensor C_i = tensor_select(C_t, 1, i);
            Tensor A_i = tensor_narrow(A_t, 1, offset, n);
            oeq::bmm_out(C_i, tensor_transpose(A_i, 1, 2), B_i);
        }
        offset += n;
    }
}

} // namespace oeq
