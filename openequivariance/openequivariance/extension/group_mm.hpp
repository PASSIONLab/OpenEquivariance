#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

#ifdef CUDA_BACKEND
    #include "cublas_v2.h"
    #include <cuda_runtime.h>

    using BlasHandleT = cublasHandle_t;
#elif defined(HIP_BACKEND)
    #include <hipblas/hipblas.h>
    #include <hip/hip_runtime.h>

    using BlasHandleT = hipblasHandle_t;
#endif

BlasHandleT get_op_blas_handle();

template<typename T>
void group_gemm_blas(void* A_raw, void* B_raw, void* C_raw,
        int64_t* ragged_counts, int num_W, int batch_size, int m, int k, int ragged_inner) {

    BlasHandleT handle = get_op_blas_handle();
    T alpha = 1.0, beta = 0.0;
    T* A_base = reinterpret_cast<T*>(A_raw);
    T* B_base = reinterpret_cast<T*>(B_raw);
    T* C_base = reinterpret_cast<T*>(C_raw);

    int64_t ragged_offset = 0;
    for (int i = 0; i < num_W; i++) {
        int M, K, N, lda, ldb, ldc, strideA, strideB, strideC;
        T *A, *B, *C;
#ifdef CUDA_BACKEND
        cublasOperation_t transa, transb;
#elif defined(HIP_BACKEND)
        hipblasOperation_t transa, transb;
#endif

        if (ragged_inner == 0) {
            M = m; K = k; N = static_cast<int>(ragged_counts[i]);
            A = A_base + (m * k * batch_size * i);
            lda = k; strideA = M * K;
            B = B_base + (k * batch_size * ragged_offset);
            ldb = K * batch_size; strideB = K;
            C = C_base + (m * batch_size * ragged_offset);
            ldc = M * batch_size; strideC = M;
#ifdef CUDA_BACKEND
            transa = CUBLAS_OP_T; transb = CUBLAS_OP_N;
#elif defined(HIP_BACKEND)
            transa = HIPBLAS_OP_T; transb = HIPBLAS_OP_N;
#endif
        } else {
            M = k; K = static_cast<int>(ragged_counts[i]); N = m;
            A = B_base + (k * batch_size * ragged_offset);
            lda = k * batch_size; strideA = M;
            B = A_base + (m * batch_size * ragged_offset);
            ldb = m * batch_size; strideB = N;
            C = C_base + (m * k * batch_size * i);
            ldc = k; strideC = M * N;
#ifdef CUDA_BACKEND
            transa = CUBLAS_OP_N; transb = CUBLAS_OP_T;
#elif defined(HIP_BACKEND)
            transa = HIPBLAS_OP_N; transb = HIPBLAS_OP_T;
#endif
        }
        ragged_offset += ragged_counts[i];

        if (ragged_counts[i] > 0) {
#ifdef CUDA_BACKEND
            cublasStatus_t stat;
            if (std::is_same<T, float>::value) {
                stat = cublasSgemmStridedBatched(handle,
                    transa, transb, M, N, K,
                    reinterpret_cast<float*>(&alpha),
                    reinterpret_cast<float*>(A), lda, strideA,
                    reinterpret_cast<float*>(B), ldb, strideB,
                    reinterpret_cast<float*>(&beta),
                    reinterpret_cast<float*>(C), ldc, strideC,
                    batch_size);
            } else if (std::is_same<T, double>::value) {
                stat = cublasDgemmStridedBatched(handle,
                    transa, transb, M, N, K,
                    reinterpret_cast<double*>(&alpha),
                    reinterpret_cast<double*>(A), lda, strideA,
                    reinterpret_cast<double*>(B), ldb, strideB,
                    reinterpret_cast<double*>(&beta),
                    reinterpret_cast<double*>(C), ldc, strideC,
                    batch_size);
            } else {
                throw std::logic_error("Unsupported datatype for grouped GEMM!");
            }
            if (stat != CUBLAS_STATUS_SUCCESS)
                throw std::logic_error("Grouped GEMM failed!");
#elif defined(HIP_BACKEND)
            hipblasStatus_t stat;
            if (std::is_same<T, float>::value) {
                stat = hipblasSgemmStridedBatched(handle,
                    transa, transb, M, N, K,
                    reinterpret_cast<float*>(&alpha),
                    reinterpret_cast<float*>(A), lda, strideA,
                    reinterpret_cast<float*>(B), ldb, strideB,
                    reinterpret_cast<float*>(&beta),
                    reinterpret_cast<float*>(C), ldc, strideC,
                    batch_size);
            } else if (std::is_same<T, double>::value) {
                stat = hipblasDgemmStridedBatched(handle,
                    transa, transb, M, N, K,
                    reinterpret_cast<double*>(&alpha),
                    reinterpret_cast<double*>(A), lda, strideA,
                    reinterpret_cast<double*>(B), ldb, strideB,
                    reinterpret_cast<double*>(&beta),
                    reinterpret_cast<double*>(C), ldc, strideC,
                    batch_size);
            } else {
                throw std::logic_error("Unsupported datatype for grouped GEMM!");
            }
            if (stat != HIPBLAS_STATUS_SUCCESS)
                throw std::logic_error("Grouped GEMM failed!");
#endif
        }
    }
}
