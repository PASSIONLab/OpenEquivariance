#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef CUDA_BACKEND
    #include "cublas_v2.h"
    #include <cuda_runtime.h>

    struct BlasHandle {
        cublasHandle_t handle;
        BlasHandle() {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
                throw std::logic_error("CUBLAS initialization failed");
        }
        ~BlasHandle() { cublasDestroy(handle); }
    };
#elif defined(HIP_BACKEND)
    #include "rocblas/rocblas.h"
    #include <hip/hip_runtime.h>

    struct BlasHandle {
        rocblas_handle handle;
        BlasHandle() {
            if (rocblas_create_handle(&handle) != rocblas_status_success)
                throw std::logic_error("rocBLAS initialization failed");
        }
        ~BlasHandle() { rocblas_destroy_handle(handle); }
    };
#elif defined(SYCL_BACKEND)
    #include <oneapi/mkl/blas.hpp>
    #include <sycl/sycl.hpp>

    // oneMKL takes the queue per call rather than a persistent handle, so this
    // exists only to keep a single shape across the three backends.
    struct BlasHandle {
        BlasHandle() = default;
        ~BlasHandle() = default;
    };

    // A small shared-USM array. oneMKL's pointer-array GEMM reads the pointer
    // lists from the device, so they cannot live on the host stack.
    template<typename T>
    class UsmArray {
        sycl::queue &q_;
        T *ptr_;
    public:
        UsmArray(sycl::queue &q, size_t n) : q_(q),
            ptr_(sycl::malloc_shared<T>(n, q)) {
            if (ptr_ == nullptr)
                throw std::runtime_error("Shared USM allocation failed!");
        }
        ~UsmArray() { sycl::free(ptr_, q_); }
        UsmArray(const UsmArray &) = delete;
        UsmArray &operator=(const UsmArray &) = delete;
        T &operator[](size_t i) { return ptr_[i]; }
        T *get() { return ptr_; }
    };
#endif

inline BlasHandle& get_blas_handle() {
    static BlasHandle handle;
    return handle;
}

template<typename T>
void group_gemm_blas(void* A_raw, void* B_raw, void* C_raw,
        int64_t* ragged_counts, int num_W, int batch_size, int m, int k, int ragged_inner) {

    auto& blas = get_blas_handle();
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
        rocblas_operation transa, transb;
#elif defined(SYCL_BACKEND)
        oneapi::mkl::transpose transa, transb;
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
            transa = rocblas_operation_transpose; transb = rocblas_operation_none;
#elif defined(SYCL_BACKEND)
            transa = oneapi::mkl::transpose::trans;
            transb = oneapi::mkl::transpose::nontrans;
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
            transa = rocblas_operation_none; transb = rocblas_operation_transpose;
#elif defined(SYCL_BACKEND)
            transa = oneapi::mkl::transpose::nontrans;
            transb = oneapi::mkl::transpose::trans;
#endif
        }
        ragged_offset += ragged_counts[i];

        if (ragged_counts[i] > 0) {
#ifdef CUDA_BACKEND
            cublasStatus_t stat;
            if (std::is_same<T, float>::value) {
                stat = cublasSgemmStridedBatched(blas.handle,
                    transa, transb, M, N, K,
                    reinterpret_cast<float*>(&alpha),
                    reinterpret_cast<float*>(A), lda, strideA,
                    reinterpret_cast<float*>(B), ldb, strideB,
                    reinterpret_cast<float*>(&beta),
                    reinterpret_cast<float*>(C), ldc, strideC,
                    batch_size);
            } else if (std::is_same<T, double>::value) {
                stat = cublasDgemmStridedBatched(blas.handle,
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
            rocblas_status stat;
            if (std::is_same<T, float>::value) {
                stat = rocblas_sgemm_strided_batched(blas.handle,
                    transa, transb, M, N, K,
                    reinterpret_cast<float*>(&alpha),
                    reinterpret_cast<float*>(A), lda, strideA,
                    reinterpret_cast<float*>(B), ldb, strideB,
                    reinterpret_cast<float*>(&beta),
                    reinterpret_cast<float*>(C), ldc, strideC,
                    batch_size);
            } else if (std::is_same<T, double>::value) {
                stat = rocblas_dgemm_strided_batched(blas.handle,
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
            if (stat != rocblas_status_success)
                throw std::logic_error("Grouped GEMM failed!");
#elif defined(SYCL_BACKEND)
            (void) blas;
            // Submit onto the same queue the kernels use so the GEMM is
            // ordered against them.
            sycl::queue &q = resolve_queue(get_current_stream());

            // oneMKL's strided batch API requires stride_c >= ldc * n, which
            // this interleaved layout deliberately violates (consecutive
            // matrices overlap in the batch dimension). The pointer-array form
            // imposes no such constraint, so build explicit pointer lists.
            // oneMKL reads these arrays on the device, so they live in shared
            // USM rather than on the host stack.
            UsmArray<const T *> a_ptrs(q, batch_size), b_ptrs(q, batch_size);
            UsmArray<T *> c_ptrs(q, batch_size);
            for (int j = 0; j < batch_size; j++) {
                a_ptrs[j] = A + static_cast<int64_t>(strideA) * j;
                b_ptrs[j] = B + static_cast<int64_t>(strideB) * j;
                c_ptrs[j] = C + static_cast<int64_t>(strideC) * j;
            }

            int64_t m64 = M, n64 = N, k64 = K;
            int64_t lda64 = lda, ldb64 = ldb, ldc64 = ldc;
            int64_t group_size = batch_size;

            try {
                oneapi::mkl::blas::column_major::gemm_batch(
                    q, &transa, &transb, &m64, &n64, &k64, &alpha,
                    a_ptrs.get(), &lda64,
                    b_ptrs.get(), &ldb64,
                    &beta, c_ptrs.get(), &ldc64,
                    1, &group_size)
                    // The pointer arrays are freed when this scope exits, so
                    // the submission has to complete before then.
                    .wait();
            } catch (const sycl::exception &e) {
                throw std::logic_error("Grouped GEMM failed: " + std::string(e.what()));
            }
#endif
        }
    }
}
