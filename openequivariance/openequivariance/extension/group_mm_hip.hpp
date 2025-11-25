#pragma once

#include "rocblas/rocblas.h" 
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <iostream>


template<typename T>
class GroupMMHIP {
    rocblas_status stat;
    rocblas_handle handle;

    int num_W;
    int batch_size;

    T alpha;
    T beta;

public:
    GroupMMHIP(int num_W, int batch_size) : 
            num_W(num_W),
            batch_size(batch_size),
            alpha(1.0),
            beta(0.0) {
        if(rocblas_create_handle(&handle) != rocblas_status_success) {
            throw std::logic_error("rocBLAS initialization failed");
        }
    }

    void group_gemm(void* A_raw, void* B_raw, void* C_raw, 
            int64_t* ragged_counts, int m, int k, int ragged_inner) {
    
        T* A_base = reinterpret_cast<T*>(A_raw);
        T* B_base = reinterpret_cast<T*>(B_raw);
        T* C_base = reinterpret_cast<T*>(C_raw);

        int64_t ragged_offset = 0;
        for(int i = 0; i < num_W; i++) {
            int M, K, N, lda, ldb, ldc;
            T *A, *B, *C;

            int strideA, strideB, strideC;
            rocblas_operation transa, transb;

            if(ragged_inner == 0) {
                M = m;
                K = k;
                N = static_cast<int>(ragged_counts[i]);

                A = A_base + (m * k * batch_size * i);
                lda = k; strideA = M * K; 
                
                B = B_base + (k * batch_size * ragged_offset); 
                ldb = K * batch_size; strideB = K; 

                C = C_base + (m * batch_size * ragged_offset); 
                ldc = M * batch_size; strideC = M; 
               
                transa = rocblas_operation_transpose;
                transb = rocblas_operation_none;
            }
            else {
                M = k;
                K = static_cast<int>(ragged_counts[i]);
                N = m;

                A = B_base + (k * batch_size * ragged_offset);
                lda = k * batch_size; strideA = M; 

                B = A_base + (m * batch_size * ragged_offset);
                ldb = m * batch_size; strideB = N;
                
                C = C_base + (m * k * batch_size * i);
                ldc = k; strideC = M * N;

                transa = rocblas_operation_none;
                transb = rocblas_operation_transpose;
            }
            ragged_offset += ragged_counts[i];
            
            if(ragged_counts[i] > 0) {
                if(std::is_same<T, float>::value) {
                    stat = rocblas_sgemm_strided_batched(handle,
                        transa, transb, 
                        M, N, K,
                        reinterpret_cast<float*>(&alpha),
                        reinterpret_cast<float*>(A), lda, strideA,
                        reinterpret_cast<float*>(B), ldb, strideB, 
                        reinterpret_cast<float*>(&beta), 
                        reinterpret_cast<float*>(C), ldc, strideC,
                        batch_size);
                }
                else if(std::is_same<T, double>::value) {
                    stat = rocblas_dgemm_strided_batched(handle,
                        transa, transb, 
                        M, N, K,
                        reinterpret_cast<double*>(&alpha),
                        reinterpret_cast<double*>(A), lda, strideA,
                        reinterpret_cast<double*>(B), ldb, strideB, 
                        reinterpret_cast<double*>(&beta), 
                        reinterpret_cast<double*>(C), ldc, strideC,
                        batch_size);
                }
                else {
                    throw std::logic_error("Unsupported datatype for grouped GEMM!");
                }
                if (stat != rocblas_status_success) {
                    throw std::logic_error("Grouped GEMM failed!");
                }
            }
        }
    }

    void group_gemm_intptr(uint64_t weights, 
            uint64_t vectors, uint64_t output, 
            uint64_t ragged_counts, int m, int k, int ragged_inner) {    
        group_gemm(
            reinterpret_cast<void*>(weights), 
            reinterpret_cast<void*>(vectors), 
            reinterpret_cast<void*>(output), 
            reinterpret_cast<int64_t*>(ragged_counts), 
            m, k, ragged_inner);
    }

    ~GroupMMHIP() {
        rocblas_destroy_handle(handle);
    }
};