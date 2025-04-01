#pragma once

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

using namespace std;

template<typename T>
class GroupMMCUDA {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int num_W;
    int batch_size;

    T alpha;
    T beta;

public:
    GroupMMCUDA(int num_W, int batch_size) : 
            num_W(num_W),
            batch_size(batch_size),
            alpha(1.0),
            beta(0.0) { 
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error("CUBLAS initialization failed");
        }
    }

    void group_gemm(void* A_raw, void* B_raw, void* C_raw, 
            int64_t* ragged_counts, int m, int k, int ragged_inner) {
        /*
        * Performs one of two grouped, batched GEMMs with a single ragged dimension:
        * 
        * a) If ragged_inner = 0, multiplies each M x K row-major weight matrix A
        *    against B, where B is stored in column-major order with each matrix of
        *    dimensions K x [offset_diff]. Output has dimensions M x [offset_diff],
        *    stored in column-major order. 
        * b) If ragged_inner = 1, multiplies each M x [offset_diff] A matrix 
        *    against each B K x [offset_diff] matrix transposed to produce a 
        *    M x K matrix output.
        */

        T* A_base = reinterpret_cast<T*>(A_raw);
        T* B_base = reinterpret_cast<T*>(B_raw);
        T* C_base = reinterpret_cast<T*>(C_raw);

        int64_t ragged_offset = 0;
        for(int i = 0; i < num_W; i++) {
            int M, K, N, lda, ldb, ldc;
            T *A, *B, *C;

            int strideA, strideB, strideC;
            cublasOperation_t transa, transb;

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
               
                transa = CUBLAS_OP_T;
                transb = CUBLAS_OP_N;
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

                transa = CUBLAS_OP_N;
                transb = CUBLAS_OP_T;
            }
            ragged_offset += ragged_counts[i];
            
            if(ragged_counts[i] > 0) {
                if(std::is_same<T, float>::value) {
                    stat = cublasSgemmStridedBatched(handle,
                        transa, transb, 
                        M, N, K,
                        &alpha,
                        A, lda, strideA,
                        B, ldb, strideB, 
                        &beta, 
                        C, ldc, strideC,
                        batch_size);

                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        throw std::logic_error("Grouped GEMM failed!");
                    }
                }
                else {
                    throw std::logic_error("Double precision support in progress.");
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

    ~GroupMMCUDA() {
        cublasDestroy(handle);
    }
};