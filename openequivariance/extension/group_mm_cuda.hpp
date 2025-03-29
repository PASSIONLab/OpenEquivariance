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

    vector<cublasOperation_t> transa_array; 
    vector<cublasOperation_t> transb_array; 

    vector<int> m_array;
    vector<int> n_array;
    vector<int> k_array;

    vector<T> alpha_array;
    vector<T*> Aarray;
    vector<int> lda_array; 
    vector<T*> Barray;
    vector<int> ldb_array;
    vector<T> beta_array;
    vector<T*> Carray;
    vector<int> ldc_array;

    vector<int> group_size;

public:
    GroupMMCUDA(int num_W) 
            num_weight_matrices(num_W, 0),
            transa_array(num_W, 0),
            transb_array(num_W, 0),

            m_array(num_W, 0),
            n_array(num_W, 0),
            k_array(num_W, 0),

            transa_array(num_W, CUBLAS_OP_N),
            transb_array(num_W, CUBLAS_OP_N),

            alpha_array(num_W, 1.0),

            Aarray(num_W, nullptr),
            lda_array(num_W, 0),

            Barray(num_W, nullptr),
            ldb_array(num_W, 0),

            beta_array(num_W, 0),
            Carray(num_W, nullptr),
            ldc_array(num_W, 0),

            group_size(num_W, 1) // There's a single matrix in each group 
    { 
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error("CUBLAS initialization failed");
        }
    }

    void group_gemm(void* weights, void* vectors, void* output, 
            int64_t* v_offsets, int m, int k) {

        T* w_start = static_cast<T>(weights);
        T* v_start = static_cast<T>(vectors);

        for(int i = 0; i < num_W; i++) {
            m_array[i] = m;
            k_array[i] = k;
            n_array[i] = static_cast<int>(v_offsets[i+1] - v_offsets[i]);

            Aarray[i] = weights + (m * k) * i;
            lda_array[i] = m;    // TODO: Check this!
            
            Barray[i] = vectors + (k * v_offsets[i]); 
            ldb_array[i] = n_array[i]; // TODO: Check this! 

            Carray[i] = output + (m * v_offsets[i]);
            ldc_array[i] = m * v_offsets[i]; // TODO: Check this! 
        }

        if(std::is_same<T, float>::value) {
            stat = cublasSgemmGroupedBatched(handle,
                transa_array.data(), 
                transb_array.data(), 
                m_array.data(),
                n_array.data(),
                k_array.data(),
                alpha_array.data(),
                Aarray.data(),
                lda_array.data(),
                Barray.data(),
                ldb_array.data(),
                beta_array.data(),
                Carray.data(),
                ldc_array.data(),
                num_W,
                group_size.data());

            if (stat != CUBLAS_STATUS_SUCCESS) {
                throw std::logic_error("Grouped GEMM failed!");
            }
        }
        else {
            throw std::logic_error("Double precision support in progress");
        }
    }

    void group_gemm_intptr(uint64_t weights, 
            uint64_t vectors, uint64_t output, 
            uint64_t v_offsets, int m, int k) {
        
        group_gemm(
            reinterpret_cast<void*>(weights), 
            reinterpret_cast<void*>(vectors), 
            reinterpret_cast<void*>(output), 
            reinterpret_cast<int64_t*>(v_offsets), 
            m, k);
    }

    ~GroupMMCUDA() { 
        cublasDestroy(handle);
    }
};

