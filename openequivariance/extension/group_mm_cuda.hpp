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

    int num_weight_matrices;

    vector<cublasOperation_t> transa_array; 
    vector<cublasOperation_t> transb_array; 

    vector<int> m_array;
    vector<int> n_array;
    vector<int> k_array;

    vector<T> alpha_array;
    vector<T> Aarray;
    vector<int> lda_array; 
    vector<T> Barray;
    vector<int> ldb_array;
    vector<T> beta_array;
    vector<T> Carray;
    vector<int> ldc_array;

    vector<int> group_size;

public:
    GroupMMCUDA(int num_weight_matrices) { 
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error("CUBLAS initialization failed");
        }
    }

    void group_gemm(void* weights, void* vectors, 
            int64_t* v_offsets, int m, int n, int k) {


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
                num_weight_matrices,
                const int group_size[]);
        }
        else {
            throw std::logic_error("Double precision support in progress");
        }
    }

    ~GroupMMCUDA() { 
        cublasDestroy(handle);
    }
};

