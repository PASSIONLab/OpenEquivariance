#pragma once

#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

class __attribute__ ((visibility ("default"))) ConvolutionImpl {
public:
    bool record_internal_stats = false;

    ConvolutionImpl() {
    }

    virtual void exec_conv(
        void* L1_in,
        void* L2_in,
        void* weights, 
        void* L3_out,
        void* rows,
        void* cols,
        uint64_t nnz,
        uint64_t node_count,
        void* workspace) = 0;

    void exec_conv_rawptrs(
        uint64_t L1_in,
        uint64_t L2_in,
        uint64_t weights,
        uint64_t L3_out,
        uint64_t rows,
        uint64_t cols,
        uint64_t nnz,
        uint64_t node_count,
        uint64_t workspace) {

        exec_conv(
            reinterpret_cast<void*>(L1_in),
            reinterpret_cast<void*>(L2_in),
            reinterpret_cast<void*>(weights),
            reinterpret_cast<void*>(L3_out),
            reinterpret_cast<void*>(rows),
            reinterpret_cast<void*>(cols),
            nnz,
            node_count,
            reinterpret_cast<void*>(workspace));
    }

    virtual void backward(
        void* L1_in, void* L1_grad,
        void* L2_in, void* L2_grad,
        void* weight, void* weight_grad,
        void* L3_grad,
        void* rows, void* cols,
        uint64_t nnz, uint64_t node_count,
        void* workspace, void* inverse_perm) = 0;

    void backward_rawptrs(
        uint64_t L1_in, uint64_t L1_grad,
        uint64_t L2_in, uint64_t L2_grad,
        uint64_t weight, uint64_t weight_grad,
        uint64_t L3_grad,
        uint64_t rows, uint64_t cols,
        uint64_t nnz, uint64_t node_count,
        uint64_t workspace, uint64_t inverse_perm) {

        backward(
            reinterpret_cast<void*>(L1_in),
            reinterpret_cast<void*>(L1_grad),
            reinterpret_cast<void*>(L2_in),
            reinterpret_cast<void*>(L2_grad),
            reinterpret_cast<void*>(weight),
            reinterpret_cast<void*>(weight_grad),
            reinterpret_cast<void*>(L3_grad),
            reinterpret_cast<void*>(rows),
            reinterpret_cast<void*>(cols),
            nnz,
            node_count,
            reinterpret_cast<void*>(workspace),
            reinterpret_cast<void*>(inverse_perm));
    }

    virtual ~ConvolutionImpl() {};
};

struct ConvData {
    void* rows;
    void* cols;
    unsigned long nnz;
    unsigned long node_count;
};

template<typename JIT_IMPL>
class __attribute__ ((visibility ("default"))) JITConvImpl : public ConvolutionImpl{
public:
    JIT_IMPL jit;
    KernelLaunchConfig forward_config; 
    KernelLaunchConfig backward_config; 

    JITConvImpl(
        std::string jit_kernel,
        KernelLaunchConfig forward_config_i,
        KernelLaunchConfig backward_config_i) :
            jit(jit_kernel),
            forward_config(forward_config_i),  
            backward_config(backward_config_i) {

        vector<string> kernels = {"forward", "backward", "fixup_forward", "fixup_backward"};
        jit.compile(kernels, {{}, {}, {}, {}}); 

        if(forward_config.smem > 0) {
            jit.set_max_smem(0, forward_config.smem);
        }

        if(backward_config.smem > 0) {
            jit.set_max_smem(1, backward_config.smem);
        }
    }

    void exec_conv(
            void* L1_in,
            void* L2_in,
            void* weights, 
            void* L3_out,
            void* rows,
            void* cols,
            uint64_t nnz,
            uint64_t node_count,
            void* workspace) {

        ConvData conv_data = {rows, cols, nnz, node_count};

        void *args[] = {&L1_in, &L2_in, &weights, &L3_out, &conv_data, &workspace}; 
        jit.execute(0, args, forward_config);

        if(reinterpret_cast<uint64_t>(workspace) != 0) {
            void *fixup_args[] = {&workspace, &L3_out};
            jit.execute(2, fixup_args, forward_config);
        }
    } 

    void backward(
            void* L1_in, void* L1_grad,
            void* L2_in, void* L2_grad,
            void* weight, void* weight_grad,
            void* L3_grad,
            void* rows, void* cols,
            uint64_t nnz, uint64_t node_count,
            void* workspace,
            void* transpose_perm) {

        ConvData conv_data = {rows, cols, nnz, node_count};
        void *args[] = {&L1_in, &L1_grad, &L2_in, &L2_grad, &weight, &weight_grad, &L3_grad, &conv_data, &workspace, &transpose_perm};
        jit.execute(1, args, backward_config);

        if(reinterpret_cast<uint64_t>(workspace) != 0) {
            void *fixup_args[] = {&workspace, &L1_grad};
            jit.execute(3, fixup_args, backward_config);
        }
    }

    ~JITConvImpl() = default; 
};