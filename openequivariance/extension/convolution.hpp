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
        void* workspace,
        Stream stream) = 0;

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
            reinterpret_cast<void*>(workspace),
            0 // Null aka Default Stream
            );
    }

    virtual void backward(
        void* L1_in, void* L1_grad,
        void* L2_in, void* L2_grad,
        void* weight, void* weight_grad,
        void* L3_grad,
        void* rows, void* cols,
        uint64_t nnz, uint64_t node_count,
        void* workspace, void* inverse_perm,
        Stream stream) = 0;

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
            reinterpret_cast<void*>(inverse_perm),
            0 // Null aka Default Stream
            );
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
    KernelLaunchConfig double_backward_config;
    bool is_uvw; 

    JITConvImpl(
        std::string jit_kernel,
        KernelLaunchConfig forward_config_i,
        KernelLaunchConfig backward_config_i,
        KernelLaunchConfig double_backward_config_i,
        bool is_uvw_i) :
            jit(jit_kernel),
            forward_config(forward_config_i),  
            backward_config(backward_config_i),
            double_backward_config(double_backward_config_i),
            is_uvw(is_uvw_i) {

        vector<string> kernels = {"forward", "backward", "fixup_forward", "fixup_backward", "double_backward_A", "double_backward_B", "fixup_double_backwardB"};

        int opt_level = 3;
        #ifdef HIP_BACKEND
        if(is_uvw) {
            opt_level = 1;
        }
        #endif 
        jit.compile(kernels, {{}, {}, {}, {}, {}, {}, {}}, opt_level); 

        if(forward_config.smem > 0) {
            jit.set_max_smem(0, forward_config.smem);
            jit.set_max_smem(4, forward_config.smem);
        }

        if(backward_config.smem > 0) {
            jit.set_max_smem(1, backward_config.smem);
        }

        if(double_backward_config.smem > 0) {
            jit.set_max_smem(5, double_backward_config.smem);
        }
    }

    JITConvImpl(
            std::string jit_kernel,
            std::unordered_map<string, int64_t> fwd_dict, 
            std::unordered_map<string, int64_t> bwd_dict,
            std::unordered_map<string, int64_t> dbl_bwd_dict,
            std::unordered_map<string, int64_t> kernel_dims 
    ) : JITConvImpl(
            jit_kernel,
            KernelLaunchConfig(
                fwd_dict["num_blocks"],
                fwd_dict["num_threads"],
                fwd_dict["smem"]
            ),
            KernelLaunchConfig(
                bwd_dict["num_blocks"],
                bwd_dict["num_threads"],
                bwd_dict["smem"]
            ),
            KernelLaunchConfig(
                dbl_bwd_dict["num_blocks"],
                dbl_bwd_dict["num_threads"],
                dbl_bwd_dict["smem"]
            ),
            kernel_dims["is_uvw"] == 1) { }

    void exec_conv(
            void* L1_in,
            void* L2_in,
            void* weights, 
            void* L3_out,
            void* rows,
            void* cols,
            uint64_t nnz,
            uint64_t node_count,
            void* workspace, 
            Stream stream) {

        ConvData conv_data = {rows, cols, nnz, node_count};

        void *args[] = {&L1_in, &L2_in, &weights, &L3_out, &conv_data, &workspace};
        forward_config.hStream = stream;  
        jit.execute(0, args, forward_config);

        if(reinterpret_cast<uint64_t>(workspace) != 0) {
            void *fixup_args[] = {&workspace, &L3_out};
            
            KernelLaunchConfig fixup_config;
            fixup_config.num_blocks = forward_config.num_blocks;
            fixup_config.num_threads = forward_config.num_threads;
            fixup_config.smem = 0;
            fixup_config.hStream = stream; 

            jit.execute(2, fixup_args, fixup_config);
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
            void* transpose_perm, 
            Stream stream) {

        ConvData conv_data = {rows, cols, nnz, node_count};
        void *args[] = {&L1_in, &L1_grad, &L2_in, &L2_grad, &weight, &weight_grad, &L3_grad, &conv_data, &workspace, &transpose_perm};
        backward_config.hStream = stream; 
        jit.execute(1, args, backward_config);

        if(reinterpret_cast<uint64_t>(workspace) != 0) {
            void *fixup_args[] = {&workspace, &L1_grad};

            KernelLaunchConfig fixup_config;
            fixup_config.num_blocks = backward_config.num_blocks;
            fixup_config.num_threads = backward_config.num_threads;
            fixup_config.smem = 0; fixup_config.hStream = stream; 

            jit.execute(3, fixup_args, fixup_config);
        }
    }

    void double_backward(
            void* L1_in, void* L2_in, void* W, void* L3_grad, 
            void* L1_dgrad, void* L2_dgrad, void* w_dgrad, 
            void* L1_grad, void* L2_grad, void* W_grad, void* L3_dgrad, 
            void* rows, void* cols,
            uint64_t nnz, uint64_t node_count,
            void* wspace, void* transpose_perm, 
            Stream stream) {

        ConvData conv_data = {rows, cols, nnz, node_count};
        void* args[] = { 
            &L1_in, &L2_in, &W, &L3_grad, &L1_dgrad, &L2_dgrad, &w_dgrad, 
            &L1_grad, &L2_grad, &W_grad, &L3_dgrad, &conv_data, &wspace, &transpose_perm
        };
        double_backward_config.hStream = stream; 
        jit.execute(4, args, forward_config);
        if(reinterpret_cast<uint64_t>(wspace) != 0) {
            void *fixup_args[] = {&wspace, &L3_dgrad};    
            KernelLaunchConfig fixup_config;
            fixup_config.num_blocks = forward_config.num_blocks;
            fixup_config.num_threads = forward_config.num_threads;
            fixup_config.smem = 0; fixup_config.hStream = stream; 
            jit.execute(2, fixup_args, fixup_config);
        }

        jit.execute(5, args, double_backward_config);
        if(reinterpret_cast<uint64_t>(wspace) != 0) {
            void *fixup_args[] = {&wspace, &L1_grad};
            KernelLaunchConfig fixup_config;
            fixup_config.num_blocks = double_backward_config.num_blocks;
            fixup_config.num_threads = double_backward_config.num_threads;
            fixup_config.smem = 0; fixup_config.hStream = stream; 
            jit.execute(6, fixup_args, fixup_config);
        }
    }

    ~JITConvImpl() = default; 
};