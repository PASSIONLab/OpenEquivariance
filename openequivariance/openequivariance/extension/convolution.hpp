#pragma once

#include <stdexcept>
#include <iostream>
#include <cstdint>

struct ConvData {
    void* rows;
    void* cols;
    unsigned long nnz;
    unsigned long node_count;
};

template<typename JIT_IMPL>
class __attribute__ ((visibility ("default"))) JITConvImpl {
public:
    JIT_IMPL jit;

    KernelLaunchConfig forward_config_ref; 
    KernelLaunchConfig backward_config_ref;
    KernelLaunchConfig double_backward_config_ref;
    int opt_level; 
    bool deterministic;

    enum Kernel {
        FORWARD = 0,
        BACKWARD = 1,
        DOUBLE_BACKWARD_A = 2,
        DOUBLE_BACKWARD_B = 3,
        FIXUP_FORWARD = 4,
        FIXUP_BACKWARD = 5,
        FIXUP_DOUBLE_BACKWARD_B = 6
    };

    JITConvImpl(
        std::string jit_kernel,
        KernelLaunchConfig forward_config_i,
        KernelLaunchConfig backward_config_i,
        KernelLaunchConfig double_backward_config_i,
        int opt_level_i,
        bool deterministic_i) :
            jit(jit_kernel),
            forward_config_ref(forward_config_i),  
            backward_config_ref(backward_config_i),
            double_backward_config_ref(double_backward_config_i),
            opt_level(opt_level_i),
            deterministic(deterministic_i) {

        vector<string> kernels = {"forward", "backward", "double_backward_A", "double_backward_B"};
        if(deterministic) {
            kernels.insert(kernels.end(), {"fixup_forward", "fixup_backward", "fixup_double_backwardB"});
        }
        jit.compile(kernels, vector<vector<int>>(kernels.size()), opt_level); 

        if(forward_config_ref.smem > 0) {
            jit.set_max_smem(FORWARD, forward_config_ref.smem);
            jit.set_max_smem(DOUBLE_BACKWARD_A, forward_config_ref.smem);
        }

        if(backward_config_ref.smem > 0) {
            jit.set_max_smem(BACKWARD, backward_config_ref.smem);
        }

        if(double_backward_config_ref.smem > 0) {
            jit.set_max_smem(DOUBLE_BACKWARD_B, double_backward_config_ref.smem);
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
            static_cast<int>(kernel_dims["opt_level"]),
            kernel_dims["deterministic"] != 0) { }

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
        jit.execute(FORWARD, args, with_stream(forward_config_ref, stream));

        if(deterministic) {
            void *fixup_args[] = {&workspace, &L3_out};
            
            KernelLaunchConfig fixup_config(
                forward_config_ref.num_blocks,
                forward_config_ref.num_threads,
                0
            );
            fixup_config.hStream = stream; 

            jit.execute(FIXUP_FORWARD, fixup_args, fixup_config);
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
        jit.execute(BACKWARD, args, with_stream(backward_config_ref, stream));

        if(deterministic) {
            void *fixup_args[] = {&workspace, &L1_grad};

            KernelLaunchConfig fixup_config(
                backward_config_ref.num_blocks,
                backward_config_ref.num_threads,
                0
            );
            fixup_config.hStream = stream;

            jit.execute(FIXUP_BACKWARD, fixup_args, fixup_config);
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

        jit.execute(DOUBLE_BACKWARD_A, args, with_stream(forward_config_ref, stream));
        if(deterministic) {
            void *fixup_args[] = {&wspace, &L3_dgrad};    
            KernelLaunchConfig fixup_config(
                forward_config_ref.num_blocks,
                forward_config_ref.num_threads,
                0
            );
            fixup_config.hStream = stream; 
            jit.execute(FIXUP_FORWARD, fixup_args, fixup_config);
        }

        jit.execute(DOUBLE_BACKWARD_B, args, with_stream(double_backward_config_ref, stream));
        if(deterministic) {
            void *fixup_args[] = {&wspace, &L1_grad};
            KernelLaunchConfig fixup_config(
                    double_backward_config_ref.num_blocks,
                    double_backward_config_ref.num_threads,
                    0
            );
            fixup_config.hStream = stream; 
            jit.execute(FIXUP_DOUBLE_BACKWARD_B, fixup_args, fixup_config);
        }
    }

    ~JITConvImpl() = default; 
};