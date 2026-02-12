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

    JITConvImpl(
        std::string jit_kernel,
        KernelLaunchConfig forward_config_i,
        KernelLaunchConfig backward_config_i,
        KernelLaunchConfig double_backward_config_i,
        int opt_level_i) :
            jit(jit_kernel),
            forward_config_ref(forward_config_i),  
            backward_config_ref(backward_config_i),
            double_backward_config_ref(double_backward_config_i),
            opt_level(opt_level_i) {

        vector<string> kernels = {"forward", "backward", "fixup_forward", "fixup_backward", "double_backward_A", "double_backward_B", "fixup_double_backwardB"};
        jit.compile(kernels, {{}, {}, {}, {}, {}, {}, {}}, opt_level); 

        if(forward_config_ref.smem > 0) {
            jit.set_max_smem(0, forward_config_ref.smem);
            jit.set_max_smem(4, forward_config_ref.smem);
        }

        if(backward_config_ref.smem > 0) {
            jit.set_max_smem(1, backward_config_ref.smem);
        }

        if(double_backward_config_ref.smem > 0) {
            jit.set_max_smem(5, double_backward_config_ref.smem);
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
            static_cast<int>(kernel_dims["opt_level"])) { }

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
        jit.execute(0, args, with_stream(forward_config_ref, stream));

        if(reinterpret_cast<uint64_t>(workspace) != 0) {
            void *fixup_args[] = {&workspace, &L3_out};
            
            KernelLaunchConfig fixup_config(
                forward_config_ref.num_blocks,
                forward_config_ref.num_threads,
                0
            );
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
        jit.execute(1, args, with_stream(backward_config_ref, stream));

        if(reinterpret_cast<uint64_t>(workspace) != 0) {
            void *fixup_args[] = {&workspace, &L1_grad};

            KernelLaunchConfig fixup_config(
                backward_config_ref.num_blocks,
                backward_config_ref.num_threads,
                0
            );
            fixup_config.hStream = stream;

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

        jit.execute(4, args, with_stream(forward_config_ref, stream));
        if(reinterpret_cast<uint64_t>(wspace) != 0) {
            void *fixup_args[] = {&wspace, &L3_dgrad};    
            KernelLaunchConfig fixup_config(
                forward_config_ref.num_blocks,
                forward_config_ref.num_threads,
                0
            );
            fixup_config.hStream = stream; 
            jit.execute(2, fixup_args, fixup_config);
        }

        jit.execute(5, args, with_stream(double_backward_config_ref, stream));
        if(reinterpret_cast<uint64_t>(wspace) != 0) {
            void *fixup_args[] = {&wspace, &L1_grad};
            KernelLaunchConfig fixup_config(
                    double_backward_config_ref.num_blocks,
                    double_backward_config_ref.num_threads,
                    0
            );
            fixup_config.hStream = stream; 
            jit.execute(6, fixup_args, fixup_config);
        }
    }

    ~JITConvImpl() = default; 
};