#pragma once

#include <stdexcept>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <iostream>

template<typename JIT_IMPL>
class __attribute__ ((visibility ("default"))) JITTPImpl {
public:
    JIT_IMPL jit;
    KernelLaunchConfig forward_config, backward_config, double_backward_config; 
    int opt_level;

    JITTPImpl(
        std::string jit_kernel,
        KernelLaunchConfig forward_config_i,
        KernelLaunchConfig backward_config_i,
        KernelLaunchConfig double_backward_config_i,
        int opt_level_i) :
            jit(jit_kernel),
            forward_config(forward_config_i),  
            backward_config(backward_config_i),
            double_backward_config(double_backward_config_i),
            opt_level(opt_level_i) {

        vector<string> kernels = {"forward", "backward", "double_backward_A", "double_backward_B"};
        jit.compile(kernels, {{}, {}, {}, {}}, opt_level); 

        if(forward_config.smem > 0) {
            jit.set_max_smem(0, forward_config.smem);
            jit.set_max_smem(2, forward_config.smem);
        }

        if(backward_config.smem > 0) {
            jit.set_max_smem(1, backward_config.smem);
        
        }
        if(double_backward_config.smem > 0) {
            jit.set_max_smem(3, double_backward_config.smem);
        }
    }

    JITTPImpl(
            std::string jit_kernel,
            std::unordered_map<string, int64_t> fwd_dict, 
            std::unordered_map<string, int64_t> bwd_dict,
            std::unordered_map<string, int64_t> dbl_bwd_dict,
            std::unordered_map<string, int64_t> kernel_dims 
    ) : JITTPImpl(
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
            static_cast<int>(kernel_dims["opt_level"]) 
        ) { } 

    void exec_tensor_product(
        uint64_t num_products,
        void* L1_in,
        void* L2_in,
        void* L3_out,
        void* weights,
        Stream stream) {

        void *args[] = { &num_products, &L1_in, &L2_in, &L3_out, &weights};
        forward_config.hStream = stream; 
        jit.execute(0, args, forward_config);
    }

    void backward(
            size_t num_products,
            void* L1_in, void* L1_grad,
            void* L2_in, void* L2_grad,
            void* weight, void* weight_grad,
            void* L3_grad, Stream stream) {
        void *args[] = { &num_products, &L1_in, &L1_grad, &L2_in, &L2_grad, &weight, &weight_grad, &L3_grad};
        backward_config.hStream = stream; 
        jit.execute(1, args, backward_config);
    }

    void double_backward(
        size_t num_products,
        void* L1_in, void* L2_in, void* W, void* L3_grad, // Inputs of backward op 
        void* L1_dgrad, void* L2_dgrad, void* w_dgrad, // Gradients w.r.t outputs of backward op
        void* L1_grad, void* L2_grad, void* W_grad, void* L3_dgrad, Stream stream) {

        void* args[] = { 
            &num_products, &L1_in, &L2_in, &W, &L3_grad, &L1_dgrad, &L2_dgrad, &w_dgrad, 
            &L1_grad, &L2_grad, &W_grad, &L3_dgrad
        };
        double_backward_config.hStream = stream; 
        jit.execute(2, args, forward_config);
        jit.execute(3, args, double_backward_config);
    }

    ~JITTPImpl() = default; 

    // Integer pointer versions of the functions above
    void exec_tensor_product_device_rawptrs(uint64_t num_products,
            uint64_t L1_in, uint64_t L2_in, uint64_t L3_out, uint64_t weights) {
        exec_tensor_product(num_products,
            reinterpret_cast<void*>(L1_in),
            reinterpret_cast<void*>(L2_in),
            reinterpret_cast<void*>(L3_out),
            reinterpret_cast<void*>(weights),
            0 // Default Stream
        );
    } 

    void backward_device_rawptrs(uint64_t num_products,
            uint64_t L1_in, uint64_t L1_grad,
            uint64_t L2_in, uint64_t L2_grad, 
            uint64_t weight, uint64_t weight_grad,
            uint64_t L3_grad) {

        backward(num_products,
            reinterpret_cast<void*>(L1_in), reinterpret_cast<void*>(L1_grad),
            reinterpret_cast<void*>(L2_in), reinterpret_cast<void*>(L2_grad),
            reinterpret_cast<void*>(weight), reinterpret_cast<void*>(weight_grad),
            reinterpret_cast<void*>(L3_grad), 0 // Null = Default Stream
        );
    }
};