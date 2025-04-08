#pragma once

#include <stdexcept>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <iostream>

class __attribute__ ((visibility ("default"))) GenericTensorProductImpl {
public:
    GenericTensorProductImpl() { }

    virtual void exec_tensor_product(uint64_t num_products,
            void* L1_in, void* L2_in, void* L3_out, void* weights) = 0;

    void exec_tensor_product_device_rawptrs(uint64_t num_products,
            uint64_t L1_in, uint64_t L2_in, uint64_t L3_out, uint64_t weights) {
        
        exec_tensor_product(num_products,
            reinterpret_cast<void*>(L1_in),
            reinterpret_cast<void*>(L2_in),
            reinterpret_cast<void*>(L3_out),
            reinterpret_cast<void*>(weights));
    } 

    virtual void backward(size_t num_products,
            void* L1_in, void* L1_grad,
            void* L2_in, void* L2_grad,
            void* weight, void* weight_grad,
            void* L3_grad) {

        throw std::logic_error("Backward pass not implemented yet!");
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
            reinterpret_cast<void*>(L3_grad)
        );
    }

    virtual ~GenericTensorProductImpl() {};
};

template<typename JIT_IMPL>
class __attribute__ ((visibility ("default"))) JITTPImpl : public GenericTensorProductImpl {
public:
    JIT_IMPL jit;
    KernelLaunchConfig forward_config; 
    KernelLaunchConfig backward_config; 

    JITTPImpl(
        std::string jit_kernel,
        KernelLaunchConfig forward_config_i,
        KernelLaunchConfig backward_config_i) :
            jit(jit_kernel),
            forward_config(forward_config_i),  
            backward_config(backward_config_i) {
        vector<string> kernels = {"forward", "backward"};
        jit.compile(kernels, {{}, {}}); 

        if(forward_config.smem > 0) {
            jit.set_max_smem(0, forward_config.smem);
        }

        if(backward_config.smem > 0) {
            jit.set_max_smem(1, backward_config.smem);
        }
    }

    JITTPImpl(
            std::string jit_kernel,
            std::unordered_map<string, int64_t> fwd_dict, 
            std::unordered_map<string, int64_t> bwd_dict
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
            )) { } 

    void exec_tensor_product(
        uint64_t num_products,
        void* L1_in,
        void* L2_in,
        void* L3_out,
        void* weights) {

        void *args[] = { &num_products, &L1_in, &L2_in, &L3_out, &weights};
        jit.execute(0, args, forward_config);
    }

    void backward(
            size_t num_products,
            void* L1_in, void* L1_grad,
            void* L2_in, void* L2_grad,
            void* weight, void* weight_grad,
            void* L3_grad) {
        void *args[] = { &num_products, &L1_in, &L1_grad, &L2_in, &L2_grad, &weight, &weight_grad, &L3_grad};
        jit.execute(1, args, backward_config);
    }

    ~JITTPImpl() = default; 
};