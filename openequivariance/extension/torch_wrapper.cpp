#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <torch/all.h>
#include <torch/library.h>

#ifdef CUDA_BACKEND
#include "backend_cuda.hpp"
#include "group_mm_cuda.hpp"
using JITKernel = CUJITKernel;
using Allocator = CUDA_Allocator;

template<typename T>
using GroupMM = GroupMMCUDA<T>; 
#endif

#ifdef HIP_BACKEND
#include "backend_hip.hpp"
using JITKernel = HIPJITKernel;
using Allocator = HIP_Allocator;
template<typename T>
using GroupMM = GroupMMHIP<T>; 
#endif

#include "buffer.hpp"
#include "tensorproducts.hpp"
#include "convolution.hpp"

using namespace std;

using Map_t=torch::Dict<string, int64_t>; // Represent kernel dimensions / launch configurations as flat dictionaries 

inline void* data_ptr(const torch::Tensor &tensor) {
    return reinterpret_cast<void*>(tensor.data_ptr<float>()); // Unsure if this will also work for doubles 
}

class __attribute__ ((visibility ("default"))) TorchJITProduct: public torch::CustomClassHolder {
public:
    Map_t fwd_dict, bwd_dict, kernel_dims;
    JITTPImpl<JITKernel> internal;
    int64_t L3_dim;

    TorchJITProduct(string kernel_plaintext, 
                    Map_t fwd_dict_i, 
                    Map_t bwd_dict_i, 
                    Map_t kernel_dims_i) :
                    fwd_dict(fwd_dict_i.copy()),
                    bwd_dict(bwd_dict_i.copy()),
                    kernel_dims(kernel_dims_i.copy()),
                    internal(kernel_plaintext, 
                            KernelLaunchConfig(
                                fwd_dict.at("num_blocks"),
                                fwd_dict.at("threads_per_block"),
                                fwd_dict.at("smem")
                            ),
                            KernelLaunchConfig(
                                bwd_dict.at("num_blocks"),
                                bwd_dict.at("threads_per_block"),
                                bwd_dict.at("smem")
                            )
                        ),
                    L3_dim(kernel_dims.at("L3_dim"))
    { }

    tuple<string, Map_t, Map_t, Map_t> __obj_flatten__() {
        return make_tuple<string, Map_t, Map_t, Map_t>(internal.jit.kernel_plaintext, fwd_dict, bwd_dict, kernel_dims); 
    }
};

torch::Tensor jit_tp_forward(
        const c10::intrusive_ptr<TorchJITProduct> &jit_instance,
        const torch::Tensor &L1_in,
        const torch::Tensor &L2_in,
        const torch::Tensor &W) {
    
    int64_t num_batch = L1_in.sizes()[0];
    torch::Tensor L3_out = torch::zeros({num_batch, jit_instance->L3_dim}, L1_in.options());
        
    at::Tensor L1_contig = L1_in.contiguous();
    at::Tensor L2_contig = L2_in.contiguous();
    at::Tensor W_contig = W.contiguous();
    
    jit_instance->internal.exec_tensor_product(
            num_batch,
            data_ptr(L1_contig), 
            data_ptr(L2_contig), 
            data_ptr(L3_out),
            data_ptr(W_contig));
    
    return L3_out;
}

TORCH_LIBRARY_FRAGMENT(torch_wrapper, m) { 
    m.class_<TorchJITProduct>("TorchJITProduct")
        .def(torch::init<string, Map_t, Map_t, Map_t> ())
        .def("__obj_flatten__", &TorchJITProduct::__obj_flatten__)
        .def("__len__", [](const c10::intrusive_ptr<TorchJITProduct>& obj) -> int64_t {
            return 0;
        })
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<TorchJITProduct>& self)
                -> tuple<string, Map_t, Map_t, Map_t> {
              return self->__obj_flatten__(); 
            },
            // __setstate__
            [](tuple<string, Map_t, Map_t, Map_t>& state)
                -> c10::intrusive_ptr<TorchJITProduct> {
              return c10::make_intrusive<TorchJITProduct>(get<0>(state), get<1>(state), get<2>(state), get<3>(state));
            });
    m.def("jit_tp_forward(__torch__.torch.classes.torch_wrapper.TorchJITProduct jit, Tensor L1_in, Tensor L2_in, Tensor L3_in) -> Tensor");
};

TORCH_LIBRARY_IMPL(torch_wrapper, CUDA, m) { 
    m.impl("jit_tp_forward", &jit_tp_forward);
};

PYBIND11_MODULE(torch_wrapper, m) {};