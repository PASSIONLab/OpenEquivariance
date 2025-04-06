#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <ATen/Operators.h>
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
namespace py = pybind11;

inline void* data_ptr(const torch::Tensor &tensor) {
    return reinterpret_cast<void*>(tensor.data_ptr<float>());
}

/*
* Kernel launch configuration provides the number of blocks, number of threads per block, 
* and shared memory in bytes (int, int, int). 
*/
class __attribute__ ((visibility ("default"))) TorchJITProduct: public torch::CustomClassHolder {
public:
    std::tuple<int64_t, int64_t, int64_t> fwd_conf_tup, bwd_conf_tup;
    int64_t L3_dim;
    KernelLaunchConfig fwd_config, bwd_config;
    JITTPImpl<JITKernel> internal;

    TorchJITProduct(string kernel_plaintext, 
                    tuple<int64_t, int64_t, int64_t> fwd_conf, 
                    tuple<int64_t, int64_t, int64_t> bwd_conf,
                    int64_t L3_dim) :
                    fwd_conf_tup(fwd_conf),
                    bwd_conf_tup(bwd_conf),
                    fwd_config(get<0>(fwd_conf), get<1>(fwd_conf), get<2>(fwd_conf)),
                    bwd_config(get<0>(bwd_conf), get<1>(bwd_conf), get<2>(bwd_conf)),
                    internal(kernel_plaintext, fwd_config, bwd_config)
    { }

    std::tuple<string, tuple<int64_t, int64_t, int64_t>, tuple<int64_t, int64_t, int64_t>, int64_t> __obj_flatten__() {
        return std::tuple(internal.jit.kernel_plaintext, fwd_conf_tup, bwd_conf_tup, L3_dim);
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
        .def(torch::init<   string, 
                            tuple<int64_t, int64_t, int64_t>, 
                            tuple<int64_t, int64_t, int64_t>,
                            int64_t> ())
        .def("__obj_flatten__", &TorchJITProduct::__obj_flatten__)
        .def("__len__", [](const c10::intrusive_ptr<TorchJITProduct>& obj) -> int64_t {
            return 0;
        })
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<TorchJITProduct>& self)
                -> std::tuple<string, tuple<int64_t, int64_t, int64_t>, tuple<int64_t, int64_t, int64_t>, int64_t>  {
              return self->__obj_flatten__(); 
            },
            // __setstate__
            [](std::tuple<string, tuple<int64_t, int64_t, int64_t>, tuple<int64_t, int64_t, int64_t>, int64_t> state)
                -> c10::intrusive_ptr<TorchJITProduct> {
              return c10::make_intrusive<TorchJITProduct>(get<0>(state), get<1>(state), get<2>(state), get<3>(state));
            });
    m.def("jit_tp_forward(__torch__.torch.classes.torch_wrapper.TorchJITProduct jit, Tensor L1_in, Tensor L2_in, Tensor L3_in) -> Tensor");
};

TORCH_LIBRARY_IMPL(torch_wrapper, CUDA, m) { 
    m.impl("jit_tp_forward", &jit_tp_forward);
};

PYBIND11_MODULE(torch_wrapper, m) {};