#include <pybind11/pybind11.h> #include <pybind11/numpy.h>
#include <iostream>
#include <unordered_map>
#include <stdexcept>

#ifdef CUDA_BACKEND
    #include "backend_cuda.hpp"
    #include "group_mm_cuda.hpp"
    using JITKernel = CUJITKernel;
    using GPU_Allocator = CUDA_Allocator;

    template<typename T>
    using GroupMM = GroupMMCUDA<T>; 
#endif

#ifdef HIP_BACKEND
    #include "backend_hip.hpp"
    #include "group_mm_hip.hpp"
    using JITKernel = HIPJITKernel;
    using GPU_Allocator = HIP_Allocator;

    template<typename T>
    using GroupMM = GroupMMHIP<T>; 
#endif

#include "buffer.hpp"
#include "tensorproducts.hpp"
#include "convolution.hpp"

using namespace std;
namespace py=pybind11;

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

using Map_t=torch::Dict<string, int64_t>;

inline void* data_ptr(const torch::Tensor &tensor) {
    if(tensor.dtype() == torch::kFloat)
        return reinterpret_cast<void*>(tensor.data_ptr<float>());
    else if(tensor.dtype() == torch::kDouble)
        return reinterpret_cast<void*>(tensor.data_ptr<double>());
    else if(tensor.dtype() == torch::kLong) 
        return reinterpret_cast<void*>(tensor.data_ptr<int64_t>());
    else
        throw logic_error("Unsupported tensor datatype!");
}

class __attribute__ ((visibility ("default"))) TorchJITProduct : public torch::CustomClassHolder {
public:
    Map_t fwd_dict, bwd_dict, kernel_dims;
    JITTPImpl<JITKernel> internal;
    int64_t L3_dim;
    TorchJITProduct(string kernel_plaintext, Map_t fwd_dict_i, Map_t bwd_dict_i, Map_t kernel_dims_i) :
        fwd_dict(fwd_dict_i.copy()),
        bwd_dict(bwd_dict_i.copy()),
        kernel_dims(kernel_dims_i.copy()),
        internal(kernel_plaintext, 
            KernelLaunchConfig(
                fwd_dict.at("num_blocks"),
                fwd_dict.at("num_threads"),
                fwd_dict.at("smem")
            ),
            KernelLaunchConfig(
                bwd_dict.at("num_blocks"),
                bwd_dict.at("num_threads"),
                bwd_dict.at("smem")
            )),
        L3_dim(kernel_dims.at("L3_dim")) { }

    tuple<  tuple<string, string>, 
            tuple<string, Map_t>, 
            tuple<string, Map_t>, 
            tuple<string, Map_t>> __obj_flatten__() {
        return tuple(tuple("kernel_plaintext", internal.jit.kernel_plaintext),
            tuple("fwd_config", fwd_dict),
            tuple("bwd_config", bwd_dict),
            tuple("kernel_dims", kernel_dims));
    }

    void exec_tensor_product_device_rawptrs(int64_t num_batch, int64_t L1_in, int64_t L2_in, int64_t L3_out, int64_t weights) {    
        internal.exec_tensor_product(
                num_batch,
                reinterpret_cast<void*>(L1_in), 
                reinterpret_cast<void*>(L2_in), 
                reinterpret_cast<void*>(L3_out), 
                reinterpret_cast<void*>(weights)); 
    } 

    void backward_device_rawptrs(int64_t num_batch,
            int64_t L1_in, int64_t L1_grad,
            int64_t L2_in, int64_t L2_grad, 
            int64_t weight, int64_t weight_grad,
            int64_t L3_grad) {
        internal.backward(num_batch,
            reinterpret_cast<void*>(L1_in), reinterpret_cast<void*>(L1_grad),
            reinterpret_cast<void*>(L2_in), reinterpret_cast<void*>(L2_grad),
            reinterpret_cast<void*>(weight), reinterpret_cast<void*>(weight_grad),
            reinterpret_cast<void*>(L3_grad)
        );
    }
};

torch::Tensor jit_tp_forward(
        const c10::intrusive_ptr<TorchJITProduct> &jit_instance,
        const torch::Tensor &L1_in,
        const torch::Tensor &L2_in,
        const torch::Tensor &W) {

    int64_t num_batch = L1_in.sizes()[0];
    torch::Tensor L3_out = torch::empty({num_batch, jit_instance->L3_dim}, L1_in.options());
        
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

tuple<torch::Tensor, torch::Tensor, torch::Tensor> jit_tp_backward(
        const c10::intrusive_ptr<TorchJITProduct> &jit_instance,
        const torch::Tensor &L1_in,
        const torch::Tensor &L2_in,
        const torch::Tensor &W, 
        const torch::Tensor &L3_grad
    ) {

    int64_t num_batch = L1_in.sizes()[0];
    torch::Tensor L1_grad = torch::empty(L1_in.sizes(), L1_in.options());
    torch::Tensor L2_grad = torch::empty(L2_in.sizes(), L2_in.options());
    torch::Tensor W_grad = torch::empty(W.sizes(), W.options());

    torch::Tensor L1_in_contig = L1_in.contiguous();
    torch::Tensor L2_in_contig = L2_in.contiguous();
    torch::Tensor W_contig = W.contiguous();
    torch::Tensor L3_grad_contig = L3_grad.contiguous();

    jit_instance->internal.backward(
            num_batch, 
            data_ptr(L1_in_contig), data_ptr(L1_grad),
            data_ptr(L2_in_contig), data_ptr(L2_grad),
            data_ptr(W_contig), data_ptr(W_grad),
            data_ptr(L3_grad_contig)
    );

    return tuple(L1_grad, L2_grad, W_grad);
}

// =========================================================== 

class TorchJITConv : public torch::CustomClassHolder {
public:
    Map_t fwd_dict, bwd_dict, kernel_dims;
    JITConvImpl<JITKernel> internal;
    int64_t L3_dim;

    JITConvProduct(string kernel_plaintext, Map_t fwd_dict_i, Map_t bwd_dict_i, Map_t kernel_dims_i) :
        fwd_dict(fwd_dict_i.copy()),
        bwd_dict(bwd_dict_i.copy()),
        kernel_dims(kernel_dims_i.copy()),
        internal(kernel_plaintext, 
            KernelLaunchConfig(
                fwd_dict.at("num_blocks"),
                fwd_dict.at("num_threads"),
                fwd_dict.at("smem")
            ),
            KernelLaunchConfig(
                bwd_dict.at("num_blocks"),
                bwd_dict.at("num_threads"),
                bwd_dict.at("smem")
            )),
        L3_dim(kernel_dims.at("L3_dim")) { }

    tuple<tuple<string, string>, tuple<string, Map_t>, tuple<string, Map_t>, tuple<string, Map_t>> __obj_flatten__() {
        return tuple(tuple("kernel_plaintext", internal.jit.kernel_plaintext),
            tuple("fwd_config", fwd_dict),
            tuple("bwd_config", bwd_dict),
            tuple("kernel_dims", kernel_dims));
    }

    void exec_conv_rawptrs(
            int64_t L1_in, int64_t L2_in, int64_t weights, int64_t L3_out,
            int64_t rows, int64_t cols,
            int64_t nnz, int64_t node_count,
            int64_t workspace) {
        internal.exec_conv(
            reinterpret_cast<void*>(L1_in), 
            reinterpret_cast<void*>(L2_in), 
            reinterpret_cast<void*>(weights), 
            reinterpret_cast<void*>(L3_out),
            reinterpret_cast<void*>(rows), 
            reinterpret_cast<void*>(cols),
            nnz, node_count,
            reinterpret_cast<void*>(workspace));
    }
    void backward_rawptrs(
            int64_t L1_in, int64_t L1_grad,
            int64_t L2_in, int64_t L2_grad,
            int64_t weight, int64_t weight_grad,
            int64_t L3_grad,
            int64_t rows, int64_t cols,
            int64_t nnz, int64_t node_count,
            int64_t workspace,
            int64_t transpose_perm) {
        internal.backward(
            reinterpret_cast<void*>(L1_in), reinterpret_cast<void*>(L1_grad),
            reinterpret_cast<void*>(L2_in), reinterpret_cast<void*>(L2_grad),
            reinterpret_cast<void*>(weight), reinterpret_cast<void*>(weight_grad),
            reinterpret_cast<void*>(L3_grad),
            reinterpret_cast<void*>(rows), 
            reinterpret_cast<void*>(cols),
            nnz, node_count,
            reinterpret_cast<void*>(workspace),
            reinterpret_cast<void*>(transpose_perm));
    }
};

torch::Tensor jit_conv_forward(
        const c10::intrusive_ptr<TorchJITConv> &jit_instance,
        const torch::Tensor &L1_in,
        const torch::Tensor &L2_in,
        const torch::Tensor &W,
        const torch::Tensor &rows,
        const torch::Tensor &cols,
        const torch::Tensor &workspace) {

    int64_t nnz = rows.sizes()[0];
    int64_t node_count = L1_in.sizes()[0];
    torch::Tensor L3_out = torch::zeros({nnz, jit_instance->L3_dim}, L1_in.options());
    
    torch::Tensor L1_contig = L1_in.contiguous();
    torch::Tensor L2_contig = L2_in.contiguous();
    torch::Tensor W_contig = W.contiguous();
    torch::Tensor rows_contig = rows.contiguous();
    torch::Tensor cols_contig = cols.contiguous();
    torch::Tensor workspace_contig = workspace.contiguous();

    jit_instance->internal.exec_conv(
            data_ptr(L1_contig), 
            data_ptr(L2_contig), 
            data_ptr(W_contig), 
            data_ptr(L3_out),
            data_ptr(rows_contig), 
            data_ptr(cols_contig),
            nnz, node_count,
            data_ptr(workspace_contig));

    return L3_out;
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> jit_conv_backward(
        const c10::intrusive_ptr<TorchJITConv> &jit_instance,
        const torch::Tensor &L1_in,
        const torch::Tensor &L2_in,
        const torch::Tensor &W,
        const torch::Tensor &L3_grad,
        const torch::Tensor &rows,
        const torch::Tensor &cols,
        const torch::Tensor &workspace,
        const torch::Tensor &transpose_perm) {
    
    int64_t nnz = rows.sizes()[0];
    int64_t node_count = L1_in.sizes()[0];
    torch::Tensor L1_grad = torch::zeros(L1_in.sizes(), L1_in.options());
    torch::Tensor L2_grad = torch::empty(L2_in.sizes(), L2_in.options());
    torch::Tensor W_grad = torch::empty(W.sizes(), W.options());
    
    torch::Tensor L1_in_contig = L1_in.contiguous();
    torch::Tensor L2_in_contig = L2_in.contiguous();
    torch::Tensor W_contig = W.contiguous();
    torch::Tensor L3_grad_contig = L3_grad.contiguous();
    torch::Tensor rows_contig = rows.contiguous();
    torch::Tensor cols_contig = cols.contiguous();
    torch::Tensor workspace_contig = workspace.contiguous();
    torch::Tensor transpose_perm_contig = transpose_perm.contiguous();
    jit_instance->internal.backward(
            data_ptr(L1_in_contig), data_ptr(L1_grad),
            data_ptr(L2_in_contig), data_ptr(L2_grad),
            data_ptr(W_contig), data_ptr(W_grad),
            data_ptr(L3_grad_contig),
            data_ptr(rows_contig), data_ptr(cols_contig),
            nnz, node_count,
            data_ptr(workspace_contig),
            data_ptr(transpose_perm_contig));

    return tuple(L1_grad, L2_grad, W_grad);
}

// =========================================================== 

TORCH_LIBRARY_FRAGMENT(torch_tp_jit, m) { 
    m.class_<TorchJITProduct>("TorchJITProduct")
        .def(torch::init<string, Map_t, Map_t, Map_t>())
        .def("__obj_flatten__", &TorchJITProduct::__obj_flatten__)
        .def("exec_tensor_product_rawptr", &TorchJITProduct::exec_tensor_product_device_rawptrs)
        .def("backward_rawptr", &TorchJITProduct::backward_device_rawptrs)
        .def("__len__", [](const c10::intrusive_ptr<TorchJITProduct>& test) -> int64_t {
            return 0;
        })
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<TorchJITProduct>& self)
                -> tuple<string, Map_t, Map_t, Map_t> {
                return tuple(self->internal.jit.kernel_plaintext, self->fwd_dict, self->bwd_dict, self->kernel_dims);
            },
            // __setstate__
            [](tuple<string, Map_t, Map_t, Map_t> state)
                -> c10::intrusive_ptr<TorchJITProduct> {
                return c10::make_intrusive<TorchJITProduct>(get<0>(state), get<1>(state), get<2>(state), get<3>(state));
            });

    m.def("jit_tp_forward(__torch__.torch.classes.torch_tp_jit.TorchJITProduct jit, Tensor L1_in, Tensor L2_in, Tensor W) -> Tensor");
    m.def("jit_tp_backward(__torch__.torch.classes.torch_tp_jit.TorchJITProduct jit, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad) -> (Tensor, Tensor, Tensor)");


    m.class_<TorchJITConv>("TorchJITConv")
        .def(torch::init<string, Map_t, Map_t, Map_t>())
        .def("__obj_flatten__", &TorchJITConv::__obj_flatten__)
        .def("exec_conv_rawptrs", &TorchJITConv::exec_conv_rawptrs)
        .def("backward_rawptrs", &TorchJITConv::backward_rawptrs)
        .def("__len__", [](const c10::intrusive_ptr<TorchJITConv>& test) -> int64_t {
            return 0;
        })
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<TorchJITConv>& self)
                -> tuple<string, Map_t, Map_t, Map_t> {
                return tuple(self->internal.jit.kernel_plaintext, self->fwd_dict, self->bwd_dict, self->kernel_dims);
            },
            // __setstate__
            [](tuple<string, Map_t, Map_t, Map_t> state)
                -> c10::intrusive_ptr<TorchJITConv> {
                return c10::make_intrusive<TorchJITConv>(get<0>(state), get<1>(state), get<2>(state), get<3>(state));
            });

    m.def("jit_conv_forward(__torch__.torch.classes.torch_tp_jit.TorchJITConv jit, Tensor L1_in, Tensor L2_in, Tensor W, Tensor rows, Tensor cols, Tensor workspace) -> Tensor");
    m.def("jit_conv_backward(__torch__.torch.classes.torch_tp_jit.TorchJITConv jit, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> (Tensor, Tensor, Tensor)");
};


TORCH_LIBRARY_IMPL(torch_tp_jit, CUDA, m) { 
    m.impl("jit_tp_forward", &jit_tp_forward);
    m.impl("jit_tp_backward", &jit_tp_backward);

    m.impl("jit_conv_forward", &jit_conv_forward);
    m.impl("jit_conv_backward", &jit_conv_backward);
};

PYBIND11_MODULE(torch_tp_jit, m) {}