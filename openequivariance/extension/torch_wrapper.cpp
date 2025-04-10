#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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

#ifdef COMPILE_TORCH
    #include <ATen/Operators.h>
    #include <torch/all.h>
    #include <torch/library.h>

    using Map_t=torch::Dict<string, int64_t>;

    inline void* data_ptr(const torch::Tensor &tensor) {
        if(tensor.dtype() == torch::kFloat)
            return reinterpret_cast<void*>(tensor.data_ptr<float>());
        else if(tensor.dtype() == torch::kDouble)
            return reinterpret_cast<void*>(tensor.data_ptr<double>());
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

        tuple<string, Map_t, Map_t, Map_t> __obj_flatten__() {
            return tuple(internal.jit.kernel_plaintext, fwd_dict, bwd_dict, kernel_dims);
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

    TORCH_LIBRARY_FRAGMENT(torch_wrapper, m) { 
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
                return self->__obj_flatten__(); 
                },
                // __setstate__
                [](tuple<string, Map_t, Map_t, Map_t> state)
                    -> c10::intrusive_ptr<TorchJITProduct> {
                return c10::make_intrusive<TorchJITProduct>(get<0>(state), get<1>(state), get<2>(state), get<3>(state));
                });
        m.def("jit_tp_forward(__torch__.torch.classes.torch_wrapper.TorchJITProduct jit, Tensor L1_in, Tensor L2_in, Tensor W) -> Tensor");
        m.def("jit_tp_backward(__torch__.torch.classes.torch_wrapper.TorchJITProduct jit, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad) -> (Tensor, Tensor, Tensor)");
    };

    TORCH_LIBRARY_IMPL(torch_wrapper, CUDA, m) { 
        m.impl("jit_tp_forward", &jit_tp_forward);
        m.impl("jit_tp_backward", &jit_tp_backward);
    };
#endif

PYBIND11_MODULE(torch_wrapper, m) {
    //=========== Batch tensor products =========
    py::class_<GenericTensorProductImpl>(m, "GenericTensorProductImpl")
        .def("exec_tensor_product_rawptr", &GenericTensorProductImpl::exec_tensor_product_device_rawptrs)
        .def("backward_rawptr", &GenericTensorProductImpl::backward_device_rawptrs);
    py::class_<JITTPImpl<JITKernel>, GenericTensorProductImpl>(m, "JITTPImpl")
        .def(py::init<std::string, std::unordered_map<string, int64_t>, std::unordered_map<string, int64_t>>());

    //============= Convolutions ===============
    py::class_<ConvolutionImpl>(m, "ConvolutionImpl")
        .def("exec_conv_rawptrs", &ConvolutionImpl::exec_conv_rawptrs)
        .def("backward_rawptrs", &ConvolutionImpl::backward_rawptrs);
    py::class_<JITConvImpl<JITKernel>, ConvolutionImpl>(m, "JITConvImpl")
        .def(py::init<std::string, KernelLaunchConfig, KernelLaunchConfig>());

    py::class_<GroupMM<float>>(m, "GroupMM_F32")
        .def(py::init<int, int>())
        .def("group_gemm", &GroupMM<float>::group_gemm_intptr);
    py::class_<GroupMM<double>>(m, "GroupMM_F64")
        .def(py::init<int, int>())
        .def("group_gemm", &GroupMM<double>::group_gemm_intptr);

    py::class_<DeviceProp>(m, "DeviceProp")
        .def(py::init<int>())
        .def_readonly("name", &DeviceProp::name)
        .def_readonly("warpsize", &DeviceProp::warpsize)
        .def_readonly("major", &DeviceProp::major)
        .def_readonly("minor", &DeviceProp::minor)
        .def_readonly("multiprocessorCount", &DeviceProp::multiprocessorCount)
        .def_readonly("maxSharedMemPerBlock", &DeviceProp::maxSharedMemPerBlock); 

    py::class_<PyDeviceBuffer<GPU_Allocator>>(m, "DeviceBuffer")
        .def(py::init<uint64_t>())
        .def(py::init<py::buffer>())
        .def("copy_to_host", &PyDeviceBuffer<GPU_Allocator>::copy_to_host)
        .def("data_ptr", &PyDeviceBuffer<GPU_Allocator>::data_ptr);

    py::class_<GPUTimer>(m, "GPUTimer")
        .def(py::init<>())
        .def("start", &GPUTimer::start)
        .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed)
        .def("clear_L2_cache", &GPUTimer::clear_L2_cache);
}