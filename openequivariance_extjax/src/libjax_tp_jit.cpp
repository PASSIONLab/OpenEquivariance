#include <cstdint>
#include <mutex>
#include <string>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

#define CUDA_BACKEND // Stick to CUDA for now 

#ifdef CUDA_BACKEND
    #include "util/backend_cuda.hpp"
    #include "group_mm_cuda.hpp"
    using JITKernel = CUJITKernel;
    using GPU_Allocator = CUDA_Allocator;

    template<typename T>
    using GroupMM = GroupMMCUDA<T>;
#endif

#include "tensorproducts.hpp"

xla::ffi::DataType enum_to_xla_dtype(int64_t i){
    switch(i) {
        case 1:
            return xla::ffi::DataType::F32; 
        case 2: 
            return xla::ffi::DataType::F64;
        case 3: 
            return xla::ffi::DataType::S32;
        case 4: 
            return xla::ffi::DataType::S64;
        case 5: 
            return xla::ffi::DataType::U8;
    }
    throw logic_error("Unsupported tensor datatype!");
}

inline void* data_ptr(ffi::AnyBuffer &buffer) {
    switch (buffer.element_type()) {
        case xla::ffi::DataType::F32:
            return reinterpret_cast<void*>(buffer.typed_data<float>());
        case xla::ffi::DataType::F64:
            return reinterpret_cast<void*>(buffer.typed_data<double>());
        case xla::ffi::DataType::S64:
            return reinterpret_cast<void*>(buffer.typed_data<int64_t>());
        case xla::ffi::DataType::U8:
            return reinterpret_cast<void*>(buffer.typed_data<uint8_t>());
        default:
            throw logic_error("Unsupported tensor datatype!");
    }
}

inline int byte_count(ffi::AnyBuffer &buffer) {
    switch (buffer.element_type()) {
        case xla::ffi::DataType::F32:
            return 4;
        case xla::ffi::DataType::F64:
            return 8;
        case xla::ffi::DataType::S64:
            return 8;
        case xla::ffi::DataType::U8:
            return 1;
        default:
            throw logic_error("Unsupported tensor datatype!");
    }
}

inline void* data_ptr(ffi::Result<ffi::AnyBuffer> &buffer) {
    return data_ptr(*buffer);
}

#ifdef CUDA_BACKEND
void zero_buffer(ffi::AnyBuffer &buffer) {
    cudaMemset(
        data_ptr(buffer), 
        0, 
        buffer.element_count() * byte_count(buffer));
}
#endif

struct KernelProp {
    int64_t L1_dim, L2_dim, L3_dim, weight_numel;
    bool shared_weights;
    xla::ffi::DataType irrep_dtype;
    xla::ffi::DataType weight_dtype;

    int64_t workspace_size;     // Convolution only
    bool deterministic;
    xla::ffi::DataType idx_dtype;
    xla::ffi::DataType workspace_dtype;

    KernelProp() {}

    KernelProp(
        std::unordered_map<string, int64_t> &kernel_dims, bool is_convolution):
            L1_dim(kernel_dims.at("L1_dim")),
            L2_dim(kernel_dims.at("L2_dim")),    
            L3_dim(kernel_dims.at("L3_dim")),
            weight_numel(kernel_dims.at("weight_numel")),
            shared_weights(kernel_dims.at("shared_weights")),
            irrep_dtype(enum_to_xla_dtype(kernel_dims.at("irrep_dtype"))),
            weight_dtype(enum_to_xla_dtype(kernel_dims.at("weight_dtype"))),
            workspace_dtype(xla::ffi::DataType::U8) { 
        if(is_convolution) {
            workspace_size = kernel_dims.at("workspace_size");
            deterministic = kernel_dims.at("deterministic");
            idx_dtype = enum_to_xla_dtype(kernel_dims.at("idx_dtype"));
        }
    }
};

std::unordered_map<int64_t,
    std::pair<
        std::unique_ptr<JITTPImpl<JITKernel>>,
        KernelProp
    >> kernel_cache;
std::mutex mut;

std::vector<std::string> launch_config_keys = {
    "num_blocks", 
    "num_threads", 
    "smem"};
std::vector<std::string> kernel_prop_keys = {
    "L1_dim", 
    "L2_dim", 
    "L3_dim", 
    "weight_numel", 
    "shared_weights", 
    "opt_level", 
    "irrep_dtype", 
    "weight_dtype"};

std::unordered_map<string, int64_t> parse_ffi_dict(ffi::Dictionary &dict, const std::vector<string> &keys) {
    std::unordered_map<string, int64_t> result;
    for (const auto &key : keys) {
        result[key] = dict.get<int64_t>(key).value();
    }
    return result;
}

std::pair<JITTPImpl<JITKernel>*, KernelProp> 
    compile_kernel_with_caching(std::string_view kernel,
                    ffi::Dictionary forward_config, 
                    ffi::Dictionary backward_config, 
                    ffi::Dictionary double_backward_config, 
                    ffi::Dictionary kernel_prop,
                    int64_t hash,
                    bool is_convolution) {
    
    {
        const std::lock_guard<std::mutex> lock(mut);
        auto it = kernel_cache.find(hash); 
        if (it == kernel_cache.end()) {
            auto kernel_prop_map = parse_ffi_dict(kernel_prop, kernel_prop_keys);
            auto jit_tp_impl = std::make_unique<JITTPImpl<JITKernel>>(
                std::string(kernel),
                parse_ffi_dict(forward_config, launch_config_keys),
                parse_ffi_dict(backward_config, launch_config_keys),
                parse_ffi_dict(double_backward_config, launch_config_keys),
                kernel_prop_map);
            kernel_cache.insert({hash,
                std::make_pair(std::move(jit_tp_impl), 
                KernelProp(kernel_prop_map, is_convolution))});
            it = kernel_cache.find(hash);
        }
        return {it->second.first.get(), it->second.second};
    }
}


inline void check_tensor(const ffi::AnyBuffer &buffer, 
                            std::initializer_list<int64_t> expected_shape,
                            xla::ffi::DataType expected_dtype,
                            std::string tensor_name) {
    const ffi::AnyBuffer::Dimensions dims = buffer.dimensions();
    if (dims.size() != expected_shape.size()) {
        throw std::logic_error("Rank mismatch for tensor '"
            + tensor_name 
            + "'. Expected rank " 
            + std::to_string(expected_shape.size()) 
            + ", got rank " 
            + std::to_string(dims.size()));
    }

    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i] != expected_shape.begin()[i]) {
            throw std::logic_error("Shape mismatch for tensor '"
                + tensor_name 
                + "'. Expected dimension " 
                + std::to_string(expected_shape.begin()[i]) 
                + " at index " 
                + std::to_string(i) 
                + ", got " 
                + std::to_string(dims[i]));
        }
    }

    if (buffer.element_type() != expected_dtype) {
        throw std::logic_error("Datatype mismatch.");
    }
}

ffi::Error tp_forward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::Result<ffi::AnyBuffer> L3_out,
        cudaStream_t stream, 
        std::string_view kernel, ffi::Dictionary forward_config, ffi::Dictionary backward_config, ffi::Dictionary double_backward_config, ffi::Dictionary kernel_prop,
        int64_t hash) {
   
    auto [jit_kernel, k] = compile_kernel_with_caching(
        kernel, forward_config, backward_config, double_backward_config, kernel_prop, hash, false);
    const int64_t num_batch = L1_in.dimensions()[0];

    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in"); 

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else 
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");

    jit_kernel->exec_tensor_product(
            num_batch,
            data_ptr(L1_in),
            data_ptr(L2_in),
            data_ptr(L3_out),
            data_ptr(W),
            stream);

    return ffi::Error::Success();
}

ffi::Error tp_backward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::AnyBuffer L3_grad,
        ffi::Result<ffi::AnyBuffer> L1_grad,
        ffi::Result<ffi::AnyBuffer> L2_grad,
        ffi::Result<ffi::AnyBuffer> W_grad, 
        cudaStream_t stream, 
        std::string_view kernel, ffi::Dictionary forward_config, ffi::Dictionary backward_config, ffi::Dictionary double_backward_config, ffi::Dictionary kernel_prop,
        int64_t hash) {
    
    auto [jit_kernel, k] = compile_kernel_with_caching(
        kernel, forward_config, backward_config, double_backward_config, kernel_prop, hash, false);
    const int64_t num_batch = L1_in.dimensions()[0];
    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {num_batch, k.L3_dim}, k.irrep_dtype, "L3_grad");

    if (k.shared_weights) {
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(*W_grad, {k.weight_numel}, k.weight_dtype, "W_grad");
    }
    else {
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(*W_grad, {num_batch, k.weight_numel}, k.weight_dtype, "W_grad");
    }

    if (k.shared_weights) {
        zero_buffer(*W_grad);
    } 

    jit_kernel->backward(
            num_batch,
            data_ptr(L1_in),
            data_ptr(L1_grad),
            data_ptr(L2_in),
            data_ptr(L2_grad),
            data_ptr(W),
            data_ptr(W_grad),
            data_ptr(L3_grad),
            stream);
    return ffi::Error::Success();
}


ffi::Error tp_double_backward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::AnyBuffer L3_grad,
        ffi::AnyBuffer L1_dgrad,
        ffi::AnyBuffer L2_dgrad,
        ffi::AnyBuffer W_dgrad,
        ffi::Result<ffi::AnyBuffer> L1_grad,
        ffi::Result<ffi::AnyBuffer> L2_grad,
        ffi::Result<ffi::AnyBuffer> W_grad,
        ffi::Result<ffi::AnyBuffer> L3_dgrad,
        cudaStream_t stream, 
        std::string_view kernel, ffi::Dictionary forward_config, ffi::Dictionary backward_config, ffi::Dictionary double_backward_config, ffi::Dictionary kernel_prop,
        int64_t hash) {
    
    auto [jit_kernel, k] = compile_kernel_with_caching(
        kernel, forward_config, backward_config, double_backward_config, kernel_prop, hash, false);
    const int64_t num_batch = L1_in.dimensions()[0];
    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {num_batch, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(L1_dgrad, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_dgrad");
    check_tensor(L2_dgrad, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_dgrad");

    if (k.shared_weights){
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {k.weight_numel}, k.weight_dtype, "W_dgrad");
    } else {
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {num_batch, k.weight_numel}, k.weight_dtype, "W_dgrad");
    }

    if (k.shared_weights) {
        zero_buffer(*W_grad);
    } 

    jit_kernel->double_backward(
            num_batch,
            data_ptr(L1_in),
            data_ptr(L2_in),
            data_ptr(W),
            data_ptr(L3_grad),
            data_ptr(L1_dgrad),
            data_ptr(L2_dgrad),
            data_ptr(W_dgrad),
            data_ptr(L1_grad),
            data_ptr(L2_grad),
            data_ptr(W_grad),
            data_ptr(L3_dgrad),
            stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_forward, tp_forward_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::string_view>("kernel").Attr<ffi::Dictionary>("forward_config").Attr<ffi::Dictionary>("backward_config").Attr<ffi::Dictionary>("double_backward_config").Attr<ffi::Dictionary>("kernel_prop")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_backward, tp_backward_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::string_view>("kernel").Attr<ffi::Dictionary>("forward_config").Attr<ffi::Dictionary>("backward_config").Attr<ffi::Dictionary>("double_backward_config").Attr<ffi::Dictionary>("kernel_prop")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_double_backward, tp_double_backward_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::string_view>("kernel").Attr<ffi::Dictionary>("forward_config").Attr<ffi::Dictionary>("backward_config").Attr<ffi::Dictionary>("double_backward_config").Attr<ffi::Dictionary>("kernel_prop")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

NB_MODULE(openequivariance_extjax, m) {
    m.def("registrations", []() {
        nb::dict registrations;
        registrations["tp_forward"] = nb::capsule(reinterpret_cast<void *>(tp_forward));
        registrations["tp_backward"] = nb::capsule(reinterpret_cast<void *>(tp_backward));
        registrations["tp_double_backward"] = nb::capsule(reinterpret_cast<void *>(tp_double_backward));
        return registrations;
    });

    nb::class_<DeviceProp>(m, "DeviceProp")
        .def(nb::init<int>())
        .def_ro("name", &DeviceProp::name)
        .def_ro("warpsize", &DeviceProp::warpsize)
        .def_ro("major", &DeviceProp::major)
        .def_ro("minor", &DeviceProp::minor)
        .def_ro("multiprocessorCount", &DeviceProp::multiprocessorCount)
        .def_ro("maxSharedMemPerBlock", &DeviceProp::maxSharedMemPerBlock); 

    nb::class_<GPUTimer>(m, "GPUTimer")
        .def(nb::init<>())
        .def("start", &GPUTimer::start)
        .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed)
        .def("clear_L2_cache", &GPUTimer::clear_L2_cache);

    /*nb::class_<PyDeviceBuffer<GPU_Allocator>>(m, "DeviceBuffer")
        .def(nb::init<uint64_t>())
        .def(nb::init<nb::buffer>())
        .def("copy_to_host", &PyDeviceBuffer<GPU_Allocator>::copy_to_host)
        .def("data_ptr", &PyDeviceBuffer<GPU_Allocator>::data_ptr);*/
}
