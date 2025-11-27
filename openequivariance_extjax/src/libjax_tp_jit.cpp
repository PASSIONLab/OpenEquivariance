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

namespace nb = nanobind;
namespace ffi = xla::ffi;

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

    KernelProp(Map_t &kernel_dims, bool is_convolution):
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
    > kernel_cache;
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
            auto kernel_prop = parse_ffi_dict(kernel_prop, kernel_prop_keys);
            auto jit_tp_impl = std::make_unique<JITTPImpl<JITKernel>>(
                std::string(kernel),
                parse_ffi_dict(forward_config, launch_config_keys),
                parse_ffi_dict(backward_config, launch_config_keys),
                parse_ffi_dict(double_backward_config, launch_config_keys),
                kernel_prop);
            kernel_cache.insert({hash,
                std::make_pair(std::move(jit_tp_impl), 
                KernelProp(kernel_prop, is_convolution))});
            it = kernel_cache.find(hash);
        }
    }
    return {it->second.first.get(), it->second.second};
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
        if (dims[i] != expected_shape[i]) {
            throw std::logic_error("Shape mismatch for tensor '"
                + tensor_name 
                + "'. Expected dimension " 
                + std::to_string(expected_shape[i]) 
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
    const int64_t num_batch = L1_in.dimensions[0];

    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in"); 

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else 
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");

    // TODO: Launch the forward kernel here


    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_forward, tp_forward_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Result<ffi::AnyBuffer>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::string_view>("kernel").Attr<ffi::Dictionary>("forward_config").Attr<ffi::Dictionary>("backward_config").Attr<ffi::Dictionary>("double_backward_config").Attr<ffi::Dictionary>("kernel_prop")
        .Attr<int64_t>("hash")
        .Ret<ffi::BufferR0<ffi::S32>>(),
        {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

NB_MODULE(openequivariance_extjax, m) {
    m.def("registrations", []() {
        nb::dict registrations;
        registrations["tp_forward"] = nb::capsule(reinterpret_cast<void *>(tp_forward));
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
