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

std::unordered_map<int64_t, std::unique_ptr<JITTPImpl<JITKernel>>> kernel_cache;
std::mutex mut;

std::unordered_map<string, int64_t> parse_launch_config(ffi::Dictionary dict) {
    std::unordered_map<string, int64_t> result;
    result["num_blocks"] = dict.get<int64_t>("num_blocks").value();
    result["num_threads"] = dict.get<int64_t>("num_threads").value();
    result["warp_size"] = dict.get<int64_t>("warp_size").value();
    result["smem"] = dict.get<int64_t>("smem").value();
    return result;
}

std::unordered_map<string, int64_t> parse_kernel_prop(ffi::Dictionary dict) { 
    std::unordered_map<string, int64_t> result;
    result["L1_dim"] = dict.get<int64_t>("L1_dim").value();
    result["L2_dim"] = dict.get<int64_t>("L2_dim").value();
    result["L3_dim"] = dict.get<int64_t>("L3_dim").value();
    result["weight_numel"] = dict.get<int64_t>("weight_numel").value();
    result["shared_weights"] = dict.get<int64_t>("shared_weights").value();
    result["opt_level"] = dict.get<int64_t>("opt_level").value();
    result["irrep_dtype"] = dict.get<int64_t>("irrep_dtype").value();
    result["weight_dtype"] = dict.get<int64_t>("weight_dtype").value();
    return result;
}

JITTPImpl<JITKernel>* compile_kernel_with_caching(std::string_view kernel,
                    ffi::Dictionary forward_config, 
                    ffi::Dictionary backward_config, 
                    ffi::Dictionary double_backward_config, 
                    ffi::Dictionary kernel_prop,
                    int64_t hash) {
    
    JITTPImpl<JITKernel>* result = nullptr;
    {
        const std::lock_guard<std::mutex> lock(mut);
        auto it = kernel_cache.find(hash); 
        if (it != kernel_cache.end()) {
            result = it->second.get(); 
        }
        else {
            auto jit_tp_impl = std::make_unique<JITTPImpl<JITKernel>>(
                std::string(kernel),
                parse_launch_config(forward_config),
                parse_launch_config(backward_config),
                parse_launch_config(double_backward_config),
                parse_kernel_prop(kernel_prop));
            result = jit_tp_impl.get();
            kernel_cache.insert({hash, std::move(jit_tp_impl)});
        }
    }
    return result;
}

ffi::Error tp_forward_impl(
        cudaStream_t stream, 
        std::string_view kernel, ffi::Dictionary forward_config, ffi::Dictionary backward_config, ffi::Dictionary double_backward_config, ffi::Dictionary kernel_prop,
        int64_t hash, ffi::ResultBufferR0<ffi::S32> out) {
    
    JITTPImpl<JITKernel>* jit_kernel = compile_kernel_with_caching(
        kernel, forward_config, backward_config, double_backward_config, kernel_prop, hash);

    std::cout << "SUCCESSFULLY COMPILED KERNEL!" << std::endl;
    // TODO: Launch the forward kernel here

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_forward, tp_forward_impl,
    ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Attr<std::string_view>("kernel").Attr<ffi::Dictionary>("forward_config").Attr<ffi::Dictionary>("backward_config").Attr<ffi::Dictionary>("double_backward_config").Attr<ffi::Dictionary>("kernel_prop")
      .Attr<int64_t>("hash")
      .Ret<ffi::BufferR0<ffi::S32>>(),
      {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

NB_MODULE(oeq_jax_extension, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["tp_forward"] = nb::capsule(reinterpret_cast<void *>(tp_forward));
    return registrations;
  });
}
