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
            auto result = parse_ffi_dict(forward_config, launch_config_keys);

            cout << result["smem"] << endl;
            /*auto jit_tp_impl = std::make_unique<JITTPImpl<JITKernel>>(
                std::string(kernel),
                parse_ffi_dict(forward_config, launch_config_keys),
                parse_ffi_dict(backward_config, launch_config_keys),
                parse_ffi_dict(double_backward_config, launch_config_keys),
                parse_ffi_dict(kernel_prop, kernel_prop_keys));
            result = jit_tp_impl.get();
            kernel_cache.insert({hash, std::move(jit_tp_impl)});*/
        }
    }
    return result;
}

ffi::Error tp_forward_impl(
        cudaStream_t stream, 
        std::string_view kernel, ffi::Dictionary forward_config, ffi::Dictionary backward_config, ffi::Dictionary double_backward_config, ffi::Dictionary kernel_prop,
        int64_t hash, ffi::ResultBufferR0<ffi::S32> out) {
    
    auto jit_kernel = compile_kernel_with_caching(
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

NB_MODULE(openequivariance_extjax, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["tp_forward"] = nb::capsule(reinterpret_cast<void *>(tp_forward));
    return registrations;
  });
}
