#include <cstdint>
#include <mutex>
#include <string>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <iostream>

#include "xla/ffi/api/ffi.h"
#include "json11/json11.hpp"
#include "ffi_handler_table.h"

namespace ffi = xla::ffi;
using json = json11::Json;

#ifdef CUDA_BACKEND
    #include <cuda.h>
    #include <cuda_runtime.h>

    #include "backend/backend_cuda.hpp"
    using JITKernel = CUJITKernel;
    using GPU_Allocator = CUDA_Allocator;
    using stream_t = cudaStream_t;
#endif

#ifdef HIP_BACKEND
    #include "backend/backend_hip.hpp"
    using JITKernel = HIPJITKernel;
    using GPU_Allocator = HIP_Allocator;
    using stream_t = hipStream_t;
#endif

#include "tensorproducts.hpp"
#include "convolution.hpp"
#include "factorized_projected.hpp"

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

std::string xla_dtype_to_string(xla::ffi::DataType dtype) {
    const std::unordered_map<xla::ffi::DataType, std::string> map = {
        {xla::ffi::DataType::INVALID, "INVALID"},
        {xla::ffi::DataType::PRED, "PRED"},
        {xla::ffi::DataType::S8, "S8"},
        {xla::ffi::DataType::S16, "S16"},
        {xla::ffi::DataType::S32, "S32"},
        {xla::ffi::DataType::S64, "S64"},
        {xla::ffi::DataType::U8, "U8"},
        {xla::ffi::DataType::U16, "U16"},
        {xla::ffi::DataType::U32, "U32"},
        {xla::ffi::DataType::U64, "U64"},
        {xla::ffi::DataType::F16, "F16"},
        {xla::ffi::DataType::F32, "F32"},
        {xla::ffi::DataType::F64, "F64"},
        {xla::ffi::DataType::BF16, "BF16"},
        {xla::ffi::DataType::C64, "C64"},
        {xla::ffi::DataType::C128, "C128"},
        {xla::ffi::DataType::TOKEN, "TOKEN"},
        {xla::ffi::DataType::F8E5M2, "F8E5M2"},
        {xla::ffi::DataType::F8E4M3, "F8E4M3"},
        {xla::ffi::DataType::F8E4M3FN, "F8E4M3FN"},
        {xla::ffi::DataType::F8E4M3B11FNUZ, "F8E4M3B11FNUZ"},
        {xla::ffi::DataType::F8E5M2FNUZ, "F8E5M2FNUZ"},
        {xla::ffi::DataType::F8E4M3FNUZ, "F8E4M3FNUZ"},
        {xla::ffi::DataType::F8E3M4, "F8E3M4"},
        {xla::ffi::DataType::F4E2M1FN, "F4E2M1FN"},
        {xla::ffi::DataType::F8E8M0FNU, "F8E8M0FNU"},
    };
    return map.at(dtype);
}

inline void* data_ptr(ffi::AnyBuffer &buffer) {
    return buffer.untyped_data();
}

inline void* data_ptr(const ffi::AnyBuffer &buffer) {
    return const_cast<void*>(buffer.untyped_data());
}

inline void* data_ptr(ffi::Result<ffi::AnyBuffer> &buffer) {
    return data_ptr(*buffer);
}

inline int byte_count(ffi::AnyBuffer &buffer) {
    switch (buffer.element_type()) {
        case xla::ffi::DataType::U32:
        case xla::ffi::DataType::S32:
        case xla::ffi::DataType::F32:
            return 4;
        case xla::ffi::DataType::F64:
        case xla::ffi::DataType::S64:
            return 8;
        case xla::ffi::DataType::U8:
            return 1;
        default:
            throw logic_error("Unsupported tensor datatype!");
    }
}

#ifdef CUDA_BACKEND
void zero_buffer(ffi::AnyBuffer &buffer, stream_t stream) {
    cudaMemsetAsync(
        data_ptr(buffer), 
        0, 
        buffer.element_count() * byte_count(buffer),
        stream);
}
#endif
#ifdef HIP_BACKEND
void zero_buffer(ffi::AnyBuffer &buffer, stream_t stream) {
    std::ignore = hipMemsetAsync(
        data_ptr(buffer), 
        0, 
        buffer.element_count() * byte_count(buffer),
        stream);
}
#endif

std::unordered_map<std::string, int64_t> parse_json_config(const json &j_obj) {
    std::unordered_map<std::string, int64_t> result;
    for (const auto &kv : j_obj.object_items()) {
        result[kv.first] = static_cast<int64_t>(kv.second.number_value());
    }
    return result;
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
    >> tp_cache;

std::unordered_map<int64_t,
    std::pair<
        std::unique_ptr<JITConvImpl<JITKernel>>,
        KernelProp
    >> conv_cache;

std::unordered_map<int64_t, std::unique_ptr<JITFactorizedProjectedImpl<JITKernel>>>
    factorized_projected_cache;
std::mutex mut;

template <typename Cache, typename Factory>
typename Cache::mapped_type &find_or_compile_cached(
    Cache &cache, int64_t hash, Factory &&factory) {
    const std::lock_guard<std::mutex> lock(mut);
    auto it = cache.find(hash);
    if (it == cache.end()) {
        it = cache.emplace(hash, std::forward<Factory>(factory)()).first;
    }
    return it->second;
}

std::pair<JITTPImpl<JITKernel>*, KernelProp> 
    compile_tp_with_caching(std::string_view json_payload,
                    int64_t hash,
                    bool is_convolution) {
    
    auto &cached = find_or_compile_cached(
        tp_cache, hash, [&] {
            std::string err;
            json root = json::parse(std::string(json_payload), err);
            if (!err.empty()) throw std::runtime_error("JSON Parse Error: " + err);

            std::string kernel_src = root["kernel"].string_value();
            auto forward_cfg = parse_json_config(root["forward_config"]);
            auto backward_cfg = parse_json_config(root["backward_config"]);
            auto dbackward_cfg = parse_json_config(root["double_backward_config"]);
            auto kernel_prop_map = parse_json_config(root["kernel_prop"]);

            auto jit_tp_impl = std::make_unique<JITTPImpl<JITKernel>>(
                kernel_src,
                forward_cfg,
                backward_cfg,
                dbackward_cfg,
                kernel_prop_map);
            return std::make_pair(std::move(jit_tp_impl),
                                  KernelProp(kernel_prop_map, is_convolution));
        });
    return {cached.first.get(), cached.second};
}

std::pair<JITConvImpl<JITKernel>*, KernelProp> 
    compile_conv_with_caching(std::string_view json_payload,
                    int64_t hash,
                    bool is_convolution) {
    
    auto &cached = find_or_compile_cached(
        conv_cache, hash, [&] {
            std::string err;
            json root = json::parse(std::string(json_payload), err);
            if (!err.empty()) throw std::runtime_error("JSON Parse Error: " + err);

            std::string kernel_src = root["kernel"].string_value();
            auto forward_cfg = parse_json_config(root["forward_config"]);
            auto backward_cfg = parse_json_config(root["backward_config"]);
            auto dbackward_cfg = parse_json_config(root["double_backward_config"]);
            auto kernel_prop_map = parse_json_config(root["kernel_prop"]);

            auto jit_conv_impl = std::make_unique<JITConvImpl<JITKernel>>(
                kernel_src,
                forward_cfg,
                backward_cfg,
                dbackward_cfg,
                kernel_prop_map);
            return std::make_pair(std::move(jit_conv_impl),
                                  KernelProp(kernel_prop_map, is_convolution));
        });
    return {cached.first.get(), cached.second};
}

JITFactorizedProjectedImpl<JITKernel>* compile_factorized_projected_with_caching(
    std::string_view source, int64_t hash, int64_t num_threads,
    int64_t logical_cohort_width, int64_t shared_memory_bytes) {
    auto& cached = find_or_compile_cached(
        factorized_projected_cache, hash, [&] {
            return std::make_unique<JITFactorizedProjectedImpl<JITKernel>>(
                std::string(source), num_threads, logical_cohort_width,
                shared_memory_bytes);
        });
    return cached.get();
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
        throw std::logic_error("Datatype mismatch for tensor " + tensor_name +
            ". Expected datatype " + xla_dtype_to_string(expected_dtype) + 
            ", got " + xla_dtype_to_string(buffer.element_type()));
    }
}

// --------------------- Tensor Products --------------------------
ffi::Error tp_forward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::Result<ffi::AnyBuffer> L3_out,
        stream_t stream,
        std::string_view kernel_json,
        int64_t hash) {
   
    auto [jit_kernel, k] = compile_tp_with_caching(
        kernel_json, hash, false);
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
        stream_t stream, 
        std::string_view kernel_json,
        int64_t hash) {
    
    auto [jit_kernel, k] = compile_tp_with_caching(
        kernel_json, hash, false);
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

    zero_buffer(*L1_grad, stream);
    zero_buffer(*L2_grad, stream);
    zero_buffer(*W_grad, stream);

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
        stream_t stream, 
        std::string_view kernel_json,
        int64_t hash) {
    
    auto [jit_kernel, k] = compile_tp_with_caching(
        kernel_json, hash, false);
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

    zero_buffer(*L1_grad, stream);
    zero_buffer(*L2_grad, stream);
    zero_buffer(*W_grad, stream);
    zero_buffer(*L3_dgrad, stream);

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
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Attr<std::string_view>("kernel")
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
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Attr<std::string_view>("kernel")
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
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

// --------------------- Convolution --------------------------
ffi::Error conv_forward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::AnyBuffer rows,
        ffi::AnyBuffer cols,
        ffi::AnyBuffer workspace,
        ffi::AnyBuffer transpose_perm,
        ffi::Result<ffi::AnyBuffer> L3_out,
        stream_t stream, 
        std::string_view kernel_json,
        int64_t hash) {
   
    auto [jit_kernel, k] = compile_conv_with_caching(
        kernel_json, hash, true);
    const int64_t nnz = rows.dimensions()[0];
    const int64_t node_count = L1_in.dimensions()[0];
    void* workspace_ptr = data_ptr(workspace);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic){
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    }
    else {
        workspace_ptr = nullptr;
    }
    zero_buffer(*L3_out, stream);

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else 
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");

    jit_kernel->exec_conv(
            data_ptr(L1_in),
            data_ptr(L2_in),
            data_ptr(W),
            data_ptr(L3_out),
            data_ptr(rows),
            data_ptr(cols),
            nnz, node_count,
            workspace_ptr,
            stream);

    return ffi::Error::Success();
}

ffi::Error conv_backward_impl(
        ffi::AnyBuffer L1_in,
        ffi::AnyBuffer L2_in,
        ffi::AnyBuffer W,
        ffi::AnyBuffer L3_grad,
        ffi::Result<ffi::AnyBuffer> L1_grad,
        ffi::Result<ffi::AnyBuffer> L2_grad,
        ffi::Result<ffi::AnyBuffer> W_grad, 
        ffi::AnyBuffer rows,
        ffi::AnyBuffer cols,
        ffi::AnyBuffer workspace,
        ffi::AnyBuffer transpose_perm,
        stream_t stream, 
        std::string_view kernel_json,
        int64_t hash) {
    
    auto [jit_kernel, k] = compile_conv_with_caching(
        kernel_json, hash, true);
    const int64_t nnz = rows.dimensions()[0];
    const int64_t node_count = L1_in.dimensions()[0];
    void* workspace_ptr = data_ptr(workspace);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {node_count, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic) {
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    }
    else {
        workspace_ptr = nullptr;
    }
    zero_buffer(*L1_grad, stream);
    zero_buffer(*L2_grad, stream);
    zero_buffer(*W_grad, stream);

    if (k.shared_weights) {
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(*W_grad, {k.weight_numel}, k.weight_dtype, "W_grad");
    }
    else {
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(*W_grad, {nnz, k.weight_numel}, k.weight_dtype, "W_grad");
    }

    jit_kernel->backward(
            data_ptr(L1_in),
            data_ptr(L1_grad),
            data_ptr(L2_in),
            data_ptr(L2_grad),
            data_ptr(W),
            data_ptr(W_grad),
            data_ptr(L3_grad),
            data_ptr(rows),
            data_ptr(cols),
            nnz, node_count,
            workspace_ptr,
            data_ptr(transpose_perm),
            stream);
    return ffi::Error::Success();
}

ffi::Error conv_double_backward_impl(
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
        ffi::AnyBuffer rows,
        ffi::AnyBuffer cols,
        ffi::AnyBuffer workspace,
        ffi::AnyBuffer transpose_perm,
        stream_t stream, 
        std::string_view kernel_json,
        int64_t hash) {
    
    auto [jit_kernel, k] = compile_conv_with_caching(
        kernel_json, hash, true);
    const int64_t nnz = rows.dimensions()[0];
    const int64_t node_count = L1_in.dimensions()[0];
    void* workspace_ptr = data_ptr(workspace);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {node_count, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(L1_dgrad, {node_count, k.L1_dim}, k.irrep_dtype, "L1_dgrad");
    check_tensor(L2_dgrad, {nnz, k.L2_dim}, k.irrep_dtype, "L2_dgrad");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic) {
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    }
    else {
        workspace_ptr = nullptr;
    }
    zero_buffer(*L1_grad, stream);
    zero_buffer(*L2_grad, stream);
    zero_buffer(*W_grad, stream);
    zero_buffer(*L3_dgrad, stream);
    
    if (k.shared_weights) {
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {k.weight_numel}, k.weight_dtype, "W_dgrad");
    } else {
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {nnz, k.weight_numel}, k.weight_dtype, "W_dgrad");
    }

    jit_kernel->double_backward(
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
            data_ptr(rows),
            data_ptr(cols),
            nnz, node_count,
            workspace_ptr,
            data_ptr(transpose_perm),
            stream);
    return ffi::Error::Success();
}

// --------------------- FFI Bindings --------------------------

ffi::Error tp_initialize_impl(ffi::RemainingArgs, ffi::RemainingRets, stream_t,
                              std::string_view kernel_json, int64_t hash) {
    compile_tp_with_caching(kernel_json, hash, false);
    return ffi::Error::Success();
}

ffi::Error conv_initialize_impl(ffi::RemainingArgs, ffi::RemainingRets, stream_t,
                                std::string_view kernel_json, int64_t hash) {
    compile_conv_with_caching(kernel_json, hash, true);
    return ffi::Error::Success();
}

#define OEQ_STOCK_INITIALIZE_ATTRIBUTES                                                \
    .RemainingArgs()                                                                   \
        .RemainingRets()                                                               \
        .Ctx<ffi::PlatformStream<stream_t>>()                                          \
        .Attr<std::string_view>("kernel")                                              \
        .Attr<int64_t>("hash")

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_initialize, tp_initialize_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>() OEQ_STOCK_INITIALIZE_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_initialize, conv_initialize_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>() OEQ_STOCK_INITIALIZE_ATTRIBUTES);

#undef OEQ_STOCK_INITIALIZE_ATTRIBUTES

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_forward, conv_forward_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_backward, conv_backward_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    conv_double_backward, conv_double_backward_impl,
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
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<stream_t>>()
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("hash"),
        {xla::ffi::Traits::kCmdBufferCompatible});

// ------------------- Generated factorized convolution -------------------
void validate_projected_inputs(ffi::AnyBuffer &x, ffi::AnyBuffer &sh,
                               ffi::AnyBuffer &senders, int64_t input_dim,
                               int64_t edge_dim) {
    if (x.dimensions().size() != 2 || sh.dimensions().size() != 2) {
        throw std::logic_error("projected factorized inputs must have rank two");
    }
    if (x.element_type() != xla::ffi::DataType::F32 &&
        x.element_type() != xla::ffi::DataType::F64) {
        throw std::logic_error("projected factorized kernels support only f32 and f64");
    }
    const int64_t edge_count = sh.dimensions()[0];
    check_tensor(x, {x.dimensions()[0], input_dim}, x.element_type(), "x");
    check_tensor(sh, {edge_count, edge_dim}, x.element_type(), "sh");
    check_tensor(senders, {edge_count}, xla::ffi::DataType::S32, "senders");
}

ffi::Error factorized_projected_forward_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer sh, ffi::AnyBuffer weights, ffi::AnyBuffer senders,
    ffi::AnyBuffer row_ptr, ffi::Result<ffi::AnyBuffer> out, stream_t stream,
    std::string_view source, int64_t hash, int64_t channels,
    int64_t input_dim, int64_t edge_dim, int64_t weight_dim, int64_t output_dim,
    int64_t num_threads, int64_t logical_cohort_width,
    int64_t shared_memory_bytes) {
    validate_projected_inputs(x, sh, senders, input_dim, edge_dim);
    const int64_t node_count = x.dimensions()[0], edge_count = sh.dimensions()[0];
    check_tensor(weights, {edge_count, weight_dim}, x.element_type(), "weights");
    check_tensor(row_ptr, {node_count + 1}, xla::ffi::DataType::S32, "row_ptr");
    check_tensor(*out, {node_count, output_dim}, x.element_type(), "out");
    auto* jit_kernel = compile_factorized_projected_with_caching(
        source, hash, num_threads, logical_cohort_width, shared_memory_bytes);
    jit_kernel->forward(
        node_count, edge_count, channels, data_ptr(x), data_ptr(sh), data_ptr(weights),
        data_ptr(senders), data_ptr(row_ptr), data_ptr(out), stream);
    return ffi::Error::Success();
}

ffi::Error factorized_projected_forward_jvp_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer sh, ffi::AnyBuffer weights, ffi::AnyBuffer senders,
    ffi::AnyBuffer row_ptr, const ffi::AnyBuffer* tx, const ffi::AnyBuffer* tsh,
    const ffi::AnyBuffer* tweights, ffi::Result<ffi::AnyBuffer> out, stream_t stream,
    std::string_view source, int64_t hash, int64_t channels,
    int64_t input_dim, int64_t edge_dim, int64_t weight_dim, int64_t output_dim,
    int64_t num_threads, int64_t logical_cohort_width,
    int64_t shared_memory_bytes) {
    validate_projected_inputs(x, sh, senders, input_dim, edge_dim);
    const int64_t node_count = x.dimensions()[0], edge_count = sh.dimensions()[0];
    check_tensor(weights, {edge_count, weight_dim}, x.element_type(), "weights");
    check_tensor(row_ptr, {node_count + 1}, xla::ffi::DataType::S32, "row_ptr");
    if (tx != nullptr) check_tensor(*tx, {node_count, input_dim}, x.element_type(), "tx");
    if (tsh != nullptr) check_tensor(*tsh, {edge_count, edge_dim}, x.element_type(), "tsh");
    if (tweights != nullptr) {
        check_tensor(*tweights, {edge_count, weight_dim}, x.element_type(), "tweights");
    }
    check_tensor(*out, {node_count, output_dim}, x.element_type(), "out");
    auto* jit_kernel = compile_factorized_projected_with_caching(
        source, hash, num_threads, logical_cohort_width, shared_memory_bytes);
    jit_kernel->forward_jvp(
        node_count, edge_count, channels, data_ptr(x), data_ptr(sh), data_ptr(weights),
        data_ptr(senders), data_ptr(row_ptr), tx == nullptr ? nullptr : data_ptr(*tx),
        tsh == nullptr ? nullptr : data_ptr(*tsh),
        tweights == nullptr ? nullptr : data_ptr(*tweights), data_ptr(out), stream);
    return ffi::Error::Success();
}

ffi::Error factorized_projected_backward_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer sh, ffi::AnyBuffer weights, ffi::AnyBuffer senders,
    ffi::AnyBuffer receivers, ffi::AnyBuffer dout, ffi::Result<ffi::AnyBuffer> dx,
    ffi::Result<ffi::AnyBuffer> dsh, ffi::Result<ffi::AnyBuffer> dweights,
    stream_t stream, std::string_view source, int64_t hash,
    int64_t channels, int64_t input_dim, int64_t edge_dim, int64_t weight_dim,
    int64_t output_dim, int64_t num_threads, int64_t logical_cohort_width,
    int64_t shared_memory_bytes) {
    validate_projected_inputs(x, sh, senders, input_dim, edge_dim);
    const int64_t node_count = x.dimensions()[0], edge_count = sh.dimensions()[0];
    check_tensor(weights, {edge_count, weight_dim}, x.element_type(), "weights");
    check_tensor(receivers, {edge_count}, xla::ffi::DataType::S32, "receivers");
    check_tensor(dout, {node_count, output_dim}, x.element_type(), "dout");
    check_tensor(*dx, {node_count, input_dim}, x.element_type(), "dx");
    check_tensor(*dsh, {edge_count, edge_dim}, x.element_type(), "dsh");
    check_tensor(*dweights, {edge_count, weight_dim}, x.element_type(), "dweights");
    zero_buffer(*dx, stream);
    zero_buffer(*dsh, stream);
    zero_buffer(*dweights, stream);
    auto* jit_kernel = compile_factorized_projected_with_caching(
        source, hash, num_threads, logical_cohort_width, shared_memory_bytes);
    jit_kernel->backward(
        node_count, edge_count, data_ptr(x), data_ptr(sh), data_ptr(weights), data_ptr(senders),
        data_ptr(receivers), data_ptr(dout), data_ptr(dx), data_ptr(dsh),
        data_ptr(dweights), stream);
    return ffi::Error::Success();
}

ffi::Error factorized_projected_backward_jvp_impl(
    ffi::AnyBuffer x, ffi::AnyBuffer sh, ffi::AnyBuffer weights, ffi::AnyBuffer senders,
    ffi::AnyBuffer receivers, ffi::AnyBuffer dout, const ffi::AnyBuffer* tx,
    const ffi::AnyBuffer* tsh, const ffi::AnyBuffer* tweights,
    const ffi::AnyBuffer* tdout,
    ffi::Result<ffi::AnyBuffer> tdx, ffi::Result<ffi::AnyBuffer> tdsh,
    ffi::Result<ffi::AnyBuffer> tdweights, stream_t stream,
    std::string_view source, int64_t hash, int64_t channels, int64_t input_dim,
    int64_t edge_dim, int64_t weight_dim, int64_t output_dim,
    int64_t num_threads, int64_t logical_cohort_width,
    int64_t shared_memory_bytes) {
    validate_projected_inputs(x, sh, senders, input_dim, edge_dim);
    const int64_t node_count = x.dimensions()[0], edge_count = sh.dimensions()[0];
    check_tensor(weights, {edge_count, weight_dim}, x.element_type(), "weights");
    check_tensor(receivers, {edge_count}, xla::ffi::DataType::S32, "receivers");
    check_tensor(dout, {node_count, output_dim}, x.element_type(), "dout");
    if (tx != nullptr) check_tensor(*tx, {node_count, input_dim}, x.element_type(), "tx");
    if (tsh != nullptr) check_tensor(*tsh, {edge_count, edge_dim}, x.element_type(), "tsh");
    if (tweights != nullptr) {
        check_tensor(*tweights, {edge_count, weight_dim}, x.element_type(), "tweights");
    }
    if (tdout != nullptr) {
        check_tensor(*tdout, {node_count, output_dim}, x.element_type(), "tdout");
    }
    check_tensor(*tdx, {node_count, input_dim}, x.element_type(), "tdx");
    check_tensor(*tdsh, {edge_count, edge_dim}, x.element_type(), "tdsh");
    check_tensor(*tdweights, {edge_count, weight_dim}, x.element_type(), "tdweights");
    zero_buffer(*tdx, stream);
    zero_buffer(*tdsh, stream);
    zero_buffer(*tdweights, stream);
    auto* jit_kernel = compile_factorized_projected_with_caching(
        source, hash, num_threads, logical_cohort_width, shared_memory_bytes);
    jit_kernel->backward_jvp(
        node_count, edge_count, data_ptr(x), data_ptr(sh), data_ptr(weights), data_ptr(senders),
        data_ptr(receivers), data_ptr(dout), tx == nullptr ? nullptr : data_ptr(*tx),
        tsh == nullptr ? nullptr : data_ptr(*tsh),
        tweights == nullptr ? nullptr : data_ptr(*tweights),
        tdout == nullptr ? nullptr : data_ptr(*tdout), data_ptr(tdx), data_ptr(tdsh),
        data_ptr(tdweights), stream);
    return ffi::Error::Success();
}


struct GeneratedBuffers {
    std::vector<ffi::AnyBuffer> args;
    std::vector<ffi::Result<ffi::AnyBuffer>> rets;
};

ffi::ErrorOr<GeneratedBuffers> decode_generated_buffers(
    ffi::RemainingArgs args, ffi::RemainingRets rets, size_t expected_args,
    size_t expected_rets, std::string_view family, int64_t operation) {
    if (args.size() != expected_args || rets.size() != expected_rets) {
        return ffi::Unexpected(ffi::Error::InvalidArgument(
            std::string(family) + " operation " + std::to_string(operation) +
            " received an unexpected number of buffers"));
    }
    GeneratedBuffers buffers;
    buffers.args.reserve(expected_args);
    buffers.rets.reserve(expected_rets);
    for (size_t index = 0; index < expected_args; ++index) {
        auto value = args.get<ffi::AnyBuffer>(index);
        if (!value) return ffi::Unexpected(value.error());
        buffers.args.push_back(*value);
    }
    for (size_t index = 0; index < expected_rets; ++index) {
        auto value = rets.get<ffi::AnyBuffer>(index);
        if (!value) return ffi::Unexpected(value.error());
        buffers.rets.push_back(*value);
    }
    return buffers;
}

ffi::Error factorized_projected_execute_impl(
    ffi::RemainingArgs args, ffi::RemainingRets rets, stream_t stream,
    std::string_view source,
    int64_t hash, int64_t operation, int64_t channels, int64_t input_dim,
    int64_t edge_dim, int64_t weight_dim, int64_t output_dim,
    int64_t num_threads, int64_t logical_cohort_width,
    int64_t shared_memory_bytes) {
    constexpr std::string_view kFamily = "factorized_projected";
    const auto active_count = [](int64_t mask) {
        size_t count = 0;
        for (; mask != 0; mask >>= 1) count += static_cast<size_t>(mask & 1);
        return count;
    };
    if (operation >= 17 && operation <= 23) {
        const int64_t mask = operation - 16;
        auto buffers = decode_generated_buffers(
            args, rets, 5 + active_count(mask), 1, kFamily, operation);
        if (!buffers) return buffers.error();
        auto& a = buffers->args;
        size_t index = 5;
        const auto next = [&](int64_t bit) -> const ffi::AnyBuffer* {
            return mask & bit ? &a[index++] : nullptr;
        };
        const auto* tx = next(1);
        const auto* tsh = next(2);
        const auto* tweights = next(4);
        return factorized_projected_forward_jvp_impl(
            a[0], a[1], a[2], a[3], a[4], tx, tsh, tweights,
            buffers->rets[0], stream, source, hash, channels,
            input_dim, edge_dim, weight_dim, output_dim, num_threads,
            logical_cohort_width, shared_memory_bytes);
    }
    if (operation >= 33 && operation <= 47) {
        const int64_t mask = operation - 32;
        auto buffers = decode_generated_buffers(
            args, rets, 6 + active_count(mask), 3, kFamily, operation);
        if (!buffers) return buffers.error();
        auto& a = buffers->args;
        size_t index = 6;
        const auto next = [&](int64_t bit) -> const ffi::AnyBuffer* {
            return mask & bit ? &a[index++] : nullptr;
        };
        const auto* tx = next(1);
        const auto* tsh = next(2);
        const auto* tweights = next(4);
        const auto* tdout = next(8);
        return factorized_projected_backward_jvp_impl(
            a[0], a[1], a[2], a[3], a[4], a[5], tx, tsh, tweights, tdout,
            buffers->rets[0], buffers->rets[1], buffers->rets[2], stream,
            source, hash, channels, input_dim, edge_dim,
            weight_dim, output_dim, num_threads, logical_cohort_width,
            shared_memory_bytes);
    }
    size_t expected_args;
    size_t expected_rets;
    switch (operation) {
        case 0:
            expected_args = 5;
            expected_rets = 1;
            break;
        case 2:
            expected_args = 6;
            expected_rets = 3;
            break;
        default:
            return ffi::Error::InvalidArgument("unknown factorized projected operation");
    }
    auto buffers = decode_generated_buffers(
        args, rets, expected_args, expected_rets, kFamily, operation);
    if (!buffers) return buffers.error();
    auto& a = buffers->args;
    auto& r = buffers->rets;
    switch (operation) {
        case 0:
            return factorized_projected_forward_impl(
                a[0], a[1], a[2], a[3], a[4], r[0], stream, source, hash,
                channels, input_dim, edge_dim, weight_dim, output_dim,
                num_threads, logical_cohort_width, shared_memory_bytes);
        case 2:
            return factorized_projected_backward_impl(
                a[0], a[1], a[2], a[3], a[4], a[5], r[0], r[1], r[2], stream, source, hash, channels, input_dim, edge_dim, weight_dim,
                output_dim, num_threads, logical_cohort_width,
                shared_memory_bytes);
    }
    return ffi::Error::Internal("unreachable factorized projected operation");
}


ffi::Error factorized_projected_initialize_impl(
    ffi::RemainingArgs, ffi::RemainingRets, stream_t, std::string_view source,
    int64_t hash, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t num_threads, int64_t logical_cohort_width,
    int64_t shared_memory_bytes) {
    compile_factorized_projected_with_caching(
        source, hash, num_threads, logical_cohort_width, shared_memory_bytes);
    return ffi::Error::Success();
}

#define OEQ_GENERATED_INITIALIZE_CONTEXTS                                             \
    .RemainingArgs()                                                                   \
        .RemainingRets()                                                               \
        .Ctx<ffi::PlatformStream<stream_t>>()

#define OEQ_GENERATED_ATTRIBUTES                                                       \
    .Attr<std::string_view>("source")                                                 \
        .Attr<int64_t>("hash")                                                        \
        .Attr<int64_t>("operation")

#define OEQ_PROJECTED_ATTRIBUTES                                                       \
    OEQ_GENERATED_ATTRIBUTES                                                           \
        .Attr<int64_t>("channels")                                                    \
        .Attr<int64_t>("input_dim")                                                   \
        .Attr<int64_t>("edge_dim")                                                    \
        .Attr<int64_t>("weight_dim")                                                  \
        .Attr<int64_t>("output_dim")                                                  \
        .Attr<int64_t>("num_threads")                                                \
        .Attr<int64_t>("logical_cohort_width")                                       \
        .Attr<int64_t>("shared_memory_bytes")

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factorized_projected_initialize, factorized_projected_initialize_impl,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
        OEQ_GENERATED_INITIALIZE_CONTEXTS OEQ_PROJECTED_ATTRIBUTES);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factorized_projected, factorized_projected_execute_impl,
    ffi::Ffi::Bind()
        .RemainingArgs()
        .RemainingRets()
        .Ctx<ffi::PlatformStream<stream_t>>() OEQ_PROJECTED_ATTRIBUTES,
    {xla::ffi::Traits::kCmdBufferCompatible});

#undef OEQ_PROJECTED_ATTRIBUTES
#undef OEQ_GENERATED_ATTRIBUTES
#undef OEQ_GENERATED_INITIALIZE_CONTEXTS

namespace {

#define OEQ_FFI_HANDLER(NAME, INITIALIZE)                                             \
    {#NAME, nullptr, nullptr, reinterpret_cast<void*>(INITIALIZE),                    \
     reinterpret_cast<void*>(NAME), OEQ_FFI_TRAIT_COMMAND_BUFFER_COMPATIBLE}

const OeqFfiHandler kFfiHandlers[] = {
    OEQ_FFI_HANDLER(tp_forward, tp_initialize),
    OEQ_FFI_HANDLER(tp_backward, tp_initialize),
    OEQ_FFI_HANDLER(tp_double_backward, tp_initialize),
    OEQ_FFI_HANDLER(conv_forward, conv_initialize),
    OEQ_FFI_HANDLER(conv_backward, conv_initialize),
    OEQ_FFI_HANDLER(conv_double_backward, conv_initialize),
    OEQ_FFI_HANDLER(factorized_projected, factorized_projected_initialize),
};

#undef OEQ_FFI_HANDLER

const OeqFfiHandlerTable kFfiHandlerTable = {
    OEQ_FFI_ABI_VERSION,
    sizeof(kFfiHandlers) / sizeof(kFfiHandlers[0]),
    kFfiHandlers,
};

}  // namespace

extern "C" const OeqFfiHandlerTable* oeq_ffi_handler_table() {
    return &kFfiHandlerTable;
}
