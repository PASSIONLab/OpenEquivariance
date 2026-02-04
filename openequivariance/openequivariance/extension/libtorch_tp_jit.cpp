#include <iostream>
#include <unordered_map>
#include <initializer_list>
#include <string>
#include <stdexcept>
#include <mutex>
#include <memory>

#include "json11/json11.hpp"

#ifdef CUDA_BACKEND
    #include <ATen/cuda/CUDAContext.h>
    #include "backend_cuda.hpp"
    #include "group_mm_cuda.hpp"
    using JITKernel = CUJITKernel;
    using GPU_Allocator = CUDA_Allocator;

    template<typename T>
    using GroupMM = GroupMMCUDA<T>; 

    inline Stream get_current_stream() {
        return c10::cuda::getCurrentCUDAStream(); 
    }
#endif

#ifdef HIP_BACKEND
    #include <c10/hip/HIPStream.h>
    #include "backend_hip.hpp"
    #include "group_mm_hip.hpp"
    using JITKernel = HIPJITKernel;
    using GPU_Allocator = HIP_Allocator;

    template<typename T>
    using GroupMM = GroupMMHIP<T>;

    inline Stream get_current_stream() { 
        return c10::hip::getCurrentHIPStream();  
    }
#endif

#include "tensorproducts.hpp"
#include "convolution.hpp"

using namespace std;
using json = json11::Json;

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

torch::Dtype enum_to_torch_dtype(int64_t i){
    switch(i) {
        case 1: return torch::kFloat; 
        case 2: return torch::kDouble;
        case 3: return torch::kInt;
        case 4: return torch::kLong;
        case 5: return torch::kUInt8;
    }
    throw logic_error("Unsupported tensor datatype!");
} 

inline void check_tensor(const torch::Tensor &tensor, 
                              std::initializer_list<int64_t> expected_shape,
                              torch::Dtype expected_dtype,  
                              std::string tensor_name) {
    TORCH_CHECK(tensor.sizes() == expected_shape, 
                "Shape mismatch for tensor '", tensor_name, 
                "'. Expected: ", torch::IntArrayRef(expected_shape), 
                ". Got: ", tensor.sizes());
    TORCH_CHECK(tensor.device().is_cuda(), "Tensor '", tensor_name, "' is not on the GPU.");
    TORCH_CHECK(tensor.dtype() == expected_dtype, "Dtype mismatch for tensor '", tensor_name, "'. Expected: ", expected_dtype, ". Got: ", tensor.dtype());
}

inline void* data_ptr(const torch::Tensor &tensor) {
    if(tensor.dtype() == torch::kFloat)
        return reinterpret_cast<void*>(tensor.data_ptr<float>());
    else if(tensor.dtype() == torch::kDouble)
        return reinterpret_cast<void*>(tensor.data_ptr<double>());
    else if(tensor.dtype() == torch::kLong) 
        return reinterpret_cast<void*>(tensor.data_ptr<int64_t>());
    else if(tensor.dtype() == torch::kByte) 
        return reinterpret_cast<void*>(tensor.data_ptr<uint8_t>()); // Replaces kUInt8
    else if(tensor.dtype() == torch::kInt)
        return reinterpret_cast<void*>(tensor.data_ptr<int32_t>());
    else
        throw logic_error("Unsupported tensor datatype!");
}

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
    torch::Dtype irrep_dtype;
    torch::Dtype weight_dtype;

    int64_t workspace_size;     // Convolution only
    bool deterministic;
    torch::Dtype idx_dtype;
    torch::Dtype workspace_dtype;

    KernelProp() : 
        L1_dim(0), L2_dim(0), L3_dim(0), weight_numel(0), 
        shared_weights(false), 
        irrep_dtype(torch::kFloat), weight_dtype(torch::kFloat),
        workspace_size(0), deterministic(false), 
        idx_dtype(torch::kInt), workspace_dtype(torch::kByte) {}

    KernelProp(
        std::unordered_map<string, int64_t> &kernel_dims, bool is_convolution):
            L1_dim(kernel_dims.at("L1_dim")),
            L2_dim(kernel_dims.at("L2_dim")),    
            L3_dim(kernel_dims.at("L3_dim")),
            weight_numel(kernel_dims.at("weight_numel")),
            shared_weights(kernel_dims.at("shared_weights")),
            irrep_dtype(enum_to_torch_dtype(kernel_dims.at("irrep_dtype"))),
            weight_dtype(enum_to_torch_dtype(kernel_dims.at("weight_dtype"))),
            workspace_dtype(torch::kByte) { 
        if(is_convolution) {
            workspace_size = kernel_dims.at("workspace_size");
            deterministic = kernel_dims.at("deterministic");
            idx_dtype = enum_to_torch_dtype(kernel_dims.at("idx_dtype"));
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

std::mutex mut;

std::pair<JITTPImpl<JITKernel>*, KernelProp> 
    compile_tp_with_caching(const torch::Tensor &json_bytes,
                            int64_t hash) {
    {
        const std::lock_guard<std::mutex> lock(mut);
        auto it = tp_cache.find(hash); 
        if (it == tp_cache.end()) {
            torch::Tensor cpu_tensor = json_bytes.to(torch::kCPU).contiguous();
            std::string json_payload(
                reinterpret_cast<const char*>(cpu_tensor.data_ptr<uint8_t>()), 
                cpu_tensor.numel()
            );

            std::string err;
            json root = json::parse(json_payload, err);
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
            
            tp_cache.insert({hash,
                std::make_pair(std::move(jit_tp_impl), 
                KernelProp(kernel_prop_map, false))});
            it = tp_cache.find(hash);
        }
        return {it->second.first.get(), it->second.second};
    }
}

std::pair<JITConvImpl<JITKernel>*, KernelProp> 
    compile_conv_with_caching(const torch::Tensor &json_bytes,
                              int64_t hash) {
    {
        const std::lock_guard<std::mutex> lock(mut);
        auto it = conv_cache.find(hash); 
        if (it == conv_cache.end()) {
            torch::Tensor cpu_tensor = json_bytes.to(torch::kCPU).contiguous();
            std::string json_payload(
                reinterpret_cast<const char*>(cpu_tensor.data_ptr<uint8_t>()), 
                cpu_tensor.numel()
            );

            std::string err;
            json root = json::parse(json_payload, err);
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
            
            conv_cache.insert({hash,
                std::make_pair(std::move(jit_conv_impl), 
                KernelProp(kernel_prop_map, true))});
            it = conv_cache.find(hash);
        }
        return {it->second.first.get(), it->second.second};
    }
}

// --------------------- Tensor Products --------------------------

torch::Tensor jit_tp_forward(
        torch::Tensor json_bytes, int64_t hash,
        torch::Tensor L1_in,
        torch::Tensor L2_in,
        torch::Tensor W) {
    
    auto [jit_kernel, k] = compile_tp_with_caching(json_bytes, hash);
    Stream stream = get_current_stream(); 

    const int64_t num_batch = L1_in.size(0);

    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in"); 

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else 
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");

    torch::Tensor L3_out = torch::empty({num_batch, k.L3_dim}, L1_in.options());
        
    at::Tensor L1_contig = L1_in.contiguous();
    at::Tensor L2_contig = L2_in.contiguous();
    at::Tensor W_contig = W.contiguous();
    
    jit_kernel->exec_tensor_product(
            num_batch,
            data_ptr(L1_contig), 
            data_ptr(L2_contig), 
            data_ptr(L3_out),
            data_ptr(W_contig),
            stream
        );

    return L3_out;
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> jit_tp_backward(
        torch::Tensor json_bytes, int64_t hash,
        torch::Tensor L1_in,
        torch::Tensor L2_in,
        torch::Tensor W, 
        torch::Tensor L3_grad) {

    auto [jit_kernel, k] = compile_tp_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t num_batch = L1_in.size(0);

    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {num_batch, k.L3_dim}, k.irrep_dtype, "L3_grad");

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");

    torch::Tensor L1_grad = torch::empty(L1_in.sizes(), L1_in.options());
    torch::Tensor L2_grad = torch::empty(L2_in.sizes(), L2_in.options());
    torch::Tensor W_grad = torch::empty(W.sizes(), W.options());

    if(k.shared_weights)
        W_grad.zero_();

    torch::Tensor L1_in_contig = L1_in.contiguous();
    torch::Tensor L2_in_contig = L2_in.contiguous();
    torch::Tensor W_contig = W.contiguous();
    torch::Tensor L3_grad_contig = L3_grad.contiguous();

    jit_kernel->backward(
            num_batch, 
            data_ptr(L1_in_contig), data_ptr(L1_grad),
            data_ptr(L2_in_contig), data_ptr(L2_grad),
            data_ptr(W_contig), data_ptr(W_grad),
            data_ptr(L3_grad_contig),
            stream
    );

    return tuple(L1_grad, L2_grad, W_grad);
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> jit_tp_double_backward(
        torch::Tensor json_bytes, int64_t hash,
        torch::Tensor L1_in, 
        torch::Tensor L2_in, 
        torch::Tensor W, 
        torch::Tensor L3_grad, 
        torch::Tensor L1_dgrad, 
        torch::Tensor L2_dgrad, 
        torch::Tensor W_dgrad) {
    
    auto [jit_kernel, k] = compile_tp_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t num_batch = L1_in.size(0);

    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {num_batch, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(L1_dgrad, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_dgrad");
    check_tensor(L2_dgrad, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_dgrad");

    if (k.shared_weights){
        check_tensor(W, {k.weight_numel}, k.weight_dtype,  "W");
        check_tensor(W_dgrad, {k.weight_numel}, k.weight_dtype, "W_dgrad");
    } else {
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {num_batch, k.weight_numel}, k.weight_dtype, "W_dgrad");
    }

    torch::Tensor L1_grad = torch::empty(L1_in.sizes(), L1_in.options());
    torch::Tensor L2_grad = torch::empty(L2_in.sizes(), L2_in.options());
    torch::Tensor W_grad = torch::empty(W.sizes(), W.options());
    torch::Tensor L3_dgrad = torch::empty(L3_grad.sizes(), L3_grad.options());

    torch::Tensor L1_in_contig = L1_in.contiguous();
    torch::Tensor L2_in_contig = L2_in.contiguous();
    torch::Tensor W_contig = W.contiguous();
    torch::Tensor L3_grad_contig = L3_grad.contiguous();

    torch::Tensor L1_dgrad_contig = L1_dgrad.contiguous();
    torch::Tensor L2_dgrad_contig = L2_dgrad.contiguous();
    torch::Tensor W_dgrad_contig = W_dgrad.contiguous();

    if(k.shared_weights) {
        W_grad.zero_();
        TORCH_CHECK(W.dim() == 1);
    } 

    jit_kernel->double_backward(
            num_batch,
            data_ptr(L1_in_contig), data_ptr(L2_in_contig),
            data_ptr(W_contig), data_ptr(L3_grad_contig),
            data_ptr(L1_dgrad_contig), data_ptr(L2_dgrad_contig),
            data_ptr(W_dgrad_contig),
            data_ptr(L1_grad), data_ptr(L2_grad),
            data_ptr(W_grad), data_ptr(L3_dgrad),
            stream
    );

    return tuple(L1_grad, L2_grad, W_grad, L3_dgrad); 
}


// ========================= Convolution ================================== 

torch::Tensor jit_conv_forward(
        torch::Tensor json_bytes, int64_t hash,
        torch::Tensor L1_in,
        torch::Tensor L2_in,
        torch::Tensor W,
        torch::Tensor rows,
        torch::Tensor cols,
        torch::Tensor workspace,
        torch::Tensor transpose_perm) {

    auto [jit_kernel, k] = compile_conv_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t nnz = rows.size(0);
    const int64_t node_count = L1_in.size(0);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic){
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    } else {
        at::globalContext().alertNotDeterministic("OpenEquivariance_conv_atomic_forward");
    }
    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");

    torch::Tensor L3_out = torch::zeros({node_count, k.L3_dim}, L1_in.options());
    
    torch::Tensor L1_contig = L1_in.contiguous();
    torch::Tensor L2_contig = L2_in.contiguous();
    torch::Tensor W_contig = W.contiguous();
    torch::Tensor rows_contig = rows.contiguous();
    torch::Tensor cols_contig = cols.contiguous();
    torch::Tensor workspace_contig = workspace.contiguous();

    jit_kernel->exec_conv(
            data_ptr(L1_contig), 
            data_ptr(L2_contig), 
            data_ptr(W_contig), 
            data_ptr(L3_out),
            data_ptr(rows_contig), 
            data_ptr(cols_contig),
            nnz, node_count,
            data_ptr(workspace_contig),
            stream);

    return L3_out;
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> jit_conv_backward(
        torch::Tensor json_bytes, int64_t hash,
        torch::Tensor L1_in,
        torch::Tensor L2_in,
        torch::Tensor W,
        torch::Tensor L3_grad,
        torch::Tensor rows,
        torch::Tensor cols,
        torch::Tensor workspace,
        torch::Tensor transpose_perm) {
    
    auto [jit_kernel, k] = compile_conv_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t nnz = rows.size(0);
    const int64_t node_count = L1_in.size(0);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {node_count, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic){
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    } else {
         at::globalContext().alertNotDeterministic("OpenEquivariance_conv_atomic_backward");
    }
    
    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");

    torch::Tensor L1_grad = torch::zeros(L1_in.sizes(), L1_in.options());
    torch::Tensor L2_grad = torch::zeros(L2_in.sizes(), L2_in.options());
    torch::Tensor W_grad = torch::empty(W.sizes(), W.options());
    
    torch::Tensor L1_in_contig = L1_in.contiguous();
    torch::Tensor L2_in_contig = L2_in.contiguous();
    torch::Tensor W_contig = W.contiguous();
    torch::Tensor L3_grad_contig = L3_grad.contiguous();

    torch::Tensor rows_contig = rows.contiguous();
    torch::Tensor cols_contig = cols.contiguous();
    torch::Tensor workspace_contig = workspace.contiguous();
    torch::Tensor transpose_perm_contig = transpose_perm.contiguous();

    if(k.shared_weights)
        W_grad.zero_();

    jit_kernel->backward(
            data_ptr(L1_in_contig), data_ptr(L1_grad),
            data_ptr(L2_in_contig), data_ptr(L2_grad),
            data_ptr(W_contig), data_ptr(W_grad),
            data_ptr(L3_grad_contig),
            data_ptr(rows_contig), data_ptr(cols_contig),
            nnz, node_count,
            data_ptr(workspace_contig),
            data_ptr(transpose_perm_contig),
            stream);

    return tuple(L1_grad, L2_grad, W_grad);
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> jit_conv_double_backward(
        torch::Tensor json_bytes, int64_t hash,
        torch::Tensor L1_in, 
        torch::Tensor L2_in, 
        torch::Tensor W, 
        torch::Tensor L3_grad, 
        torch::Tensor L1_dgrad, 
        torch::Tensor L2_dgrad, 
        torch::Tensor W_dgrad, 
        torch::Tensor rows,
        torch::Tensor cols,
        torch::Tensor workspace,
        torch::Tensor transpose_perm) {
    
    auto [jit_kernel, k] = compile_conv_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t nnz = rows.size(0);
    const int64_t node_count = L1_in.size(0);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in"); 
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in"); 
    check_tensor(L3_grad, {node_count, k.L3_dim}, k.irrep_dtype, "L3_grad"); 
    check_tensor(L1_dgrad, {node_count, k.L1_dim}, k.irrep_dtype, "L1_dgrad"); 
    check_tensor(L2_dgrad, {nnz, k.L2_dim}, k.irrep_dtype, "L2_dgrad");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows"); 
    check_tensor(cols, {nnz},  k.idx_dtype, "cols"); 

    if (k.deterministic) {
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    } else {
        at::globalContext().alertNotDeterministic("OpenEquivariance_conv_atomic_double_backward");
    }

    if (k.shared_weights) {
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {k.weight_numel}, k.weight_dtype, "W_dgrad");
    }
    else {
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {nnz, k.weight_numel}, k.weight_dtype, "W_dgrad"); 
    }

    torch::Tensor L1_grad = torch::zeros(L1_in.sizes(), L1_in.options());
    torch::Tensor L2_grad = torch::zeros(L2_in.sizes(), L2_in.options());
    torch::Tensor W_grad = torch::empty(W.sizes(), W.options());
    torch::Tensor L3_dgrad = torch::zeros(L3_grad.sizes(), L3_grad.options());

    torch::Tensor L1_in_contig = L1_in.contiguous();
    torch::Tensor L2_in_contig = L2_in.contiguous();
    torch::Tensor W_contig = W.contiguous();
    torch::Tensor L3_grad_contig = L3_grad.contiguous();
    torch::Tensor L1_dgrad_contig = L1_dgrad.contiguous();
    torch::Tensor L2_dgrad_contig = L2_dgrad.contiguous();
    torch::Tensor W_dgrad_contig = W_dgrad.contiguous();

    torch::Tensor rows_contig = rows.contiguous();
    torch::Tensor cols_contig = cols.contiguous();
    torch::Tensor workspace_contig = workspace.contiguous();
    torch::Tensor transpose_perm_contig = transpose_perm.contiguous();

    if(k.shared_weights)
        W_grad.zero_();

    jit_kernel->double_backward(
            data_ptr(L1_in_contig), data_ptr(L2_in_contig),
            data_ptr(W_contig), data_ptr(L3_grad_contig),
            data_ptr(L1_dgrad_contig), data_ptr(L2_dgrad_contig),
            data_ptr(W_dgrad_contig),
            data_ptr(L1_grad), data_ptr(L2_grad),
            data_ptr(W_grad), data_ptr(L3_dgrad),
            data_ptr(rows_contig), data_ptr(cols_contig),
            nnz, node_count,
            data_ptr(workspace_contig), data_ptr(transpose_perm_contig),
            stream
    );

    return tuple(L1_grad, L2_grad, W_grad, L3_dgrad);
}

// =========================================================== 

TORCH_LIBRARY_IMPL(libtorch_tp_jit, CUDA, m) { 
    m.impl("jit_tp_forward", &jit_tp_forward);
    m.impl("jit_tp_backward", &jit_tp_backward);
    m.impl("jit_tp_double_backward", &jit_tp_double_backward);

    m.impl("jit_conv_forward", &jit_conv_forward);
    m.impl("jit_conv_backward", &jit_conv_backward);
    m.impl("jit_conv_double_backward", &jit_conv_double_backward);
};

TORCH_LIBRARY(libtorch_tp_jit, m) {
    m.def("jit_tp_forward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W) -> Tensor");
    m.def("jit_tp_backward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad) -> (Tensor, Tensor, Tensor)");
    m.def("jit_tp_double_backward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor L1_dgrad, Tensor L2_dgrad, Tensor W_dgrad) -> (Tensor, Tensor, Tensor, Tensor)");

    m.def("jit_conv_forward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> Tensor");
    m.def("jit_conv_backward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> (Tensor, Tensor, Tensor)");
    m.def("jit_conv_double_backward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor L1_dgrad, Tensor L2_dgrad, Tensor W_dgrad, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> (Tensor, Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(libtorch_tp_jit, m) {}