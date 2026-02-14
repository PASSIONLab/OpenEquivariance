#pragma once

#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "json11/json11.hpp"

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

#include "tensorproducts.hpp"
#include "convolution.hpp"

using namespace std;
using json = json11::Json;

Dtype enum_to_torch_dtype(int64_t i);

Tensor tensor_to_cpu_contiguous(const Tensor &tensor);
Tensor tensor_contiguous(const Tensor &tensor);
Tensor tensor_empty_like(const Tensor &ref, const std::vector<int64_t> &sizes);
Tensor tensor_zeros_like(const Tensor &ref, const std::vector<int64_t> &sizes);
void tensor_zero_(Tensor &tensor);

//Dtype tensor_dtype(const Tensor &tensor);
bool tensor_is_cuda(const Tensor &tensor);
int64_t tensor_dim(const Tensor &tensor);
int64_t tensor_size(const Tensor &tensor, int64_t dim);
int64_t tensor_numel(const Tensor &tensor);

void alert_not_deterministic(const char *name);
Stream get_current_stream();

const uint8_t *tensor_data_ptr_u8(const Tensor &tensor);
void *data_ptr(const Tensor &tensor);

inline std::string shape_to_string(std::initializer_list<int64_t> shape) {
    std::ostringstream oss;
    oss << "[";
    size_t i = 0;
    for (int64_t dim : shape) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dim;
        ++i;
    }
    oss << "]";
    return oss.str();
}

inline std::string tensor_sizes_str(const Tensor &tensor) {
    std::ostringstream oss;
    oss << "[";
    int64_t dims = tensor_dim(tensor);
    for (int64_t i = 0; i < dims; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << tensor_size(tensor, i);
    }
    oss << "]";
    return oss.str();
}

inline std::vector<int64_t> tensor_sizes_vec(const Tensor &tensor) {
    int64_t dims = tensor_dim(tensor);
    std::vector<int64_t> sizes;
    sizes.reserve(static_cast<size_t>(dims));
    for (int64_t i = 0; i < dims; ++i) {
        sizes.push_back(tensor_size(tensor, i));
    }
    return sizes;
}

inline std::vector<int64_t> make_sizes(std::initializer_list<int64_t> sizes) {
    return std::vector<int64_t>(sizes);
}

inline Dtype enum_to_torch_dtype(int64_t i) {
    switch (i) {
        case 1: return kFloat;
        case 2: return kDouble;
        case 3: return kInt;
        case 4: return kLong;
        case 5: return kByte;
    }
    throw logic_error("Unsupported tensor datatype!");
}

inline void check_tensor(const Tensor &tensor,
                         std::initializer_list<int64_t> expected_shape,
                         Dtype expected_dtype,
                         std::string tensor_name) {
    bool shape_ok = (tensor_dim(tensor) == static_cast<int64_t>(expected_shape.size()));
    if (shape_ok) {
        int64_t i = 0;
        for (int64_t dim : expected_shape) {
            if (tensor_size(tensor, i) != dim) {
                shape_ok = false;
                break;
            }
            ++i;
        }
    }

    TCHECK(shape_ok,
          "Shape mismatch for tensor '", tensor_name,
          "'. Expected: ", shape_to_string(expected_shape),
          ". Got: ", tensor_sizes_str(tensor));
    TCHECK(tensor_is_cuda(tensor), "Tensor '", tensor_name, "' is not on the GPU.");
    /*TCHECK(tensor_dtype(tensor) == expected_dtype,
          "Dtype mismatch for tensor '", tensor_name,
          "'. Expected: ", static_cast<int>(expected_dtype),
          ". Got: ", static_cast<int>(tensor_dtype(tensor)));*/
}

inline std::unordered_map<std::string, int64_t> parse_json_config(const json &j_obj) {
    std::unordered_map<std::string, int64_t> result;
    for (const auto &kv : j_obj.object_items()) {
        result[kv.first] = static_cast<int64_t>(kv.second.number_value());
    }
    return result;
}

struct KernelProp {
    int64_t L1_dim, L2_dim, L3_dim, weight_numel;
    bool shared_weights;
    Dtype irrep_dtype;
    Dtype weight_dtype;

    int64_t workspace_size;     // Convolution only
    bool deterministic;
    Dtype idx_dtype;
    Dtype workspace_dtype;

    KernelProp() :
        L1_dim(0), L2_dim(0), L3_dim(0), weight_numel(0),
        shared_weights(false),
        irrep_dtype(kFloat), weight_dtype(kFloat),
        workspace_size(0), deterministic(false),
        idx_dtype(kInt), workspace_dtype(kByte) {}

    KernelProp(
        std::unordered_map<string, int64_t> &kernel_dims, bool is_convolution) :
            L1_dim(kernel_dims.at("L1_dim")),
            L2_dim(kernel_dims.at("L2_dim")),
            L3_dim(kernel_dims.at("L3_dim")),
            weight_numel(kernel_dims.at("weight_numel")),
            shared_weights(kernel_dims.at("shared_weights")),
            irrep_dtype(enum_to_torch_dtype(kernel_dims.at("irrep_dtype"))),
            weight_dtype(enum_to_torch_dtype(kernel_dims.at("weight_dtype"))),
            workspace_dtype(kByte) {
        if (is_convolution) {
            workspace_size = kernel_dims.at("workspace_size");
            deterministic = kernel_dims.at("deterministic");
            idx_dtype = enum_to_torch_dtype(kernel_dims.at("idx_dtype"));
        }
    }
};

inline std::unordered_map<int64_t,
    std::pair<
        std::unique_ptr<JITTPImpl<JITKernel>>,
        KernelProp
    >> tp_cache;

inline std::unordered_map<int64_t,
    std::pair<
        std::unique_ptr<JITConvImpl<JITKernel>>,
        KernelProp
    >> conv_cache;

inline std::mutex mut;

inline std::pair<JITTPImpl<JITKernel>*, KernelProp>
    compile_tp_with_caching(const Tensor &json_bytes,
                            int64_t hash) {
    {
        const std::lock_guard<std::mutex> lock(mut);
        auto it = tp_cache.find(hash);
        if (it == tp_cache.end()) {
            Tensor cpu_tensor = tensor_to_cpu_contiguous(json_bytes);
            std::string json_payload(
                reinterpret_cast<const char *>(tensor_data_ptr_u8(cpu_tensor)),
                tensor_numel(cpu_tensor)
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

inline std::pair<JITConvImpl<JITKernel>*, KernelProp>
    compile_conv_with_caching(const Tensor &json_bytes,
                              int64_t hash) {
    {
        const std::lock_guard<std::mutex> lock(mut);
        auto it = conv_cache.find(hash);
        if (it == conv_cache.end()) {
            Tensor cpu_tensor = tensor_to_cpu_contiguous(json_bytes);
            std::string json_payload(
                reinterpret_cast<const char *>(tensor_data_ptr_u8(cpu_tensor)),
                tensor_numel(cpu_tensor)
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

inline Tensor jit_tp_forward(
        Tensor json_bytes, int64_t hash,
        Tensor L1_in,
        Tensor L2_in,
        Tensor W,
        int64_t L3_dim) {

    auto [jit_kernel, k] = compile_tp_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t num_batch = tensor_size(L1_in, 0);

    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in");

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");

    Tensor L3_out = tensor_empty_like(L1_in, make_sizes({num_batch, k.L3_dim}));

    Tensor L1_contig = tensor_contiguous(L1_in);
    Tensor L2_contig = tensor_contiguous(L2_in);
    Tensor W_contig = tensor_contiguous(W);

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

inline tuple<Tensor, Tensor, Tensor> jit_tp_backward(
        Tensor json_bytes, int64_t hash,
        Tensor L1_in,
        Tensor L2_in,
        Tensor W,
        Tensor L3_grad) {

    auto [jit_kernel, k] = compile_tp_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t num_batch = tensor_size(L1_in, 0);

    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {num_batch, k.L3_dim}, k.irrep_dtype, "L3_grad");

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");

    Tensor L1_grad = tensor_empty_like(L1_in, tensor_sizes_vec(L1_in));
    Tensor L2_grad = tensor_empty_like(L2_in, tensor_sizes_vec(L2_in));
    Tensor W_grad = tensor_empty_like(W, tensor_sizes_vec(W));

    if (k.shared_weights)
        tensor_zero_(W_grad);

    Tensor L1_in_contig = tensor_contiguous(L1_in);
    Tensor L2_in_contig = tensor_contiguous(L2_in);
    Tensor W_contig = tensor_contiguous(W);
    Tensor L3_grad_contig = tensor_contiguous(L3_grad);

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

inline tuple<Tensor, Tensor, Tensor, Tensor> jit_tp_double_backward(
        Tensor json_bytes, int64_t hash,
        Tensor L1_in,
        Tensor L2_in,
        Tensor W,
        Tensor L3_grad,
        Tensor L1_dgrad,
        Tensor L2_dgrad,
        Tensor W_dgrad) {

    auto [jit_kernel, k] = compile_tp_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t num_batch = tensor_size(L1_in, 0);

    check_tensor(L1_in, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {num_batch, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(L1_dgrad, {num_batch, k.L1_dim}, k.irrep_dtype, "L1_dgrad");
    check_tensor(L2_dgrad, {num_batch, k.L2_dim}, k.irrep_dtype, "L2_dgrad");

    if (k.shared_weights) {
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {k.weight_numel}, k.weight_dtype, "W_dgrad");
    } else {
        check_tensor(W, {num_batch, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {num_batch, k.weight_numel}, k.weight_dtype, "W_dgrad");
    }

    Tensor L1_grad = tensor_empty_like(L1_in, tensor_sizes_vec(L1_in));
    Tensor L2_grad = tensor_empty_like(L2_in, tensor_sizes_vec(L2_in));
    Tensor W_grad = tensor_empty_like(W, tensor_sizes_vec(W));
    Tensor L3_dgrad = tensor_empty_like(L3_grad, tensor_sizes_vec(L3_grad));

    Tensor L1_in_contig = tensor_contiguous(L1_in);
    Tensor L2_in_contig = tensor_contiguous(L2_in);
    Tensor W_contig = tensor_contiguous(W);
    Tensor L3_grad_contig = tensor_contiguous(L3_grad);

    Tensor L1_dgrad_contig = tensor_contiguous(L1_dgrad);
    Tensor L2_dgrad_contig = tensor_contiguous(L2_dgrad);
    Tensor W_dgrad_contig = tensor_contiguous(W_dgrad);

    if (k.shared_weights) {
        tensor_zero_(W_grad);
        TCHECK(tensor_dim(W) == 1);
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

inline Tensor jit_conv_forward(
        Tensor json_bytes, int64_t hash,
        Tensor L1_in,
        Tensor L2_in,
        Tensor W,
        int64_t L3_dim,
        Tensor rows,
        Tensor cols,
        Tensor workspace,
        Tensor transpose_perm) {

    auto [jit_kernel, k] = compile_conv_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t nnz = tensor_size(rows, 0);
    const int64_t node_count = tensor_size(L1_in, 0);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic) {
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    } else {
        alert_not_deterministic("OpenEquivariance_conv_atomic_forward");
    }
    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");

    Tensor L3_out = tensor_zeros_like(L1_in, make_sizes({node_count, k.L3_dim}));

    Tensor L1_contig = tensor_contiguous(L1_in);
    Tensor L2_contig = tensor_contiguous(L2_in);
    Tensor W_contig = tensor_contiguous(W);
    Tensor rows_contig = tensor_contiguous(rows);
    Tensor cols_contig = tensor_contiguous(cols);
    Tensor workspace_contig = tensor_contiguous(workspace);

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

inline tuple<Tensor, Tensor, Tensor> jit_conv_backward(
        Tensor json_bytes, int64_t hash,
        Tensor L1_in,
        Tensor L2_in,
        Tensor W,
        Tensor L3_grad,
        Tensor rows,
        Tensor cols,
        Tensor workspace,
        Tensor transpose_perm) {

    auto [jit_kernel, k] = compile_conv_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t nnz = tensor_size(rows, 0);
    const int64_t node_count = tensor_size(L1_in, 0);

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {node_count, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic) {
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    } else {
         alert_not_deterministic("OpenEquivariance_conv_atomic_backward");
    }

    if (k.shared_weights)
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
    else
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");

    Tensor L1_grad = tensor_zeros_like(L1_in, tensor_sizes_vec(L1_in));
    Tensor L2_grad = tensor_zeros_like(L2_in, tensor_sizes_vec(L2_in));
    Tensor W_grad = tensor_empty_like(W, tensor_sizes_vec(W));

    Tensor L1_in_contig = tensor_contiguous(L1_in);
    Tensor L2_in_contig = tensor_contiguous(L2_in);
    Tensor W_contig = tensor_contiguous(W);
    Tensor L3_grad_contig = tensor_contiguous(L3_grad);

    Tensor rows_contig = tensor_contiguous(rows);
    Tensor cols_contig = tensor_contiguous(cols);
    Tensor workspace_contig = tensor_contiguous(workspace);
    Tensor transpose_perm_contig = tensor_contiguous(transpose_perm);

    if (k.shared_weights)
        tensor_zero_(W_grad);

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

inline tuple<Tensor, Tensor, Tensor, Tensor> jit_conv_double_backward(
        Tensor json_bytes, int64_t hash,
        Tensor L1_in,
        Tensor L2_in,
        Tensor W,
        Tensor L3_grad,
        Tensor L1_dgrad,
        Tensor L2_dgrad,
        Tensor W_dgrad,
        Tensor rows,
        Tensor cols,
        Tensor workspace,
        Tensor transpose_perm) {

    auto [jit_kernel, k] = compile_conv_with_caching(json_bytes, hash);
    Stream stream = get_current_stream();

    const int64_t nnz = tensor_size(rows, 0);
    const int64_t node_count = tensor_size(L1_in, 0);

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
    } else {
        alert_not_deterministic("OpenEquivariance_conv_atomic_double_backward");
    }

    if (k.shared_weights) {
        check_tensor(W, {k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {k.weight_numel}, k.weight_dtype, "W_dgrad");
    } else {
        check_tensor(W, {nnz, k.weight_numel}, k.weight_dtype, "W");
        check_tensor(W_dgrad, {nnz, k.weight_numel}, k.weight_dtype, "W_dgrad");
    }

    Tensor L1_grad = tensor_zeros_like(L1_in, tensor_sizes_vec(L1_in));
    Tensor L2_grad = tensor_zeros_like(L2_in, tensor_sizes_vec(L2_in));
    Tensor W_grad = tensor_empty_like(W, tensor_sizes_vec(W));
    Tensor L3_dgrad = tensor_zeros_like(L3_grad, tensor_sizes_vec(L3_grad));

    Tensor L1_in_contig = tensor_contiguous(L1_in);
    Tensor L2_in_contig = tensor_contiguous(L2_in);
    Tensor W_contig = tensor_contiguous(W);
    Tensor L3_grad_contig = tensor_contiguous(L3_grad);
    Tensor L1_dgrad_contig = tensor_contiguous(L1_dgrad);
    Tensor L2_dgrad_contig = tensor_contiguous(L2_dgrad);
    Tensor W_dgrad_contig = tensor_contiguous(W_dgrad);

    Tensor rows_contig = tensor_contiguous(rows);
    Tensor cols_contig = tensor_contiguous(cols);
    Tensor workspace_contig = tensor_contiguous(workspace);
    Tensor transpose_perm_contig = tensor_contiguous(transpose_perm);

    if (k.shared_weights)
        tensor_zero_(W_grad);

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

REGISTER_LIBRARY_IMPL(libtorch_tp_jit, CUDA, m) {
    m.impl("jit_tp_forward", BOX(&jit_tp_forward));
    m.impl("jit_tp_backward", BOX(&jit_tp_backward));
    m.impl("jit_tp_double_backward", BOX(&jit_tp_double_backward));

    m.impl("jit_conv_forward", BOX(&jit_conv_forward));
    m.impl("jit_conv_backward", BOX(&jit_conv_backward));
    m.impl("jit_conv_double_backward", BOX(&jit_conv_double_backward));
};

REGISTER_LIBRARY(libtorch_tp_jit, m) {
    m.def("jit_tp_forward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, int L3_dim) -> Tensor");
    m.def("jit_tp_backward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad) -> (Tensor, Tensor, Tensor)");
    m.def("jit_tp_double_backward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor L1_dgrad, Tensor L2_dgrad, Tensor W_dgrad) -> (Tensor, Tensor, Tensor, Tensor)");

    m.def("jit_conv_forward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, int L3_dim, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> Tensor");
    m.def("jit_conv_backward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> (Tensor, Tensor, Tensor)");
    m.def("jit_conv_double_backward(Tensor json_bytes, int hash, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor L1_dgrad, Tensor L2_dgrad, Tensor W_dgrad, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> (Tensor, Tensor, Tensor, Tensor)");
};
