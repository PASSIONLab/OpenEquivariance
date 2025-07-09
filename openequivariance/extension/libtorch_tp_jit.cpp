#include <iostream>
#include <unordered_map>
#include <initializer_list>
#include <string>
#include <stdexcept>

#include <pybind11/pybind11.h> 
#include <pybind11/numpy.h>

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

#include "buffer.hpp"
#include "tensorproducts.hpp"
#include "convolution.hpp"

using namespace std;
namespace py=pybind11;

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

using Map_t=torch::Dict<string, int64_t>;

std::unordered_map<string, int64_t> to_map(const Map_t &map) {
    std::unordered_map<string, int64_t> result;
    for(auto it = map.begin(); it != map.end(); ++it) {
        result[it->key()] = it->value();
    }
    return result;
} 

torch::Dtype enum_to_torch_dtype(int64_t i){
    switch(i) {
        case 1:
            return torch::kFloat; 
        case 2: 
            return torch::kDouble;
        case 3: 
            return torch::kInt;
        case 4: 
            return torch::kLong;
        case 5: 
            return torch::kUInt8;
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
        return reinterpret_cast<void*>(tensor.data_ptr<uint8_t>());
    else
        throw logic_error("Unsupported tensor datatype!");
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

    KernelProp(Map_t &kernel_dims, bool is_convolution):
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

class __attribute__ ((visibility ("default"))) TorchJITProduct : public torch::CustomClassHolder {
public:
    Map_t fwd_dict, bwd_dict, dbl_bwd_dict, kernel_dims;
    JITTPImpl<JITKernel> internal;
    KernelProp kernelProp;
    int64_t L3_dim; 

    TorchJITProduct(string kernel_plaintext, Map_t fwd_dict_i, Map_t bwd_dict_i, Map_t dbl_bwd_dict_i, Map_t kernel_dims_i) :
        fwd_dict(fwd_dict_i.copy()),
        bwd_dict(bwd_dict_i.copy()),
        dbl_bwd_dict(dbl_bwd_dict_i.copy()),
        kernel_dims(kernel_dims_i.copy()),
        internal(kernel_plaintext, 
                to_map(fwd_dict_i),
                to_map(bwd_dict_i),
                to_map(dbl_bwd_dict_i),
                to_map(kernel_dims_i)
            ),
        kernelProp(kernel_dims, false),
        L3_dim(kernelProp.L3_dim) 
        { }

    tuple<  tuple<string, string>, 
            tuple<string, Map_t>, 
            tuple<string, Map_t>, 
            tuple<string, Map_t>, 
            tuple<string, Map_t>> __obj_flatten__() {
        return tuple(tuple("kernel_plaintext", internal.jit.kernel_plaintext),
            tuple("fwd_config", fwd_dict),
            tuple("bwd_config", bwd_dict),
            tuple("dbl_bwd_config", dbl_bwd_dict),
            tuple("kernel_dims", kernel_dims));
    }

    void exec_tensor_product_device_rawptrs(int64_t num_batch, int64_t L1_in, int64_t L2_in, int64_t L3_out, int64_t weights) { 
        Stream stream = get_current_stream(); 
        internal.exec_tensor_product(
                num_batch,
                reinterpret_cast<void*>(L1_in), 
                reinterpret_cast<void*>(L2_in), 
                reinterpret_cast<void*>(L3_out), 
                reinterpret_cast<void*>(weights),
                stream
            ); 
    } 

    void backward_device_rawptrs(int64_t num_batch,
            int64_t L1_in, int64_t L1_grad,
            int64_t L2_in, int64_t L2_grad, 
            int64_t weight, int64_t weight_grad,
            int64_t L3_grad) {
        Stream stream = get_current_stream(); 
        internal.backward(num_batch,
            reinterpret_cast<void*>(L1_in), reinterpret_cast<void*>(L1_grad),
            reinterpret_cast<void*>(L2_in), reinterpret_cast<void*>(L2_grad),
            reinterpret_cast<void*>(weight), reinterpret_cast<void*>(weight_grad),
            reinterpret_cast<void*>(L3_grad), stream
        );
    }
};

torch::Tensor jit_tp_forward(
        const c10::intrusive_ptr<TorchJITProduct> &jit_instance,
        const torch::Tensor &L1_in,
        const torch::Tensor &L2_in,
        const torch::Tensor &W) {
    
    Stream stream = get_current_stream(); 

    const int64_t num_batch = L1_in.size(0);
    const KernelProp &k = jit_instance->kernelProp;

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
    
    jit_instance->internal.exec_tensor_product(
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
        const c10::intrusive_ptr<TorchJITProduct> &jit_instance,
        const torch::Tensor &L1_in,
        const torch::Tensor &L2_in,
        const torch::Tensor &W, 
        const torch::Tensor &L3_grad) {

    Stream stream = get_current_stream();

    const int64_t num_batch = L1_in.size(0);
    const KernelProp &k = jit_instance->kernelProp;

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

    jit_instance->internal.backward(
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
        const c10::intrusive_ptr<TorchJITProduct> &jit_instance,
        const torch::Tensor &L1_in, 
        const torch::Tensor &L2_in, 
        const torch::Tensor &W, 
        const torch::Tensor &L3_grad, 
        const torch::Tensor &L1_dgrad, 
        const torch::Tensor &L2_dgrad, 
        const torch::Tensor &W_dgrad) {
    
    Stream stream = get_current_stream();

    const int64_t num_batch = L1_in.size(0);
    const KernelProp &k = jit_instance->kernelProp;

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

    jit_instance->internal.double_backward(
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


// =========================================================== 

class TorchJITConv : public torch::CustomClassHolder {
public:
    Map_t fwd_dict, bwd_dict, dbl_bwd_dict, kernel_dims;
    JITConvImpl<JITKernel> internal;
    KernelProp kernelProp;
    int64_t L3_dim;

    TorchJITConv(string kernel_plaintext, Map_t fwd_dict_i, Map_t bwd_dict_i, Map_t dbl_bwd_dict_i, Map_t kernel_dims_i) :
        fwd_dict(fwd_dict_i.copy()),
        bwd_dict(bwd_dict_i.copy()),
        dbl_bwd_dict(bwd_dict_i.copy()),
        kernel_dims(kernel_dims_i.copy()),
        internal(kernel_plaintext,
                to_map(fwd_dict_i),
                to_map(bwd_dict_i),
                to_map(dbl_bwd_dict_i),
                to_map(kernel_dims_i)
            ),
        kernelProp(kernel_dims, true),
        L3_dim(kernelProp.L3_dim)
        { }

    tuple<tuple<string, string>, 
        tuple<string, Map_t>, 
        tuple<string, Map_t>, 
        tuple<string, Map_t>, 
        tuple<string, Map_t>> __obj_flatten__() {
        return tuple(tuple("kernel_plaintext", internal.jit.kernel_plaintext),
            tuple("fwd_config", fwd_dict),
            tuple("bwd_config", bwd_dict),
            tuple("dbl_bwd_config", dbl_bwd_dict),
            tuple("kernel_dims", kernel_dims));
    }

    void exec_conv_rawptrs(
            int64_t L1_in, int64_t L2_in, int64_t weights, int64_t L3_out,
            int64_t rows, int64_t cols,
            int64_t nnz, int64_t node_count,
            int64_t workspace) {
        Stream stream = get_current_stream();
        internal.exec_conv(
            reinterpret_cast<void*>(L1_in), 
            reinterpret_cast<void*>(L2_in), 
            reinterpret_cast<void*>(weights), 
            reinterpret_cast<void*>(L3_out),
            reinterpret_cast<void*>(rows), 
            reinterpret_cast<void*>(cols),
            nnz, node_count,
            reinterpret_cast<void*>(workspace),
            stream);
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
        Stream stream = get_current_stream();
        internal.backward(
            reinterpret_cast<void*>(L1_in), reinterpret_cast<void*>(L1_grad),
            reinterpret_cast<void*>(L2_in), reinterpret_cast<void*>(L2_grad),
            reinterpret_cast<void*>(weight), reinterpret_cast<void*>(weight_grad),
            reinterpret_cast<void*>(L3_grad),
            reinterpret_cast<void*>(rows), 
            reinterpret_cast<void*>(cols),
            nnz, node_count,
            reinterpret_cast<void*>(workspace),
            reinterpret_cast<void*>(transpose_perm),
            stream);
    }
    void double_backward_rawptrs(
            int64_t L1_in, int64_t L2_in, int64_t W, int64_t L3_grad,
            int64_t L1_dgrad, int64_t L2_dgrad, int64_t w_dgrad,
            int64_t L1_grad, int64_t L2_grad, int64_t W_grad, int64_t L3_dgrad,
            int64_t rows, int64_t cols,
            int64_t nnz, int64_t node_count,
            int64_t wspace, int64_t transpose_perm) {
       
        Stream stream = get_current_stream();
        internal.double_backward(
            reinterpret_cast<void*>(L1_in), 
            reinterpret_cast<void*>(L2_in), 
            reinterpret_cast<void*>(W), 
            reinterpret_cast<void*>(L3_grad),
            reinterpret_cast<void*>(L1_dgrad), 
            reinterpret_cast<void*>(L2_dgrad), 
            reinterpret_cast<void*>(w_dgrad),
            reinterpret_cast<void*>(L1_grad), 
            reinterpret_cast<void*>(L2_grad),
            reinterpret_cast<void*>(W_grad), 
            reinterpret_cast<void*>(L3_dgrad),
            reinterpret_cast<void*>(rows), 
            reinterpret_cast<void*>(cols),
            nnz, node_count,
            reinterpret_cast<void*>(wspace),
            reinterpret_cast<void*>(transpose_perm),
            stream);
    }
};

torch::Tensor jit_conv_forward(
        const c10::intrusive_ptr<TorchJITConv> &jit_instance,
        const torch::Tensor &L1_in,
        const torch::Tensor &L2_in,
        const torch::Tensor &W,
        const torch::Tensor &rows,
        const torch::Tensor &cols,
        const torch::Tensor &workspace,
        const torch::Tensor &transpose_perm) {

    Stream stream = get_current_stream();

    const int64_t nnz = rows.size(0);
    const int64_t node_count = L1_in.size(0);
    const KernelProp &k = jit_instance->kernelProp;

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic)
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");

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

    jit_instance->internal.exec_conv(
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
        const c10::intrusive_ptr<TorchJITConv> &jit_instance,
        const torch::Tensor &L1_in,
        const torch::Tensor &L2_in,
        const torch::Tensor &W,
        const torch::Tensor &L3_grad,
        const torch::Tensor &rows,
        const torch::Tensor &cols,
        const torch::Tensor &workspace,
        const torch::Tensor &transpose_perm) {
    
    Stream stream = get_current_stream();

    const int64_t nnz = rows.size(0);
    const int64_t node_count = L1_in.size(0);
    const KernelProp &k = jit_instance->kernelProp;

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in");
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in");
    check_tensor(L3_grad, {node_count, k.L3_dim}, k.irrep_dtype, "L3_grad");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows");
    check_tensor(cols, {nnz}, k.idx_dtype, "cols");

    if (k.deterministic) 
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");
    
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

    jit_instance->internal.backward(
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
        const c10::intrusive_ptr<TorchJITConv> &jit_instance,
        const torch::Tensor &L1_in, 
        const torch::Tensor &L2_in, 
        const torch::Tensor &W, 
        const torch::Tensor &L3_grad, 
        const torch::Tensor &L1_dgrad, 
        const torch::Tensor &L2_dgrad, 
        const torch::Tensor &W_dgrad, 
        const torch::Tensor &rows,
        const torch::Tensor &cols,
        const torch::Tensor &workspace,
        const torch::Tensor &transpose_perm) {
    
    Stream stream = get_current_stream();

    const int64_t nnz = rows.size(0);
    const int64_t node_count = L1_in.size(0);
    const KernelProp &k = jit_instance->kernelProp;

    check_tensor(L1_in, {node_count, k.L1_dim}, k.irrep_dtype, "L1_in"); 
    check_tensor(L2_in, {nnz, k.L2_dim}, k.irrep_dtype, "L2_in"); 
    check_tensor(L3_grad, {node_count, k.L3_dim}, k.irrep_dtype, "L3_grad"); 
    check_tensor(L1_dgrad, {node_count, k.L1_dim}, k.irrep_dtype, "L1_dgrad"); 
    check_tensor(L2_dgrad, {nnz, k.L2_dim}, k.irrep_dtype, "L2_dgrad");
    check_tensor(workspace, {k.workspace_size}, k.workspace_dtype, "workspace");
    check_tensor(rows, {nnz}, k.idx_dtype, "rows"); 
    check_tensor(cols, {nnz},  k.idx_dtype, "cols"); 

    if (k.deterministic)
        check_tensor(transpose_perm, {nnz}, k.idx_dtype, "transpose perm");

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

    jit_instance->internal.double_backward(
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

TORCH_LIBRARY_FRAGMENT(libtorch_tp_jit, m) { 
    m.class_<TorchJITProduct>("TorchJITProduct")
        .def(torch::init<string, Map_t, Map_t, Map_t, Map_t>())
        .def("__obj_flatten__", &TorchJITProduct::__obj_flatten__)
        .def("exec_tensor_product_rawptr", &TorchJITProduct::exec_tensor_product_device_rawptrs)
        .def("backward_rawptr", &TorchJITProduct::backward_device_rawptrs)
        .def("__len__", [](const c10::intrusive_ptr<TorchJITProduct>& test) -> int64_t {
            return 0;
        })
        .def_readonly("L3_dim", &TorchJITProduct::L3_dim)
        .def("__eq__", [](const c10::IValue & self, const c10::IValue& other) -> bool {
            return self.is(other); 
        })
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<TorchJITProduct>& self)
                -> tuple<string, Map_t, Map_t, Map_t, Map_t> {
                return tuple(self->internal.jit.kernel_plaintext, self->fwd_dict, self->bwd_dict, self->dbl_bwd_dict, self->kernel_dims);
            },
            // __setstate__
            [](tuple<string, Map_t, Map_t, Map_t, Map_t> state)
                -> c10::intrusive_ptr<TorchJITProduct> {
                return c10::make_intrusive<TorchJITProduct>(get<0>(state), get<1>(state), get<2>(state), get<3>(state), get<4>(state));
            });

    m.def("jit_tp_forward(__torch__.torch.classes.libtorch_tp_jit.TorchJITProduct jit, Tensor L1_in, Tensor L2_in, Tensor W) -> Tensor");
    m.def("jit_tp_backward(__torch__.torch.classes.libtorch_tp_jit.TorchJITProduct jit, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad) -> (Tensor, Tensor, Tensor)");
    m.def("jit_tp_double_backward(__torch__.torch.classes.libtorch_tp_jit.TorchJITProduct jit, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor L1_dgrad, Tensor L2_dgrad, Tensor W_dgrad) -> (Tensor, Tensor, Tensor, Tensor)");


    m.class_<TorchJITConv>("TorchJITConv")
        .def(torch::init<string, Map_t, Map_t, Map_t, Map_t>())
        .def("__obj_flatten__", &TorchJITConv::__obj_flatten__)
        .def("exec_conv_rawptrs", &TorchJITConv::exec_conv_rawptrs)
        .def("backward_rawptrs", &TorchJITConv::backward_rawptrs)
        .def("double_backward_rawptrs", &TorchJITConv::double_backward_rawptrs)
        .def("__len__", [](const c10::intrusive_ptr<TorchJITConv>& test) -> int64_t {
            return 0;
        })
        .def_readonly("L3_dim", &TorchJITConv::L3_dim)
        .def("__eq__", [](const c10::IValue & self, const c10::IValue& other) -> bool {
            return self.is(other); 
        })
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<TorchJITConv>& self)
                -> tuple<string, Map_t, Map_t, Map_t, Map_t> {
                return tuple(self->internal.jit.kernel_plaintext, self->fwd_dict, self->bwd_dict, self->dbl_bwd_dict, self->kernel_dims);
            },
            // __setstate__
            [](tuple<string, Map_t, Map_t, Map_t, Map_t> state)
                -> c10::intrusive_ptr<TorchJITConv> {
                return c10::make_intrusive<TorchJITConv>(get<0>(state), get<1>(state), get<2>(state), get<3>(state), get<4>(state));
            });

    m.def("jit_conv_forward(__torch__.torch.classes.libtorch_tp_jit.TorchJITConv jit, Tensor L1_in, Tensor L2_in, Tensor W, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> Tensor");
    m.def("jit_conv_backward(__torch__.torch.classes.libtorch_tp_jit.TorchJITConv jit, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> (Tensor, Tensor, Tensor)");
    m.def("jit_conv_double_backward(__torch__.torch.classes.libtorch_tp_jit.TorchJITConv jit, Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad, Tensor L1_dgrad, Tensor L2_dgrad, Tensor W_dgrad, Tensor rows, Tensor cols, Tensor workspace, Tensor transpose_perm) -> (Tensor, Tensor, Tensor, Tensor)");
};

TORCH_LIBRARY_IMPL(libtorch_tp_jit, CUDA, m) { 
    m.impl("jit_tp_forward", &jit_tp_forward);
    m.impl("jit_tp_backward", &jit_tp_backward);
    m.impl("jit_tp_double_backward", &jit_tp_double_backward);

    m.impl("jit_conv_forward", &jit_conv_forward);
    m.impl("jit_conv_backward", &jit_conv_backward);
    m.impl("jit_conv_double_backward", &jit_conv_double_backward);
};

PYBIND11_MODULE(libtorch_tp_jit, m) {}