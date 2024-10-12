{# Jinja2 Template #}

{%- set warps_per_block_forward = divide(forward_config.num_threads, forward_config.warp_size) %}

{% include 'common.cuh' %}
{%- from 'macros.jinja' import declare_smem_arrays with context %}

#define THREADS_PER_WARP {{ forward_config.warp_size }}

{%- macro set_launch_bound_variables(config) %}
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = t_idx / THREADS_PER_WARP;
    int lane_id = t_idx % THREADS_PER_WARP;
    int warp_loc = warp_id % {{ warps_per_block_forward }};
    size_t warps_launched = blockDim.x * gridDim.x / THREADS_PER_WARP;
    size_t nnz_per_warp = (num_products + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
    size_t end = min(start + nnz_per_warp, num_products);
{%- endmacro %}

__device__ __forceinline__ void forward_loop_unroll(const float* __restrict__ L1_smem, const float* __restrict__ L2_smem, float* __restrict__ L3_smem) {
    float l1_vec[{{L1.irrep_lengths | max}}];
    float l2_vec[{{L2.irrep_lengths | max}}];
    float l3_vec[{{L3.irrep_lengths | max}}];

    {%- set num_interact = interactions | length %}
    {%- for k in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[k] %}

        //==========================================
        {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
            #pragma unroll
            for(int j = 0; j < {{L1.irrep_lengths[u]}}; j++)
                l1_vec[j] = L1_smem[{{L1.mults[u]}} * j + {{ L1.offsets[u]}}];
        {%- endif %}

        {%- if k == 0 or interactions[k][1] != interactions[k-1][1] %}
            #pragma unroll
            for(int j = 0; j < {{L2.irrep_lengths[v]}}; j++)
                l2_vec[j] = L2_smem[j + {{L2.offsets[v]}}];
        {%- endif %}

        {%- if k == 0 or interactions[k][2] != interactions[k-1][2] %}
            #pragma unroll
            for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++)
                l3_vec[j] = 0.0f;
        {%- endif %}

        {%- for i in range(tensor.nnz) %}
            {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
            l3_vec[{{coord3}}] += {{value}} * l1_vec[{{coord1}}] * l2_vec[{{coord2}}];
        {%- endfor %}

        // TODO: Should change to += accumulate, buffer the output in shared memory. 
        {%- if k == num_interact - 1 or interactions[k][2] != interactions[k+1][2] %}
            #pragma unroll
            for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++)
                L3_smem[{{L3.mults[w]}} * j + {{L3.offsets[w]}}] = l3_vec[j];
        {%- endif %}
    {%- endfor %}
}

// Assumes all reps in L1 have multiplicity 32, all reps in L2 have multiplicity 1. 
// column-major data layout 
__global__ void loop_unroll_many_to_one(
    size_t num_products, float* L1_in, float* L2_in, float* L3_out) {

    {{ set_launch_bound_variables(forward_config) }}

    {{ declare_smem_arrays({
        "common": [],
        "per_warp": [
            ("L1_smem", "float", L1.rep_len),
            ("L2_smem", "float", L2.rep_len),
            ("L3_smem", "float", L3.rep_len)
        ]}, "warp_loc", forward_config)}}

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1.rep_len}} + lane_id;
        float* l2_shft = L2_in + i * {{L2.rep_len}} + lane_id; 
        float* l3_shft = L3_out + i * {{L3.rep_len}} + lane_id;

        ROW_OPERATION({{L1.rep_len}}, j, L1_smem[j + lane_id] = l1_shft[j];)
        ROW_OPERATION({{L2.rep_len}}, j, L2_smem[j + lane_id] = l2_shft[j];)
        ROW_OPERATION({{L3.rep_len}}, j, L3_smem[j + lane_id] = 0.0f;)

        __syncwarp();
        forward_loop_unroll(L1_smem + lane_id, L2_smem, L3_smem + lane_id);
        __syncwarp();

        ROW_OPERATION({{L3.rep_len}}, j, l3_shft[j] = L3_smem[j + lane_id];)
    }
}

__device__ __forceinline__ void backward_loop_unroll(
        const float* __restrict__ L1_smem,
        const float* __restrict__ L2_smem,
        const float* __restrict__ weights_smem,
        const float* __restrict__ L3_grad_smem,

        float* __restrict__ L1_grad_smem,
        float* __restrict__ L2_grad_smem,
        float* __restrict__ weights_grad_smem) {

    float l1_vec[{{L1.irrep_lengths  | max}}]; 
    float l1_grad[{{L1.irrep_lengths | max}}]; 
    float l2_vec[{{L2.irrep_lengths  | max}}];
    float l2_grad[{{L1.irrep_lengths | max}}]; 
    float l3_grad[{{L3.irrep_lengths | max}}];

    float weight, weight_grad;

    {% set num_scratch_reg = 2 %}
    float scratch1[{{num_scratch_reg}}];
    float scratch2[{{num_scratch_reg}}];

    {%- set num_interact = interactions | length %}
    {%- for k in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[k] %}
        weight = weights_smem[{{weights.offsets[k]}} * THREADS_PER_WARP];
        weight_grad = weights_grad_smem[{{weights.offsets[k]}} * THREADS_PER_WARP];

        //==========================================
        {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
            #pragma unroll
            for(int j = 0; j < {{L1.irrep_lengths[u]}}; j++) {
                l1_vec[j] = L1_smem[{{L1.mults[u]}} * j + {{ L1.offsets[u]}}];
                l1_grad[j] = L1_grad_smem[{{L1.mults[u]}} * j + {{ L1.offsets[u]}}];
            }
        {%- endif %}

        {%- if k == 0 or interactions[k][1] != interactions[k-1][1] %}
            #pragma unroll
            for(int j = 0; j < {{L2.irrep_lengths[v]}}; j++) {
                l2_vec[j] = L2_smem[j + {{L2.offsets[v]}}];
                l2_grad[j] = L2_grad_smem[j + {{L2.offsets[v]}}];
            }
        {%- endif %}

        {%- if k == 0 or interactions[k][2] != interactions[k-1][2] %}
            #pragma unroll
            for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++)
                l3_grad[j] = L3_grad_smem[{{L3.mults[u]}} * j + {{ L3.offsets[u]}}];
        {%- endif %}

        {%- for i in range(tensor.nnz) %} 
            {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
            scratch1[{{i % num_scratch_reg}}] = l3_grad[{{coord3}}] * {{value}}; 
            weight_grad += scratch1[{{i % num_scratch_reg}}] * l2_vec[{{coord2}}] * l1_vec[{{coord1}}];
            scratch2[{{i % num_scratch_reg}}] = scratch1[{{i % num_scratch_reg}}] * weight;
            l2_grad[{{coord2}}] += scratch2[{{i % num_scratch_reg}}] * l1_vec[{{coord1}}];
            l1_grad[{{coord2}}] += scratch2[{{i % num_scratch_reg}}] * l2_vec[{{coord2}}];
        {%- endfor %}

        // Storeback
        {%- if k == num_interact - 1 or interactions[k][0] != interactions[k+1][0] %}
            #pragma unroll
            for(int j = 0; j < {{L1.irrep_lengths[u]}}; j++)
                L1_grad_smem[{{L1.mults[u]}} * j + {{ L1.offsets[u]}}] = l1_grad[j];
        {%- endif %}

        {%- if k == num_interact - 1 or interactions[k][1] != interactions[k+1][1] %}
            #pragma unroll
            for(int j = 0; j < {{L2.irrep_lengths[v]}}; j++)
                L2_grad_smem[j + {{L2.offsets[v]}}] = l2_grad[j];
        {%- endif %}

        weights_grad_smem[{{weights.offsets[k]}} * THREADS_PER_WARP] = weight_grad; 
    {%- endfor %}
}

/*
* Backward pass kernel. Currently assumes that each tensor product
* has a unique set of weights. 
* 
* Inputs:
*   L1_in, L2_in, weights, L3_grad
* Outputs:
*   L1_grad, L2_grad, weights_grad 
*/
__global__ void loop_unroll_backward(
    size_t num_products,
    float* L1_in, float* L1_grad,
    float* L2_in, float* L2_grad,
    float* weights, float* weights_grad,
    float* L3_grad) {

    {{ set_launch_bound_variables() }}

    {{ declare_smem_arrays({
        "common": [],
        "per_warp": [
            ("L1_smem", "float", L1.rep_len),
            ("L1_grad_smem", "float", L1.rep_len),
            ("L2_smem", "float", L2.rep_len),
            ("L2_grad_smem", "float", L2.rep_len),
            ("weights_smem", "float", weights.total_len),
            ("weights_grad_smem", "float", weights.total_len),
            ("L3_grad_smem", "float", L3.rep_len)
        ]}, "warp_loc", backward_config)}}

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1.rep_len}} + lane_id;
        float* l2_shft = L2_in + i * {{L2.rep_len}} + lane_id; 
        float* l3_shft = L3_grad + i * {{L3.rep_len}} + lane_id;
        float* weights_shft = weights + i * {{weights.total_len}} + lane_id;

        ROW_OPERATION({{L1.rep_len}}, j, L1_smem[j + lane_id] = l1_shft[j];)
        ROW_OPERATION({{L2.rep_len}}, j, L2_smem[j + lane_id] = l2_shft[j];)
        ROW_OPERATION({{L3.rep_len}}, j, L3_grad_smem[j + lane_id] = l3_shft[j];)
        ROW_OPERATION({{weights.total_len}}, j, weights_smem[j + lane_id] = weights_shft[j];)

        ROW_OPERATION({{L1.rep_len}}, j, L1_grad_smem[j + lane_id] = 0.0f;)
        ROW_OPERATION({{L2.rep_len}}, j, L2_grad_smem[j + lane_id] = 0.0f;)
        ROW_OPERATION({{weights.total_len}}, j, weights_grad_smem[j + lane_id] = 0.0f;)

        __syncwarp();
        //backward_loop_unroll(L1_smem + lane_id, L2_smem, weights_smem + lane_id, L3_grad_smem + lane_id,
        //        L1_grad + lane_id, L2_grad, weights_grad + lane_id);
        __syncwarp();

        float* l1_grad_shft = L1_grad + i * {{L1.rep_len}} + lane_id;
        float* l2_grad_shft = L2_grad + i * {{L2.rep_len}} + lane_id; 
        float* weights_grad_shft = weights_grad + i * {{weights.total_len}} + lane_id;

        ROW_OPERATION({{L1.rep_len}}, j, l1_grad_shft[j] = L1_grad_smem[j + lane_id];)
        ROW_OPERATION({{L2.rep_len}}, j, l2_grad_shft[j] = L2_grad_smem[j + lane_id];)
        ROW_OPERATION({{weights.total_len}}, j, weights_grad_shft[j] = weights_grad_smem[j + lane_id];)
    }
}