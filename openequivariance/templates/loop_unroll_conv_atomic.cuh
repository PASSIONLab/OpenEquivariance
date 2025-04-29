{# Jinja2 Template #}

{% include 'common.cuh' %}

{%- from 'macros.jinja' import
        transpose_load, transpose_store, 
        load_ir_segments, store_ir_segments, 
        declare_smem_variables,
        set_launch_bound_variables with context %}

#define THREADS_PER_WARP {{ forward_schedule.launch_config.warp_size }} // Warp size should be the same for forward and backward
#define FULL_MASK 0xffffffff

{%- from 'loop_unroll_tp.cuh' import 
        generate_segment_kernel_forward, 
        generate_segment_kernel_backward %}

using IRREP_T  = {{ forward_schedule.irrep_dtype_cstr }};
using WEIGHT_T = {{ forward_schedule.weight_dtype_cstr }};

{%- for i, segment in enumerate(forward_schedule.segments) %}
{{ generate_segment_kernel_forward(i, segment, forward_schedule.launch_config.warp_size) }}
{%- endfor %}

struct ConvData {
    void* rows;
    void* cols;
    unsigned long nnz;
    unsigned long node_count;
};

__global__ void fixup_forward(void* workspace, IRREP_T* dst_ptr) {
    // Empty, no fixup
}

__global__ void fixup_backward(void* workspace, IRREP_T* dst_ptr) {
    // Empty, no fixup
}

__global__ void forward(
        IRREP_T* L1_in,
        IRREP_T* L2_in,
        WEIGHT_T* weights,
        IRREP_T* L3_out,
        ConvData c,
        void* workspace) {
 
    extern __shared__ char s[];
    size_t num_products = c.nnz;
    unsigned {{idx_type}}* rows = (unsigned {{idx_type}}*) c.rows;
    unsigned {{idx_type}}* cols = (unsigned {{idx_type}}*) c.cols;

    {{ set_launch_bound_variables(forward_schedule.launch_config) }}
    {%- set tpp = forward_schedule.updated_config %}
    char* smem = s + {{forward_schedule.memory_per_warp}} * warp_loc; 

    {%- for i, segment in enumerate(forward_schedule.segments) %} {
        {{ declare_smem_variables(segment, "smem") }}

        for(size_t i = start; i < end; i++) {
            unsigned {{idx_type}} row = rows[i]; unsigned {{idx_type}} col = cols[i];

            IRREP_T* l1 = L1_in + col * {{forward_schedule.L1.dim}} + lane_id;
            IRREP_T* l2 = L2_in + i * {{forward_schedule.L2.dim}} + lane_id; 
            IRREP_T* l3 = L3_out + row * {{forward_schedule.L3.dim}} + lane_id;
            WEIGHT_T* w = weights + i * {{tpp.weight_numel}};

            __syncwarp();
            {{ load_ir_segments(segment.L1Map, "l1", "L1_smem", "j") }}
            {{ load_ir_segments(segment.L2Map, "l2", "L2_smem", "j") }}
            ROW_OPERATION({{segment.L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)

            {%- if not forward_schedule.stream_weights %}
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = w[{{segment.weight_offset}} + j + lane_id];)
            {%- endif %}

            __syncwarp();
            forward_loop_unroll_{{i}}(L1_smem, L2_smem, w, weights_smem, L3_smem, scratch_smem, lane_id);
            __syncwarp();

            {{ store_ir_segments(segment.L3Map, "l3", "L3_smem", "j") }}
        } 
    } {%- endfor %}
}

{%- for i, segment in enumerate(backward_schedule.segments) %}
{{ generate_segment_kernel_backward(i, segment, backward_schedule.launch_config.warp_size) }}
{%- endfor %}

__global__ void backward(
        IRREP_T* L1_in, IRREP_T* L1_grad,
        IRREP_T* L2_in, IRREP_T* L2_grad,
        WEIGHT_T* weights, WEIGHT_T* weights_grad,
        IRREP_T* L3_grad, ConvData c, void* workspace, unsigned {{idx_type}}* transpose_perm) {

    extern __shared__ char s[];
    size_t num_products = c.nnz;
    unsigned {{idx_type}}* rows = (unsigned {{idx_type}}*) c.rows;
    unsigned {{idx_type}}* cols = (unsigned {{idx_type}}*) c.cols;

    {{ set_launch_bound_variables(backward_schedule.launch_config) }}
    char* smem = s + {{backward_schedule.memory_per_warp}} * warp_loc; 

    for(size_t i = start; i < end; i++) {
        {%- set tpp = backward_schedule.updated_config %}
        unsigned {{idx_type}} row = rows[i]; unsigned {{idx_type}} col = cols[i];

        IRREP_T* l1_shft = L1_in + col * {{backward_schedule.L1.dim}} + lane_id;
        IRREP_T* l2_shft = L2_in + i * {{backward_schedule.L2.dim}} + lane_id; 
        IRREP_T* l3_shft = L3_grad + row * {{backward_schedule.L3.dim}} + lane_id;

        {%- if not tpp.shared_weights %} 
        WEIGHT_T* w = weights + i * {{tpp.weight_numel}}; 
        WEIGHT_T* wgrad = weights_grad + i * {{tpp.weight_numel}}; 
        {%- else %}
        WEIGHT_T* w = weights; 
        WEIGHT_T* wgrad = weights_grad; 
        {%- endif %}
        WEIGHT_T* weights_shft = w + lane_id;

        {%- for i, segment in enumerate(backward_schedule.segments) %} {
            {{ declare_smem_variables(segment, "smem") }}

            {{ load_ir_segments(segment.L1Map, "l1_shft", "L1_smem", "j") }}
            {{ load_ir_segments(segment.L2Map, "l2_shft", "L2_smem", "j") }}
            {{ load_ir_segments(segment.L3Map, "l3_shft", "L3_grad_smem", "j") }}

            __syncwarp();
            {%- if not segment.L1Map.persist_load %}
                ROW_OPERATION({{segment.L1.dim}}, j, L1_grad_smem[j + lane_id] = 0.0f;)
            {%- endif %}
            {%- if not segment.L2Map.persist_load %}
                ROW_OPERATION({{segment.L2.dim}}, j, L2_grad_smem[j + lane_id] = 0.0f;)
            {%- endif %}

            {%- if not backward_schedule.stream_weights%}
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = weights_shft[{{segment.weight_offset}} + j];)
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_grad_smem[j + lane_id] = 0.0;)
            {%- endif %}

            IRREP_T* l1_grad_shft = L1_grad + col * {{backward_schedule.L1.dim}} + lane_id;
            IRREP_T* l2_grad_shft = L2_grad + i * {{backward_schedule.L2.dim}} + lane_id;
            WEIGHT_T* weights_grad_shft = wgrad + lane_id;

            __syncwarp();
            backward_loop_unroll_{{i}}(L1_smem, L2_smem, w, weights_smem, L3_grad_smem,
                    L1_grad_smem, L2_grad_smem, wgrad, weights_grad_smem, scratch_smem, lane_id);
            __syncwarp();

            {{ store_ir_segments(segment.L1Map, "l1_grad_shft", "L1_grad_smem", "j") }}
            {{ store_ir_segments(segment.L2Map, "l2_grad_shft", "L2_grad_smem", "j") }}

            {%- if not backward_schedule.stream_weights%}
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_grad_shft[{{segment.weight_offset}} + j] = weights_grad_smem[j + lane_id];)
            {%- endif %}
        } {%- endfor %}
    }
}


__global__ void double_backward_A(
        IRREP_T* L1_in, IRREP_T* L2_in, WEIGHT_T* W, IRREP_T* L3_grad,
        IRREP_T* L1_dgrad, IRREP_T* L2_dgrad, IRREP_T* W_dgrad,
        IRREP_T* L1_grad, IRREP_T* L2_grad, WEIGHT_T* W_grad, IRREP_T* L3_dgrad
        ConvData c, void* workspace, unsigned {{idx_type}}* transpose_perm) {

    printf("Hello world!");
}


__global__ void double_backward_B(
        IRREP_T* L1_in, IRREP_T* L2_in, WEIGHT_T* W, IRREP_T* L3_grad,
        IRREP_T* L1_dgrad, IRREP_T* L2_dgrad, IRREP_T* W_dgrad,
        IRREP_T* L1_grad, IRREP_T* L2_grad, WEIGHT_T* W_grad, IRREP_T* L3_dgrad
        ConvData c, void* workspace, unsigned {{idx_type}}* transpose_perm) {

    printf("Hello world!");
}

