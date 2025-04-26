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
{{ generate_segment_kernel_forward(i, segment) }}
{%- endfor %}

struct ConvData {
    void* rows;
    void* cols;
    unsigned long nnz;
    unsigned long node_count;
};


{%- macro generate_fixup_kernel(name, warp_size, dim, fixup_offset) %}
__global__ void {{name}}(void* workspace, IRREP_T* dst_ptr) {
    /*
    *  Workspace consists of: 
    *     fixup_dim * warps_launched * sizeof(IRREP_T): Data
    *     warps_launched * sizeof(long): Destinations to accumulate to 
    */
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = t_idx / {{ warp_size }};
    int lane_id = t_idx % {{ warp_size }};
    size_t warps_launched = blockDim.x * gridDim.x / {{ warp_size }};

    IRREP_T* data = (IRREP_T*) workspace;
    {{idx_type}}* dst_idxs = ({{idx_type}}*) ((char*) workspace + {{fixup_offset}}); 

    if((warp_id == 0) || (dst_idxs[warp_id] != -1 && dst_idxs[warp_id - 1] != dst_idxs[warp_id])) {
        size_t current = warp_id;
        {{idx_type}} dst_row_idx = dst_idxs[warp_id];
 
        if(dst_row_idx != -1) {
            while(current < warps_launched && dst_idxs[current] == dst_row_idx) {
                IRREP_T* src = data + {{dim}} * current + lane_id;
                IRREP_T* dst = dst_ptr + {{dim}} * dst_row_idx + lane_id;
                ROW_OPERATION({{dim}}, i, dst[i] += src[i];)
                current++;
            }
        }
    }
}
{%- endmacro %}

{{ generate_fixup_kernel("fixup_forward", forward_schedule.launch_config.warp_size, forward_schedule.L3.dim, forward_workspace_offset) }}

__global__ void forward(
        IRREP_T* L1_in,
        IRREP_T* L2_in,
        WEIGHT_T* weights,
        IRREP_T* L3_out,
        ConvData c,
        void* workspace_raw) {
 
    extern __shared__ char s[];
    size_t num_products = c.nnz;
    {{idx_type}}* rows = ({{idx_type}}*) c.rows;
    {{idx_type}}* cols = ({{idx_type}}*) c.cols;

    {{ set_launch_bound_variables(forward_schedule.launch_config) }}

    IRREP_T* workspace = (IRREP_T*) workspace_raw;
    {{idx_type}}* dst_idxs = ({{idx_type}}*) ((char*) workspace + {{forward_workspace_offset}}); 

    if(lane_id == 0) {
        if(start < end) {
            dst_idxs[warp_id] = rows[start];
        }
        else {
            dst_idxs[warp_id] = -1; 
        }
    }

    {%- set tpp = forward_schedule.updated_config %}
    char* smem = s + {{forward_schedule.memory_per_warp}} * warp_loc; 

    {%- for i, segment in enumerate(forward_schedule.segments) %} {
        {{ declare_smem_variables(segment, "smem") }}

        bool firstSegment = true;
        ROW_OPERATION({{segment.L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)

        for(size_t i = start; i < end; i++) {
            {{idx_type}} row = rows[i]; {{idx_type}} col = cols[i];

            IRREP_T* l1 = L1_in + col * {{forward_schedule.L1.dim}} + lane_id;
            IRREP_T* l2 = L2_in + i * {{forward_schedule.L2.dim}} + lane_id; 
            IRREP_T* l3 = L3_out + row * {{forward_schedule.L3.dim}} + lane_id;
            WEIGHT_T* w = weights + i * {{tpp.weight_numel}};

            {{ load_ir_segments(segment.L1Map, "l1", "L1_smem", "j") }}
            {{ load_ir_segments(segment.L2Map, "l2", "L2_smem", "j") }}

            {%- if not forward_schedule.stream_weights %}
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = w[{{segment.weight_offset}} + j + lane_id];)
            {%- endif %}

            __syncwarp();
            forward_loop_unroll_{{i}}(L1_smem, L2_smem, w, weights_smem, L3_smem, scratch_smem, lane_id);
            __syncwarp();

            bool changeRow = (i < end - 1) && (row != rows[i+1]);

            if(changeRow || i == end - 1) {
                IRREP_T* dst = l3;
                if(firstSegment) {
                    dst = workspace + {{forward_schedule.L3.dim}} * warp_id + lane_id;
                    firstSegment = false;
                }
                {{ store_ir_segments(segment.L3Map, "dst", "L3_smem", "j") }}
                __syncwarp();

                ROW_OPERATION({{segment.L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)
            }
        } 
    } {%- endfor %}
}

{%- for i, segment in enumerate(backward_schedule.segments) %}
{{ generate_segment_kernel_backward(i, segment, backward_schedule.launch_config.warp_size) }}
{%- endfor %}

{{ generate_fixup_kernel("fixup_backward", backward_schedule.launch_config.warp_size, backward_schedule.L1.dim, backward_workspace_offset) }}

__global__ void backward(
        IRREP_T* L1_in, IRREP_T* L1_grad,
        IRREP_T* L2_in, IRREP_T* L2_grad,
        WEIGHT_T* weights, WEIGHT_T* weights_grad,
        IRREP_T* L3_grad, ConvData c, void* workspace_raw, 
        {{idx_type}}* transpose_perm) {

    extern __shared__ char s[];
    size_t num_products = c.nnz;

    // Note the transpose below (cols -> rows, rows -> cols)
    {{idx_type}}* rows = ({{idx_type}}*) c.cols;
    {{idx_type}}* cols = ({{idx_type}}*) c.rows;
    {{idx_type}}* tperm = ({{idx_type}}*) transpose_perm;

    {{ set_launch_bound_variables(backward_schedule.launch_config) }}

    {%- set tpp = backward_schedule.updated_config %}
    char* smem = s + {{backward_schedule.memory_per_warp}} * warp_loc; 

    IRREP_T* workspace = (IRREP_T*) workspace_raw;
    {{idx_type}}* dst_idxs = ({{idx_type}}*) ((char*) workspace + {{backward_workspace_offset}}); 

    if(lane_id == 0) {
        if(start < end) {
            dst_idxs[warp_id] = cols[start];
        }
        else {
            dst_idxs[warp_id] = -1; 
        }
    }

    {%- for i, segment in enumerate(backward_schedule.segments) %} {
        {{ declare_smem_variables(segment, "smem") }}
        
        bool firstSegment = true;
        ROW_OPERATION({{segment.L1.dim}}, j, L1_grad_smem[j + lane_id] = 0.0f;)

        for(size_t i = start; i < end; i++) {
            {{idx_type}} row = rows[i]; {{idx_type}} col = cols[i];
            {{idx_type}} tperm_idx = tperm[i];

            IRREP_T* l1_shft = L1_in + col * {{backward_schedule.L1.dim}} + lane_id;
            IRREP_T* l2_shft = L2_in + tperm_idx * {{backward_schedule.L2.dim}} + lane_id; 
            IRREP_T* l3_shft = L3_grad + row * {{backward_schedule.L3.dim}} + lane_id;

            {%- if not tpp.shared_weights %} 
                WEIGHT_T* w = weights + tperm_idx * {{tpp.weight_numel}}; 
                WEIGHT_T* wgrad = weights_grad + tperm_idx * {{tpp.weight_numel}}; 
            {%- else %}
                WEIGHT_T* w = weights; 
                WEIGHT_T* wgrad = weights_grad; 
            {%- endif %}
            WEIGHT_T* weights_shft = w + lane_id;

            {{ load_ir_segments(segment.L1Map, "l1_shft", "L1_smem", "j") }}
            {{ load_ir_segments(segment.L2Map, "l2_shft", "L2_smem", "j") }}
            {{ load_ir_segments(segment.L3Map, "l3_shft", "L3_grad_smem", "j") }}

            __syncwarp();
            ROW_OPERATION({{segment.L2.dim}}, j, L2_grad_smem[j + lane_id] = 0.0f;)

            {%- if not backward_schedule.stream_weights %}
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = weights_shft[{{segment.weight_offset}} + j];)
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_grad_smem[j + lane_id] = 0.0;)
            {%- endif %}

            IRREP_T* l1_grad_shft = L1_grad + col * {{backward_schedule.L1.dim}} + lane_id;
            IRREP_T* l2_grad_shft = L2_grad + tperm_idx * {{backward_schedule.L2.dim}} + lane_id;
            WEIGHT_T* weights_grad_shft = wgrad + lane_id; 

            __syncwarp();
            backward_loop_unroll_{{i}}(L1_smem, L2_smem, w, weights_smem, L3_grad_smem,
                    L1_grad_smem, L2_grad_smem, wgrad, weights_grad_smem, scratch_smem, lane_id);
            __syncwarp();

            bool changeRow = (i < end - 1) && (col != cols[i+1]);

            if(changeRow || i == end - 1) {
                IRREP_T* dst = l1_grad_shft;
                if(firstSegment) {
                    dst = workspace + {{backward_schedule.L1.dim}} * warp_id + lane_id;
                    firstSegment = false;
                }
                {{ store_ir_segments(segment.L1Map, "dst", "L1_grad_smem", "j") }}
                __syncwarp();
                ROW_OPERATION({{segment.L1.dim}}, j, L1_grad_smem[j + lane_id] = 0.0f;)
            }

            {{ store_ir_segments(segment.L2Map, "l2_grad_shft", "L2_grad_smem", "j") }}

            {%- if not backward_schedule.stream_weights%}
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_grad_shft[{{segment.weight_offset}} + j] = weights_grad_smem[j + lane_id];)
            {%- endif %}
        }
    } {%- endfor %}
}