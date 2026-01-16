{# Jinja2 Template #}

{% include 'common.cuh' %}
{%- from 'macros.jinja' import
        transpose_load, transpose_store, 
        load_ir_segments, load_ir_segments_force,
        store_ir_segments, declare_smem_variables,
        set_launch_bound_variables, launch_bounds
        with context%}

{%- from 'loop_unroll_tp.cuh' import 
        generate_segment_kernel_forward, 
        generate_segment_kernel_backward %}

#define THREADS_PER_WARP {{ forward_schedule.launch_config.warp_size }} // Warp size should be the same for forward and backward
#define FULL_MASK 0xffffffff

using IRREP_T  = {{ forward_schedule.irrep_dtype_cstr }};
using WEIGHT_T = {{ forward_schedule.weight_dtype_cstr }};

{%- for i, segment in enumerate(forward_schedule.segments) %}
{{ generate_segment_kernel_forward(i, segment, forward_schedule.launch_config.warp_size) }}
{%- endfor %}

__global__ void 
{{ launch_bounds(forward_schedule) }}
forward(size_t num_products, IRREP_T* L1_in, IRREP_T* L2_in, IRREP_T* L3_out, WEIGHT_T* weights) {

    extern __shared__ char s[];
    {{ set_launch_bound_variables(forward_schedule.launch_config) }}
    {%- set tpp = forward_schedule.updated_config %}
    char* smem = s + {{forward_schedule.memory_per_warp}} * warp_loc; 

    for(size_t i = start; i < end; i++) {
        IRREP_T* l1 = L1_in + i * {{forward_schedule.L1.dim}} + lane_id;
        IRREP_T* l2 = L2_in + i * {{forward_schedule.L2.dim}} + lane_id; 
        IRREP_T* l3 = L3_out + i * {{forward_schedule.L3.dim}} + lane_id;

        {%- if not tpp.shared_weights %} 
            WEIGHT_T* w = weights + i * {{tpp.weight_numel}}; 
        {%- else %}
            WEIGHT_T* w = weights; 
        {%- endif %}

        {%- for i, segment in enumerate(forward_schedule.segments) %} {
            __syncwarp();
            {{ declare_smem_variables(segment, "smem") }}
            {{ load_ir_segments(segment.L1Map, "l1", "L1_smem", "j") }}
            {{ load_ir_segments(segment.L2Map, "l2", "L2_smem", "j") }}
            ROW_OPERATION({{segment.L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)

            {% if not forward_schedule.stream_weights%}
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = w[{{segment.weight_offset}} + j + lane_id];)
            {% endif %}

            __syncwarp();
            forward_loop_unroll_{{i}}(L1_smem, L2_smem, w, weights_smem, L3_smem, scratch_smem, lane_id);
            __syncwarp();
    
            {{ store_ir_segments(segment.L3Map, "l3", "L3_smem", "j") }}
        } {%- endfor %}
    }
}

{%- for i, segment in enumerate(backward_schedule.segments) %}
{{ generate_segment_kernel_backward(i, segment, backward_schedule.launch_config.warp_size) }}
{%- endfor %}

__global__ void 
{{ launch_bounds(backward_schedule) }}
backward(size_t num_products,
        IRREP_T* L1_in, IRREP_T* L1_grad,
        IRREP_T* L2_in, IRREP_T* L2_grad,
        WEIGHT_T* weights, WEIGHT_T* weights_grad,
        IRREP_T* L3_grad) {
    extern __shared__ char s[];
    {{ set_launch_bound_variables(backward_schedule.launch_config) }}
    char* smem = s + {{backward_schedule.memory_per_warp}} * warp_loc; 

    for(size_t i = start; i < end; i++) {
        {%- set tpp = backward_schedule.updated_config %}
        IRREP_T* l1_shft = L1_in + i * {{backward_schedule.L1.dim}} + lane_id;
        IRREP_T* l2_shft = L2_in + i * {{backward_schedule.L2.dim}} + lane_id; 
        IRREP_T* l3_shft = L3_grad + i * {{backward_schedule.L3.dim}} + lane_id;

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

            __syncwarp();
            backward_loop_unroll_{{i}}(L1_smem, L2_smem, w, weights_smem, L3_grad_smem,
                    L1_grad_smem, L2_grad_smem, wgrad, weights_grad_smem, scratch_smem, lane_id);
            __syncwarp();

            IRREP_T* l1_grad_shft = L1_grad + i * {{backward_schedule.L1.dim}} + lane_id;
            IRREP_T* l2_grad_shft = L2_grad + i * {{backward_schedule.L2.dim}} + lane_id;

            {%- if not tpp.shared_weights %}
                WEIGHT_T* weights_grad_shft = weights_grad + i * {{backward_schedule.updated_config.weight_numel}} + lane_id;
            {%- else %}
                WEIGHT_T* weights_grad_shft = weights_grad + lane_id;
            {%- endif %}

            {{ store_ir_segments(segment.L1Map, "l1_grad_shft", "L1_grad_smem", "j") }}
            {{ store_ir_segments(segment.L2Map, "l2_grad_shft", "L2_grad_smem", "j") }}

            {%- if not backward_schedule.stream_weights%}
                {%- if not tpp.shared_weights %}
                    ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_grad_shft[{{segment.weight_offset}} + j] = weights_grad_smem[j + lane_id];)
                {%- else %}  
                    ROW_OPERATION({{segment.problem.weight_numel}}, j, atomicAdd(weights_grad_shft + {{segment.weight_offset}} + j, weights_grad_smem[j + lane_id]);)
                {%- endif %}
            {%- endif %}
        } {%- endfor %}
    }
}

/*
* The double backward kernel involves two passes: one combining three forward calls (A),
* and the second combining three backward calls (B).
*/
__global__ void 
{{ launch_bounds(forward_schedule) }}
double_backward_A(
    size_t num_products,
    IRREP_T* L1_in, IRREP_T* L2_in, WEIGHT_T* W, IRREP_T* L3_grad, // Inputs of backward op 
    IRREP_T* L1_dgrad, IRREP_T* L2_dgrad, IRREP_T* W_dgrad, // Gradients w.r.t outputs of backward op
    IRREP_T* L1_grad, IRREP_T* L2_grad, WEIGHT_T* W_grad, IRREP_T* L3_dgrad) {

    extern __shared__ char s[];
    {{ set_launch_bound_variables(forward_schedule.launch_config) }}
    {%- set tpp = forward_schedule.updated_config %}
    char* smem = s + {{forward_schedule.memory_per_warp}} * warp_loc; 

    for(size_t i = start; i < end; i++) {
        IRREP_T* l1 = L1_in + i * {{forward_schedule.L1.dim}} + lane_id;
        IRREP_T* l2 = L2_in + i * {{forward_schedule.L2.dim}} + lane_id; 
        IRREP_T* l3 = L3_dgrad + i * {{forward_schedule.L3.dim}} + lane_id;

        IRREP_T* l1_dgrad = L1_dgrad + i * {{forward_schedule.L1.dim}} + lane_id;
        IRREP_T* l2_dgrad = L2_dgrad + i * {{forward_schedule.L2.dim}} + lane_id;

        {%- if not tpp.shared_weights %} 
            WEIGHT_T* w = W + i * {{tpp.weight_numel}};
            WEIGHT_T* w_dgrad = W_dgrad + i * {{tpp.weight_numel}};
        {%- else %}
            WEIGHT_T* w = W;
            WEIGHT_T* w_dgrad = W_dgrad;
        {%- endif %}

        {%- for i, segment in enumerate(forward_schedule.segments) %} {
            __syncwarp();
            {{ declare_smem_variables(segment, "smem") }}
            ROW_OPERATION({{segment.L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)
            WEIGHT_T* w_buffer;

            {{ load_ir_segments_force(segment.L1Map, "l1", "L1_smem", "j") }}
            {{ load_ir_segments_force(segment.L2Map, "l2", "L2_smem", "j") }}
            {% if not forward_schedule.stream_weights%}
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = w_dgrad[{{segment.weight_offset}} + j + lane_id];)
            {% endif %}
            w_buffer = w_dgrad;

            for(int n = 0; n < 3; n++) {
                if(n == 1) {
                    {% if not forward_schedule.stream_weights%}
                        ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = w[{{segment.weight_offset}} + j + lane_id];)
                    {% endif %}
                    {{ load_ir_segments_force(segment.L2Map, "l2_dgrad", "L2_smem", "j") }}
                    w_buffer = w;
                }
                else if(n == 2) {
                    {{ load_ir_segments_force(segment.L2Map, "l2", "L2_smem", "j") }}
                    {{ load_ir_segments_force(segment.L1Map, "l1_dgrad", "L1_smem", "j") }}
                }
                __syncwarp(); 
                forward_loop_unroll_{{i}}(L1_smem, L2_smem, w_buffer, weights_smem, L3_smem, scratch_smem, lane_id);
                __syncwarp();
            }

            {{ store_ir_segments(segment.L3Map, "l3", "L3_smem", "j") }}
        } {%- endfor %}
    }
}


{%- for i, segment in enumerate(double_backward_schedule.segments) %}
{{ generate_segment_kernel_backward(i, segment, double_backward_schedule.launch_config.warp_size, double_bwd=True) }}
{%- endfor %}

{% set schedule = double_backward_schedule %}
__global__ void 
{{ launch_bounds(double_backward_schedule) }}
double_backward_B(
        size_t num_products,
        IRREP_T* L1_in, IRREP_T* L2_in, WEIGHT_T* W, IRREP_T* L3_grad, // Inputs of backward op 
        IRREP_T* L1_dgrad, IRREP_T* L2_dgrad, IRREP_T* W_dgrad, // Gradients w.r.t outputs of backward op
        IRREP_T* L1_grad, IRREP_T* L2_grad, WEIGHT_T* W_grad, IRREP_T* L3_dgrad) {

    extern __shared__ char s[];
    {{ set_launch_bound_variables(schedule.launch_config) }}
    char* smem = s + {{schedule.memory_per_warp}} * warp_loc; 

    {%- set tpp = schedule.updated_config %}

    {%- for i, segment in enumerate(schedule.segments) %} {
        {{ declare_smem_variables(segment, "smem") }}
        for(size_t i = start; i < end; i++) {
            IRREP_T* l1_shft = L1_dgrad + i * {{schedule.L1.dim}} + lane_id;
            IRREP_T* l2_shft = L2_dgrad + i * {{schedule.L2.dim}} + lane_id; 
            IRREP_T* l3_shft = L3_grad + i * {{schedule.L3.dim}} + lane_id;

            IRREP_T* l1_original = L1_in + i * {{schedule.L1.dim}} + lane_id;
            IRREP_T* l2_original = L2_in + i * {{schedule.L2.dim}} + lane_id; 

            {%- if not tpp.shared_weights %} 
            WEIGHT_T* w = W + i * {{tpp.weight_numel}}; 
            WEIGHT_T* wgrad = W_grad + i * {{tpp.weight_numel}}; 
            WEIGHT_T* wdgrad = W_dgrad + i * {{tpp.weight_numel}}; 
            {%- else %}
            WEIGHT_T* w = W; 
            WEIGHT_T* wgrad = W_grad; 
            WEIGHT_T* wdgrad = W_dgrad; 
            {%- endif %}
            WEIGHT_T* weights_shft = w + lane_id;
            WEIGHT_T* weights_dgrad_shft = wdgrad + lane_id;


            {{ load_ir_segments(segment.L3Map, "l3_shft", "L3_grad_smem", "j") }}
            {{ load_ir_segments_force(segment.L1Map, "l1_shft", "L1_smem", "j") }}
            {{ load_ir_segments_force(segment.L2Map, "l2_shft", "L2_smem", "j") }}
            {{ load_ir_segments_force(segment.L2Map, "l2_original", "L2_dgrad_smem", "j") }}

            __syncwarp();
            {%- if not segment.L1Map.persist_load %}
                ROW_OPERATION({{segment.L1.dim}}, j, L1_grad_smem[j + lane_id] = 0.0f;)
            {%- endif %}

            {%- if not segment.L2Map.persist_load %}
                ROW_OPERATION({{segment.L2.dim}}, j, L2_grad_smem[j + lane_id] = 0.0f;)
            {%- endif %}

            {% if not schedule.stream_weights%}
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = weights_shft[{{segment.weight_offset}} + j];)
                ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_grad_smem[j + lane_id] = 0.0;)
            {% endif %}
            
            WEIGHT_T* w_buffer = w;
            IRREP_T* L2_buffer = L2_smem;
            IRREP_T* L2_dgrad_buffer = L2_dgrad_smem;

            for(int n = 0; n < 2; n++) {
                if(n == 1) {
                    {{ load_ir_segments_force(segment.L1Map, "l1_original", "L1_smem", "j") }}

                    {% if not schedule.stream_weights%}
                        ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = weights_dgrad_shft[{{segment.weight_offset}} + j];)
                    {% endif %}
                    w_buffer = wdgrad;
                    L2_buffer = L2_dgrad_smem;
                    L2_dgrad_buffer = L2_smem;
                }

                __syncwarp();
                double_backward_loop_unroll_{{i}}(L1_smem, L2_buffer, w_buffer, weights_smem, L3_grad_smem,
                        L1_grad_smem, L2_grad_smem, L2_dgrad_buffer, n, wgrad, weights_grad_smem, scratch_smem, lane_id);
                __syncwarp();
            }

            IRREP_T* l1_grad_shft = L1_grad + i * {{schedule.L1.dim}} + lane_id;
            IRREP_T* l2_grad_shft = L2_grad + i * {{schedule.L2.dim}} + lane_id;

            {%- if not tpp.shared_weights %}
                WEIGHT_T* weights_grad_shft = W_grad + i * {{schedule.updated_config.weight_numel}} + lane_id;
            {%- else %}
                WEIGHT_T* weights_grad_shft = W_grad + lane_id;
            {%- endif %}

            {{ store_ir_segments(segment.L1Map, "l1_grad_shft", "L1_grad_smem", "j") }}
            {{ store_ir_segments(segment.L2Map, "l2_grad_shft", "L2_grad_smem", "j") }}

            {% if not schedule.stream_weights%}
                {%- if not tpp.shared_weights %}
                    ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_grad_shft[{{segment.weight_offset}} + j] = weights_grad_smem[j + lane_id];)
                {%- else %}  
                    ROW_OPERATION({{segment.problem.weight_numel}}, j, atomicAdd(weights_grad_shft + {{segment.weight_offset}} + j, weights_grad_smem[j + lane_id]);)
                {%- endif %}
            {% endif %}
        }
    } {%- endfor %}
}