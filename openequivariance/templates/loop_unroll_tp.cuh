{%- from 'macros.jinja' import transpose_load, transpose_store, reg_store with context %}
{%- from 'wmm.cuh' import generate_matmul %}

{%- macro generate_segment_kernel_forward(id, segment) %}
{%- set L1, L2, L3, interactions, problem = segment.L1, segment.L2, segment.L3, segment.interactions, segment.problem %}

{%- set L1_irrep_lengths = L1 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L2_irrep_lengths = L2 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L3_irrep_lengths = L3 | map(attribute="ir") | map(attribute="dim") | list %}

{%- for i, inst in enumerate(problem.instructions) %}
    {%- set u, v, w, _ = interactions[i] %}
    {%- if inst.connection_mode == "uvw" %}
        {{generate_matmul("matmul_fwd_%d" % i, L3[w].mul, L3[w].ir.dim, L1[u].mul, 4, True)}}
        {{generate_matmul("matmul_bwd_A_%d" % i, L1[u].mul, L3[w].ir.dim, L3[w].mul, 4, True, A_CMAJOR=False, accum=False)}}
        {{generate_matmul("matmul_bwd_B_%d" % i, L3[w].mul, L1[u].mul, L3[w].ir.dim, 4, False, A_CMAJOR=False, accum=False)}}
    {%- endif %}
{%- endfor %}

__device__ __forceinline__ void forward_loop_unroll_{{id}}(IRREP_T* __restrict__ L1_smem, 
        IRREP_T* __restrict__ L2_smem, 
        WEIGHT_T* __restrict__ weights, 
        WEIGHT_T* __restrict__ weights_smem, 
        IRREP_T* __restrict__ L3_smem, 
        WEIGHT_T* __restrict__ scratch, 
        int lane_id) {
    IRREP_T l1_vec[{{L1_irrep_lengths | max}}];
    IRREP_T l2_vec[{{L2_irrep_lengths | max}}];
    IRREP_T l3_vec[{{L3_irrep_lengths | max}}];
    WEIGHT_T weight;
    int offset;

    {%- set num_interact = interactions | length %}
    
    {%- for k in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[k] %}
        {%- set weight_start, _, _ = problem.weight_range_and_shape_for_instruction(k)%}

        {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
            offset = {{ L1.slices()[u].start}}; 
            {{transpose_load(L1[u].mul, L1[u].ir.dim, 'L1_smem', 'offset', 'l1_vec')}}
        {%- endif %}

        #pragma unroll
        for(int j = 0; j < {{L3[w].ir.dim}}; j++)
            l3_vec[j] = 0.0f;

        for(int k = 0; k < {{L2[v].mul}}; k++) {
            {%- if problem.instructions[k].connection_mode == "uvu" %}
                if(lane_id < {{L1[u].mul}}) {
                    weight = weights_smem[{{weight_start}} + k * {{L1[u].mul}} + lane_id];
                }

                #pragma unroll
                for(int j = 0; j < {{L2[v].ir.dim}}; j++)
                    l2_vec[j] = L2_smem[j + {{L2.slices()[v].start}} + k * {{L2[v].ir.dim}}] * weight;
            {%- elif problem.instructions[k].connection_mode == "uvw" %}
                {# Stream weights here #}
                {%- set slice_size = L3[w].mul * L1[u].mul %}
                {
                    WEIGHT_T* tmp = weights + {{weight_start}} + k * {{slice_size}} + lane_id;
                    ROW_OPERATION({{slice_size}}, j, weights_smem[j + lane_id] = tmp[j];)
                }
                #pragma unroll
                for(int j = 0; j < {{L2[v].ir.dim}}; j++)
                    l2_vec[j] = L2_smem[j + {{L2.slices()[v].start}} + k * {{L2[v].ir.dim}}];
            {%- endif %}

            // ----------------- CORE CALCULATION -----------------
            {%- for i in range(tensor.nnz) %}
                {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                l3_vec[{{coord3}}] += {{value}} * l1_vec[{{coord1}}] * l2_vec[{{coord2}}]; 
            {%- endfor %}
            // ----------------- CORE CALCULATION -----------------

            {%- if problem.instructions[k].connection_mode == "uvw" %}
                {{transpose_store(L1[u].mul, L3[w].ir.dim, 'scratch', '0', 'l3_vec', '=', '1.0')}}
                __syncwarp();
                offset = {{ L3.slices()[w].start}}; 
                matmul_fwd_{{k}}(weights_smem, scratch, L3_smem + offset);
                __syncwarp();

                #pragma unroll
                for(int j = 0; j < {{L3[w].ir.dim}}; j++)
                    l3_vec[j] = 0.0f;
            {%- endif %}

            {%- if problem.instructions[k].connection_mode != "uvw" %}
                offset = {{ L3.slices()[w].start}}; 
                {{transpose_store(L3[w].mul, L3[w].ir.dim, 'L3_smem', 'offset', 'l3_vec', '+=', '1.0')}}

                {%- if L2[v].mul > 1%}
                #pragma unroll
                for(int j = 0; j < {{L3[w].ir.dim}}; j++)
                    l3_vec[j] = 0.0f;
                {%- endif %}
            {%- endif %}  
        }
    {%- endfor %}
}
{%- endmacro %}

{%- macro generate_segment_kernel_backward(id, segment, warp_size) %}
{%- set L1, L2, L3, interactions, problem = segment.L1, segment.L2, segment.L3, segment.interactions, segment.problem %}

{%- set L1_irrep_lengths = L1 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L2_irrep_lengths = L2 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L3_irrep_lengths = L3 | map(attribute="ir") | map(attribute="dim") | list %}

__device__ __forceinline__ void backward_loop_unroll_{{id}}(
        const IRREP_T* L1_smem,
        const IRREP_T* L2_smem,
        WEIGHT_T* weights,
        WEIGHT_T* weights_smem,
        const IRREP_T* L3_grad_smem,

        IRREP_T* L1_grad_smem,
        IRREP_T* L2_grad_smem,
        WEIGHT_T* weights_grad,
        WEIGHT_T* weights_grad_smem,
        WEIGHT_T* scratch,
        int lane_id) {

    IRREP_T l1_vec[{{L1_irrep_lengths  | max}}]; 
    IRREP_T l1_grad[{{L1_irrep_lengths | max}}]; 
    IRREP_T l2_vec[{{L2_irrep_lengths  | max}}];
    IRREP_T l2_grad[{{L2_irrep_lengths | max}}]; 
    IRREP_T l3_grad[{{L3_irrep_lengths | max}}];

    WEIGHT_T weight, weight_grad;
    int offset;

    {%- set num_scratch_reg = 1 %}
    IRREP_T scratch1[{{num_scratch_reg}}];
    IRREP_T scratch2[{{num_scratch_reg}}];

    {%- set num_interact = interactions | length %}

    {%- for k in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[k] %}
        {%- set weight_start, _, _ = problem.weight_range_and_shape_for_instruction(k)%}

        {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
            offset = {{ L1.slices()[u].start}};
            {{transpose_load(L1[u].mul, L1[u].ir.dim, 'L1_smem', 'offset', 'l1_vec')}}
            {{transpose_load(L1[u].mul, L1[u].ir.dim, 'L1_grad_smem', 'offset', 'l1_grad')}}
        {%- endif %}


        {%- if problem.instructions[k].connection_mode != "uvw" %}
            {%- if k == 0 or interactions[k][2] != interactions[k-1][2] %}
                offset = {{ L3.slices()[w].start}}; 
                {{transpose_load(L3[w].mul, L3[w].ir.dim, 'L3_grad_smem', 'offset', 'l3_grad')}}
            {%- endif %}
        {%- endif %}

        for(int k = 0; k < {{L2[v].mul}}; k++) {
            {%- if k == 0 or interactions[k][1] != interactions[k-1][1] or L2[v].mul > 1 or L1[u].mul != L1[interactions[k-1][0]].mul %}
                #pragma unroll
                for(int j = 0; j < {{L2[v].ir.dim}}; j++) {
                    l2_vec[j] = L2_smem[j + {{L2.slices()[v].start}} + k * {{L2[v].ir.dim}}]; 
                    l2_grad[j] = 0.0; 
                }
            {%- endif %}

            {%- if problem.instructions[k].connection_mode == "uvu" %}
                if(lane_id < {{L1[u].mul}}) {
                    weight = weights_smem[{{weight_start}} + k * {{L1[u].mul}} + lane_id];
                }
                weight_grad = 0.0;

                {%- for i in range(tensor.nnz) %} 
                    {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                    scratch1[{{i % num_scratch_reg}}] = l3_grad[{{coord3}}] * {{value}}; 
                    weight_grad += scratch1[{{i % num_scratch_reg}}] * l2_vec[{{coord2}}] * l1_vec[{{coord1}}];
                    scratch2[{{i % num_scratch_reg}}] = scratch1[{{i % num_scratch_reg}}] * weight;
                    l2_grad[{{coord2}}] += scratch2[{{i % num_scratch_reg}}] * l1_vec[{{coord1}}];
                    l1_grad[{{coord1}}] += scratch2[{{i % num_scratch_reg}}] * l2_vec[{{coord2}}];
                {%- endfor %}

            {%- elif problem.instructions[k].connection_mode == "uvw" %}
                {%- set slice_size = L3[w].mul * L1[u].mul %}
                {
                    WEIGHT_T* tmp = weights + {{weight_start}} + k * {{slice_size}} + lane_id;
                    ROW_OPERATION({{slice_size}}, j, weights_smem[j + lane_id] = tmp[j];)

                    __syncwarp();
                    offset = {{ L3.slices()[w].start}}; 
                    matmul_bwd_A_{{k}}(weights_smem, L3_grad_smem + offset, scratch);
                    __syncwarp();

                    {{transpose_load(L1[u].mul, L3[w].ir.dim, 'scratch', '0', 'l3_grad')}}

                    {%- for i in range(tensor.nnz) %} 
                        {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                        scratch1[{{i % num_scratch_reg}}] = l3_grad[{{coord3}}] * {{value}}; 
                        l2_grad[{{coord2}}] += scratch1[{{i % num_scratch_reg}}] * l1_vec[{{coord1}}];
                        l1_grad[{{coord1}}] += scratch1[{{i % num_scratch_reg}}] * l2_vec[{{coord2}}];
                    {%- endfor %}

                    #pragma unroll
                    for(int j = 0; j < {{L3[w].ir.dim}}; j++)
                        l3_grad[j] = 0.0;

                    {# Phase 2, computing weight grad. Reusing L3 grad here #}
                    {%- for i in range(tensor.nnz) %}
                        {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                        l3_grad[{{coord3}}] += {{value}} * l1_vec[{{coord1}}] * l2_vec[{{coord2}}]; 
                    {%- endfor %}

                    {{ reg_store(L1[u].mul, L3[w].ir.dim, "scratch", "0", "l3_grad", "=", 1.0) }}

                    __syncwarp(); 
                    matmul_bwd_B_{{k}}(L3_grad_smem + offset, scratch, weights_smem);
                    __syncwarp();

                    tmp = weights_grad + {{weight_start}} + k * {{slice_size}} + lane_id;
                    {%- if problem.shared_weights %}
                        ROW_OPERATION({{slice_size}}, j, atomicAdd(tmp + j, weights_smem[j + lane_id]);)
                    {%- else %}
                        ROW_OPERATION({{slice_size}}, j, tmp[j] = weights_smem[j + lane_id];)
                    {%- endif %}
                }
            {%- endif %}

            {%- if k == num_interact - 1 or interactions[k][1] != interactions[k+1][1] or L2[v].mul > 1 or L1[u].mul != L1[interactions[k+1][0]].mul %}
                #pragma unroll
                for(int j = 0; j < {{L2[v].ir.dim}}; j++) {
                    if(lane_id >= {{L1[u].mul}}) {
                        l2_grad[j] = 0.0;
                    }
                    #pragma unroll
                    for (int offset = {{ warp_size  // 2}}; offset > 0; offset /= 2) {
                        l2_grad[j] += __shfl_down_sync(FULL_MASK, l2_grad[j], offset);
                    } 
                }

                if(lane_id == 0) {
                    #pragma unroll 
                    for(int j = 0; j < {{L2[v].ir.dim}}; j++)
                        L2_grad_smem[j + {{L2.slices()[v].start}} + k * {{L2[v].ir.dim}}] += l2_grad[j];
                }
            {%- endif %}

            {%- if problem.instructions[k].connection_mode != "uvw" %}
                if(lane_id < {{L1[u].mul}}) {
                    weights_grad_smem[{{weight_start}} + k * {{L1[u].mul}} + lane_id] = weight_grad;
                }
            {%- endif %}
        }

        // Storeback
        {%- if k == num_interact - 1 or interactions[k][0] != interactions[k+1][0] %}
            offset = {{ L1.slices()[u].start}}; 
            {{transpose_store(L1[u].mul, L1[u].ir.dim, 'L1_grad_smem', 'offset', 'l1_grad', '=', '1.0')}}
        {%- endif %}

    {%- endfor %}
}
{%- endmacro %}