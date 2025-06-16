{%- macro generate_matmul(name, M, N, K, TILES_PER_ROW, OUTPUT_RMAJOR, warp_size, A_CMAJOR=True, B_RMAJOR=True, accum=True) %}

{%-set TILES_PER_COL = warp_size // TILES_PER_ROW %}

template<typename T> 
__device__ __forceinline__ void {{name}}(const T* __restrict__ A, const T* __restrict__ B, T* C) {    
    int lane_id = threadIdx.x % {{warp_size}};

    int const rpt = {{(M + TILES_PER_COL - 1) // TILES_PER_COL}};
    int const cpt = {{(N + TILES_PER_ROW - 1) // TILES_PER_ROW}};

    T row[cpt];
    T col[rpt];
    T tile[rpt][cpt];

    int TI_idx = lane_id / {{TILES_PER_ROW}}; 
    int TJ_idx = lane_id % {{TILES_PER_ROW}};
    int is = TI_idx * rpt; int ie = (TI_idx + 1) * rpt;
    int js = TJ_idx * cpt; int je = (TJ_idx + 1) * cpt;
    int ist = min(is, {{M}}); int iet = min(ie, {{M}});
    int jst = min(js, {{N}}); int jet = min(je, {{N}});

    // Zero the output tile
    #pragma unroll
    for(int i = 0; i < rpt; i++) {
        #pragma unroll
        for(int j = 0; j < cpt; j++) {
            tile[i][j] = 0.0f;
        }
    }

    for(int k = 0; k < {{K}}; k++) {
        #pragma unroll
        for(int i = 0; i < rpt; i++) {
            if(ist + i < {{M}}) {
                {%- if A_CMAJOR %}
                    col[i] = A[k * {{M}} + ist + i];
                {%- else %}
                    col[i] = A[(ist + i) * {{K}} + k];
                {%- endif %}
            }
        }

        #pragma unroll
        for(int j = 0; j < cpt; j++) {
            if(jst + j < {{N}}) {
                {%- if B_RMAJOR %}
                    row[j] = B[k * {{N}} + jst + j];
                {%- else %}
                    row[j] = B[j * {{K}} + k];
                {%- endif %}
            }
        }

        #pragma unroll
        for(int i = 0; i < rpt; i++) {
            #pragma unroll
            for(int j = 0; j < cpt; j++) {
                if(ist + i < {{M}} && jst + j < {{N}}) {
                    tile[i][j] += col[i] * row[j];
                }
            }
        }
    }

    {%- if accum %}
    {%- set op = "+=" %}
    {%- else %}
    {%- set op = "=" %}
    {%- endif %}

    // Store the output
    #pragma unroll
    for(int i = 0; i < rpt; i++) {
        for(int j = 0; j < cpt; j++) {
            if(i + ist < {{M}} && j + jst < {{N}}) {
                {%- if OUTPUT_RMAJOR %}
                    C[(i + ist) * {{N}} + j + jst] {{op}} tile[i][j];
                {%- else %}
                    C[(j + jst) * {{M}} + i + ist] {{op}} tile[i][j];
                {%- endif %}
            }
        } 
    }
}

{%- endmacro %}