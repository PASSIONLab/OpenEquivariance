#pragma once

template<typename T>
class GroupMMHIP {
    int num_W;
    int batch_size;

    T alpha;
    T beta;

public:
    GroupMMHIP(int num_W, int batch_size) : 
            num_W(num_W),
            batch_size(batch_size),
            alpha(1.0),
            beta(0.0) { 
        // TODO: To implement.
    }

    void group_gemm(void* A_raw, void* B_raw, void* C_raw, 
            int64_t* ragged_counts, int m, int k, int ragged_inner) {
        // TODO: To implement.
    }

    void group_gemm_intptr(uint64_t weights, 
            uint64_t vectors, uint64_t output, 
            uint64_t ragged_counts, int m, int k, int ragged_inner) {    
        // TODO: To implement.
    }

    ~GroupMMHIP() {
        // TODO: To implement.
    }
};