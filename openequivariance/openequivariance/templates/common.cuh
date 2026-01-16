#define ROW_OPERATION(ROW_LEN, LOOP_VAR, ...) \
    _Pragma ("unroll") \
    for(int LOOP_VAR = 0; LOOP_VAR < ROW_LEN; LOOP_VAR += THREADS_PER_WARP) { \
        if(LOOP_VAR >= ROW_LEN - THREADS_PER_WARP) { \
            if(lane_id < ROW_LEN - LOOP_VAR) { \
                __VA_ARGS__  \
            } \
        } \
        else { \
            __VA_ARGS__  \
        } \
    }
