#define USE_CUDA

#include <cstdint>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

extern "C" {
#ifdef SYCL_BACKEND
    AOTITorchError aoti_torch_get_current_sycl_queue(void** ret_queue) {
        return AOTI_TORCH_FAILURE;
    }
#else
    AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream) {
        return AOTI_TORCH_FAILURE;
    }
#endif
}
