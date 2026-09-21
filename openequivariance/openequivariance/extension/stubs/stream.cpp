#include <cstdint>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

extern "C" {
#ifdef SYCL_BACKEND
    AOTITorchError aoti_torch_get_current_sycl_queue(void** ret_queue) {
        *ret_queue = nullptr;
        return 0;
    }
#else
    AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream) {
        return 0;
    }
#endif
}
