#include <cstdint>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

/*
* Weak stand-ins for the stream accessors the AOTI shim declares. The real
* symbols come from libtorch_cuda / libtorch_xpu at load time; these exist so
* the extension links without a hard dependency on the accelerator runtime.
*/
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
