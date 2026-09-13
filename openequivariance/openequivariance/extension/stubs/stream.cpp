#define USE_CUDA

#include <cstdint>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>

extern "C" {
    AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream) {
        return AOTI_TORCH_FAILURE;
    }

    AOTITorchError aoti_torch_cuda_bmm_out(
            AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2) {
        return AOTI_TORCH_FAILURE;
    }
}
