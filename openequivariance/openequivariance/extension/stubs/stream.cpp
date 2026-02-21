#include <torch/csrc/inductor/aoti_torch/c/shim.h>

AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream) {
    return 0;
}