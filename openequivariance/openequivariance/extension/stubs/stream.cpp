#include <cstdint>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

// Link-time stand-ins for symbols that live in the real libtorch_cuda /
// libtorch_hip, which is already loaded at runtime and wins symbol
// resolution. These bodies never execute in production.
extern "C" {
    AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream) {
        return 0;
    }

    AOTITorchError aoti_torch_create_device_guard(int32_t device_index, DeviceGuardHandle* ret_guard) {
        return 0;
    }

    AOTITorchError aoti_torch_delete_device_guard(DeviceGuardHandle guard) {
        return 0;
    }

    AOTITorchError aoti_torch_device_guard_set_index(DeviceGuardHandle guard, int32_t device_index) {
        return 0;
    }

    AOTITorchError torch_get_current_cuda_blas_handle(void** ret_handle) {
        return 0;
    }
}