#include <cstdint>
#include <cstdio>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

// The stable extension is linked on a machine with CPU-only libtorch, where
// no libtorch_cuda/libtorch_hip exists. This file builds a stand-in library
// with the real one's SONAME so the link succeeds; at runtime the DT_NEEDED
// entry is satisfied by the real library, already loaded via `import torch`,
// and this file is never opened. It is not shipped in the wheel.
//
// These bodies can therefore only execute in a misconfigured process (e.g.
// the extension loaded without torch). Fail through the shim's error-code
// contract - TORCH_ERROR_CODE_CHECK at the call site turns this into a
// catchable exception - rather than returning success with garbage outputs.
namespace {

AOTITorchError oeq_stub_called(const char *name) {
    fprintf(stderr,
            "OpenEquivariance: link-time stub '%s' executed; the real "
            "libtorch_cuda/libtorch_hip is not loaded (import torch before "
            "loading the OpenEquivariance extension).\n",
            name);
    return AOTI_TORCH_FAILURE;
}

} // namespace

extern "C" {
    AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream) {
        if (ret_stream) *ret_stream = nullptr;
        return oeq_stub_called("aoti_torch_get_current_cuda_stream");
    }

    AOTITorchError aoti_torch_create_device_guard(int32_t device_index, DeviceGuardHandle* ret_guard) {
        if (ret_guard) *ret_guard = nullptr;
        return oeq_stub_called("aoti_torch_create_device_guard");
    }

    AOTITorchError aoti_torch_delete_device_guard(DeviceGuardHandle guard) {
        return oeq_stub_called("aoti_torch_delete_device_guard");
    }

    AOTITorchError aoti_torch_device_guard_set_index(DeviceGuardHandle guard, int32_t device_index) {
        return oeq_stub_called("aoti_torch_device_guard_set_index");
    }

    AOTITorchError torch_get_current_cuda_blas_handle(void** ret_handle) {
        if (ret_handle) *ret_handle = nullptr;
        return oeq_stub_called("torch_get_current_cuda_blas_handle");
    }
}
