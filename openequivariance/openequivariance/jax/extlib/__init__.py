import jax
import openequivariance_extjax as oeq_extjax


def postprocess_kernel(kernel):
    if oeq_extjax.is_hip():
        kernel = kernel.replace("__syncwarp();", "__threadfence_block();")
        kernel = kernel.replace("__shfl_down_sync(FULL_MASK,", "__shfl_down(")
        kernel = kernel.replace("atomicAdd", "unsafeAtomicAdd")
        return kernel
    else:
        return kernel


platform = "CUDA"
if oeq_extjax.is_hip():
    platform = "ROCM"

for name, target in oeq_extjax.registrations().items():
    jax.ffi.register_ffi_target(name, target, platform=platform)

GPUTimer = oeq_extjax.GPUTimer
DeviceProp = oeq_extjax.DeviceProp

__all__ = [
    "GPUTimer",
    "DeviceProp",
]
