import jax
import openequivariance_extjax as oeq_extjax


#def postprocess_kernel(kernel):
#    """
#    Only CUDA for now, so no postprocessing.
#    """
#    return kernel

def postprocess_kernel(kernel):
    kernel = kernel.replace("__syncwarp();", "__threadfence_block();")
    kernel = kernel.replace("__shfl_down_sync(FULL_MASK,", "__shfl_down(")
    kernel = kernel.replace("atomicAdd", "unsafeAtomicAdd")
    return kernel

for name, target in oeq_extjax.registrations().items():
    jax.ffi.register_ffi_target(name, target, platform="ROCM")

GPUTimer = oeq_extjax.GPUTimer
DeviceProp = oeq_extjax.DeviceProp

__all__ = [
    "GPUTimer",
    "DeviceProp",
]
