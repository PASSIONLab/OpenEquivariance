import jax
import hashlib

def postprocess_kernel(kernel):
    '''
    Only CUDA for now, so no postprocessing.
    '''
    return kernel

import openequivariance_extjax as oeq_extjax 
for name, target in oeq_extjax.registrations().items():
    jax.ffi.register_ffi_target(name, target, platform="CUDA")

GPUTimer = oeq_extjax.GPUTimer
DeviceProp = oeq_extjax.DeviceProp

__all__ = [
    "GPUTimer",
    "DeviceProp",
]