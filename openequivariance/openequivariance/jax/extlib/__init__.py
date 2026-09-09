import jax
import openequivariance_extjax as oeq_extjax

IS_HIP = oeq_extjax.is_hip()

platform = "CUDA"
if IS_HIP:
    platform = "ROCM"

for name, target in oeq_extjax.registrations().items():
    jax.ffi.register_ffi_target(name, target, platform=platform, api_version=1)

GPUTimer = oeq_extjax.GPUTimer
DeviceProp = oeq_extjax.DeviceProp

__all__ = [
    "GPUTimer",
    "DeviceProp",
]
