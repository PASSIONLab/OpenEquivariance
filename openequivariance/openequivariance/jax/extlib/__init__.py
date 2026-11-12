import importlib
import importlib.metadata
import os
import pkgutil

import jax

JAX_PLUGINS = {
    "jax-cuda13-plugin": "cuda13",
    "jax-cuda12-plugin": "cuda12",
    "jax-rocm7-plugin": "hip",
}


def _jax_backend():
    override = os.environ.get("OEQ_JAX_BACKEND")
    if override:
        return override
    for dist, backend in JAX_PLUGINS.items():
        try:
            importlib.metadata.version(dist)
            return backend
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _load_extension():
    built = sorted(
        module.name.removeprefix("oeq_jax_")
        for module in pkgutil.iter_modules(__path__)
        if module.name.startswith("oeq_jax_")
    )
    backend = _jax_backend()
    if backend not in built:
        if len(built) != 1:
            raise ImportError(
                f"No OpenEquivariance JAX extension for JAX backend {backend!r} "
                f"(built: {built}). Install a JAX GPU plugin, e.g. jax[cuda12], "
                f"or set OEQ_JAX_BACKEND."
            )
        backend = built[0]
    if backend == "hip":
        jax.devices()
    return backend, importlib.import_module(f"{__name__}.oeq_jax_{backend}")


BACKEND, extension_module = _load_extension()
IS_HIP = extension_module.is_hip()

DEVICE_TYPE = "cuda"

platform = "ROCM" if IS_HIP else "CUDA"
for name, target in extension_module.registrations().items():
    jax.ffi.register_ffi_target(name, target, platform=platform, api_version=1)

GPUTimer = extension_module.GPUTimer
DeviceProp = extension_module.DeviceProp

__all__ = [
    "BACKEND",
    "GPUTimer",
    "DeviceProp",
    "DEVICE_TYPE",
]
