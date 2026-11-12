import ctypes
import importlib.metadata
import importlib.util
import os
import pkgutil

import jax

from openequivariance.core.gpu_utilities import load_utilities

JAX_PLUGINS = {
    "jax-cuda13-plugin": "cuda13",
    "jax-cuda12-plugin": "cuda12",
    "jax-rocm7-plugin": "hip",
}

FFI_ABI_VERSION = 2
FFI_STAGES = ("instantiate", "prepare", "initialize", "execute")


class _Handler(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("instantiate", ctypes.c_void_p),
        ("prepare", ctypes.c_void_p),
        ("initialize", ctypes.c_void_p),
        ("execute", ctypes.c_void_p),
        ("traits", ctypes.c_uint32),
    ]


class _HandlerTable(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("handler_count", ctypes.c_uint32),
        ("handlers", ctypes.POINTER(_Handler)),
    ]


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


def _load_library():
    built = sorted(
        module.name.removeprefix("liboeq_jax_")
        for module in pkgutil.iter_modules(__path__)
        if module.name.startswith("liboeq_jax_")
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
    path = importlib.util.find_spec(f"{__name__}.liboeq_jax_{backend}").origin
    return backend, path, ctypes.CDLL(path)


BACKEND, LIBRARY_PATH, library = _load_library()
IS_HIP = BACKEND == "hip"

library.oeq_ffi_handler_table.argtypes = []
library.oeq_ffi_handler_table.restype = ctypes.POINTER(_HandlerTable)


def handler_table():
    table = library.oeq_ffi_handler_table().contents
    if table.abi_version != FFI_ABI_VERSION:
        raise ImportError(
            f"OpenEquivariance FFI handler table has ABI version "
            f"{table.abi_version}, expected {FFI_ABI_VERSION}"
        )
    return table


def registrations():
    table = handler_table()
    targets = {}
    for index in range(table.handler_count):
        handler = table.handlers[index]
        targets[handler.name.decode()] = {
            stage: jax.ffi.pycapsule(getattr(handler, stage))
            for stage in FFI_STAGES
            if getattr(handler, stage)
        }
    return targets


DEVICE_TYPE = "cuda"

platform = "ROCM" if IS_HIP else "CUDA"
for name, target in registrations().items():
    jax.ffi.register_ffi_target(name, target, platform=platform, api_version=1)

DeviceProp, GPUTimer = load_utilities(BACKEND)

__all__ = [
    "BACKEND",
    "GPUTimer",
    "DeviceProp",
    "DEVICE_TYPE",
]
