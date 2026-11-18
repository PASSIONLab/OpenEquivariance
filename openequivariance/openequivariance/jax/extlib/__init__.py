import ctypes
import importlib
import importlib.util
import re

import jax
import jax.extend

FFI_ABI_VERSION = 2
FFI_STAGES = ("instantiate", "prepare", "initialize", "execute")


class Handler(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("instantiate", ctypes.c_void_p),
        ("prepare", ctypes.c_void_p),
        ("initialize", ctypes.c_void_p),
        ("execute", ctypes.c_void_p),
        ("traits", ctypes.c_uint32),
    ]


class HandlerTable(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("handler_count", ctypes.c_uint32),
        ("handlers", ctypes.POINTER(Handler)),
    ]


def jax_backend():
    platform_version = jax.extend.backend.get_backend().platform_version
    match = re.search(r"(cuda|rocm)\s+(\d+)", platform_version)
    assert match is not None
    platform, version = match.groups()
    if platform == "rocm":
        return "hip"
    return f"cuda{int(version) // 1000}"


def load_library():
    backend = jax_backend()
    spec = importlib.util.find_spec(f"{__name__}.liboeq_jax_{backend}")
    if spec is None:
        raise ImportError(
            f"No OpenEquivariance JAX extension liboeq_jax_{backend}.so for "
            f"this JAX installation."
        )
    return backend, spec.origin, ctypes.CDLL(spec.origin)


def registrations(library):
    table_fn = library.oeq_ffi_handler_table
    table_fn.argtypes = []
    table_fn.restype = ctypes.POINTER(HandlerTable)
    table = table_fn().contents
    assert table.abi_version == FFI_ABI_VERSION

    targets = {}
    for index in range(table.handler_count):
        handler = table.handlers[index]
        targets[handler.name.decode()] = {
            stage: jax.ffi.pycapsule(getattr(handler, stage))
            for stage in FFI_STAGES
            if getattr(handler, stage)
        }
    return targets


BACKEND, LIBRARY_PATH, library = load_library()
IS_HIP = BACKEND == "hip"
DEVICE_TYPE = "cuda"

platform = "ROCM" if IS_HIP else "CUDA"
for name, target in registrations(library).items():
    jax.ffi.register_ffi_target(name, target, platform=platform, api_version=1)

ext_module_utilities = importlib.import_module(
    f"openequivariance.core.oeq_utilities_{BACKEND}"
)
DeviceProp = ext_module_utilities.DeviceProp
GPUTimer = ext_module_utilities.GPUTimer

__all__ = [
    "BACKEND",
    "GPUTimer",
    "DeviceProp",
    "DEVICE_TYPE",
]
