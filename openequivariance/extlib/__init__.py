# ruff: noqa : F401, E402
import sys
import os
import warnings
import sysconfig
from functools import lru_cache
from pathlib import Path

import torch

from openequivariance.benchmark.logging_utils import getLogger

oeq_root = str(Path(__file__).parent.parent)

BUILT_EXTENSION = False
BUILT_EXTENSION_ERROR = None

TORCH_COMPILE = False
TORCH_COMPILE_ERROR = None

LINKED_LIBPYTHON = False
LINKED_LIBPYTHON_ERROR = None

torch_module, generic_module = None, None
postprocess_kernel = lambda kernel: kernel  # noqa : E731


@lru_cache(maxsize=1)
def _compile_torch_cuda_extension():
    return torch.version.cuda and ("CUDA_HOME" in os.environ)


@lru_cache(maxsize=1)
def _compile_torch_hip_extension():
    return torch.version.hip and ("HIP_HOME" in os.environ)


try:
    python_lib_dir = sysconfig.get_config_var("LIBDIR")
    major, minor = sys.version_info.major, sys.version_info.minor
    python_lib_name = f"python{major}.{minor}"

    libpython_so = os.path.join(python_lib_dir, f"lib{python_lib_name}.so")
    libpython_a = os.path.join(python_lib_dir, f"lib{python_lib_name}.a")
    if not (os.path.exists(libpython_so) or os.path.exists(libpython_a)):
        raise FileNotFoundError(
            f"libpython not found, tried {libpython_so} and {libpython_a}"
        )

    LINKED_LIBPYTHON = True
except Exception as e:
    LINKED_LIBPYTHON_ERROR = f"Error linking libpython:\n{e}\nSysconfig variables:\n{sysconfig.get_config_vars()}"


if BUILT_EXTENSION:
    import openequivariance.extlib.generic_module

    generic_module = openequivariance.extlib.generic_module
elif _compile_torch_cuda_extension() or _compile_torch_hip_extension():
    try:
        from torch.utils.cpp_extension import library_paths, include_paths

        extra_cflags = ["-O3"]
        generic_sources = ["generic_module.cpp"]
        torch_sources = ["libtorch_tp_jit.cpp"]

        include_dirs, extra_link_args = (["util"], ["-Wl,--no-as-needed"])

        if LINKED_LIBPYTHON:
            extra_link_args.pop()
            extra_link_args.extend(
                [
                    f"-Wl,--no-as-needed,-rpath,{python_lib_dir}",
                    f"-L{python_lib_dir}",
                    f"-l{python_lib_name}",
                ],
            )
        if torch.version.cuda:
            extra_link_args.extend(["-lcuda", "-lcudart", "-lnvrtc"])

            try:
                torch_libs, cuda_libs = library_paths("cuda")
                extra_link_args.append("-Wl,-rpath," + torch_libs)
                extra_link_args.append("-L" + cuda_libs)
                if os.path.exists(cuda_libs + "/stubs"):
                    extra_link_args.append("-L" + cuda_libs + "/stubs")
            except Exception as e:
                getLogger().info(str(e))

            extra_cflags.append("-DCUDA_BACKEND")
        elif torch.version.hip:
            extra_link_args.extend(["-lhiprtc"])
            torch_libs = library_paths("cuda")[0]
            extra_link_args.append("-Wl,-rpath," + torch_libs)

            def postprocess(kernel):
                kernel = kernel.replace("__syncwarp();", "__threadfence_block();")
                kernel = kernel.replace("__shfl_down_sync(FULL_MASK,", "__shfl_down(")
                kernel = kernel.replace("atomicAdd", "unsafeAtomicAdd")
                return kernel

            postprocess_kernel = postprocess

            extra_cflags.append("-DHIP_BACKEND")

        generic_sources = [oeq_root + "/extension/" + src for src in generic_sources]
        torch_sources = [oeq_root + "/extension/" + src for src in torch_sources]
        include_dirs = [
            oeq_root + "/extension/" + d for d in include_dirs
        ] + include_paths("cuda")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                torch_module = torch.utils.cpp_extension.load(
                    "libtorch_tp_jit",
                    torch_sources,
                    extra_cflags=extra_cflags,
                    extra_include_paths=include_dirs,
                    extra_ldflags=extra_link_args,
                )
                torch.ops.load_library(torch_module.__file__)
                TORCH_COMPILE = True
            except Exception as e:
                # If compiling torch fails (e.g. low gcc version), we should fall back to the
                # version that takes integer pointers as args (but is untraceable to PyTorch JIT / export).
                TORCH_COMPILE_ERROR = e

            generic_module = torch.utils.cpp_extension.load(
                "generic_module",
                generic_sources,
                extra_cflags=extra_cflags,
                extra_include_paths=include_dirs,
                extra_ldflags=extra_link_args,
            )
            if "generic_module" not in sys.modules:
                sys.modules["generic_module"] = generic_module

        if not TORCH_COMPILE:
            warnings.warn(
                "Could not compile integrated PyTorch wrapper. Falling back to Pybind11"
                + f", but JITScript, compile fullgraph, and export will fail.\n {TORCH_COMPILE_ERROR}"
            )
        BUILT_EXTENSION = True
    except Exception as e:
        BUILT_EXTENSION_ERROR = f"Error building OpenEquivariance Extension: {e}"


def _raise_import_error_helper(import_target: str):
    if not BUILT_EXTENSION:
        raise ImportError(f"Could not import {import_target}: {BUILT_EXTENSION_ERROR}")


if BUILT_EXTENSION:
    from generic_module import (
        JITTPImpl,
        JITConvImpl,
        GroupMM_F32,
        GroupMM_F64,
        DeviceProp,
        DeviceBuffer,
        GPUTimer,
    )
else:

    def JITTPImpl(*args, **kwargs):
        _raise_import_error_helper("JITTPImpl")

    def JITConvImpl(*args, **kwargs):
        _raise_import_error_helper("JITConvImpl")

    def GroupMM_F32(*args, **kwargs):
        _raise_import_error_helper("GroupMM_F32")

    def GroupMM_F64(*args, **kwargs):
        _raise_import_error_helper("GroupMM_F64")

    def DeviceProp(*args, **kwargs):
        _raise_import_error_helper("DeviceProp")

    def DeviceBuffer(*args, **kwargs):
        _raise_import_error_helper("DeviceBuffer")

    def GPUTimer(*args, **kwargs):
        _raise_import_error_helper("GPUTimer")
