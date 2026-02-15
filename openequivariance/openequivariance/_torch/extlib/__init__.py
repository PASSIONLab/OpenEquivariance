# ruff: noqa : F401, E402
import sys
import os
import warnings
import sysconfig
from pathlib import Path

import torch

from openequivariance.benchmark.logging_utils import getLogger

oeq_root = str(Path(__file__).parent.parent.parent)

BUILT_EXTENSION = False
BUILT_EXTENSION_ERROR = None

LINKED_LIBPYTHON = False
LINKED_LIBPYTHON_ERROR = None

extension_module = None

assert torch.version.cuda or torch.version.hip, "Only CUDA and HIP backends are supported"
def postprocess_kernel(kernel):
    if torch.version.hip:
        kernel = kernel.replace("__syncwarp();", "__threadfence_block();")
        kernel = kernel.replace("__shfl_down_sync(FULL_MASK,", "__shfl_down(")
        kernel = kernel.replace("atomicAdd", "unsafeAtomicAdd")
    return kernel

# Locate libpython (required for AOTI)
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

try:
    from torch.utils.cpp_extension import library_paths, include_paths

    extra_cflags = ["-O3"]
    torch_sources = ["libtorch_tp_jit.cpp", "json11/json11.cpp"]

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
        extra_cflags.append("-DHIP_BACKEND")

    torch_sources = [oeq_root + "/extension/" + src for src in torch_sources]
    include_dirs = [
        oeq_root + "/extension/" + d for d in include_dirs
    ] + include_paths("cuda")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            extension_module = torch.utils.cpp_extension.load(
                "libtorch_tp_jit",
                torch_sources,
                extra_cflags=extra_cflags,
                extra_include_paths=include_dirs,
                extra_ldflags=extra_link_args,
            )
            torch.ops.load_library(extension_module.__file__)
            BUILT_EXTENSION = True
        except Exception as e:
            # If compiling torch fails (e.g. low gcc version), we should fall back to the
            # version that takes integer pointers as args (but is untraceable to PyTorch JIT / export).
            BUILT_EXTENSION_ERROR = e
except Exception as e:
    BUILT_EXTENSION_ERROR = f"Error JIT-compiling OpenEquivariance Extension: {e}"


def torch_ext_so_path():
    return extension_module.__file__

sys.modules["oeq_utilities"] = extension_module

if BUILT_EXTENSION:
    from oeq_utilities import (
        #GroupMM_F32,
        #GroupMM_F64,
        DeviceProp,
        GPUTimer,
    )
else:
    def _raise_import_error_helper(import_target: str):
        if not BUILT_EXTENSION:
            raise ImportError(f"Could not import {import_target}: {BUILT_EXTENSION_ERROR}")

    def GroupMM_F32(*args, **kwargs):
        _raise_import_error_helper("GroupMM_F32")

    def GroupMM_F64(*args, **kwargs):
        _raise_import_error_helper("GroupMM_F64")

    def DeviceProp(*args, **kwargs):
        _raise_import_error_helper("DeviceProp")

    def GPUTimer(*args, **kwargs):
        _raise_import_error_helper("GPUTimer")
