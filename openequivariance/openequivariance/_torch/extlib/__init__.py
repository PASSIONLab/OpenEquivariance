# ruff: noqa : F401, E402
import sys
import os
import time
import warnings
import sysconfig
import contextlib
import importlib
from pathlib import Path
from packaging.version import Version

import torch


oeq_root = str(Path(__file__).parent.parent.parent)

BUILT_EXTENSION = False
BUILT_EXTENSION_ERROR = None

LINKED_LIBPYTHON = False
LINKED_LIBPYTHON_ERROR = None

extension_module = None


def _detect_backend():
    """HIP builds report a ``torch.version.cuda`` of None, so HIP is tested first."""
    if torch.version.hip:
        return "hip"
    if torch.version.cuda:
        return "cuda"
    if getattr(torch.version, "xpu", None):
        return "sycl"
    return None


BACKEND = _detect_backend()

assert BACKEND is not None, (
    "Only the CUDA, HIP and XPU (SYCL) backends are supported. "
    "No supported accelerator was detected in this PyTorch build."
)

IS_HIP = BACKEND == "hip"
IS_SYCL = BACKEND == "sycl"

if IS_SYCL and Version(torch.__version__) < Version("2.10"):
    raise RuntimeError(
        f"The SYCL backend requires PyTorch >= 2.10, found {torch.__version__}."
    )

DEVICE_TYPE = "xpu" if IS_SYCL else "cuda"

# CUDA extensions are built per CUDA major version, matching torch's libcudart/libnvrtc.
if BACKEND == "cuda":
    EXTENSION_NAME = f"oeq_stable_cuda{torch.version.cuda.split('.')[0]}"
else:
    EXTENSION_NAME = f"oeq_stable_{BACKEND}"


@contextlib.contextmanager
def _maybe_set_max_jobs_env(limit=8):
    if "MAX_JOBS" in os.environ:
        yield
        return
    os.environ["MAX_JOBS"] = str(min(limit, os.cpu_count() or 1))
    try:
        yield
    finally:
        os.environ.pop("MAX_JOBS", None)


def _wait_for_torch_build_lock(name, timeout=300):
    from torch.utils.cpp_extension import _get_build_directory

    lock_path = os.path.join(_get_build_directory(name, False), "lock")
    start = time.time()
    while os.path.exists(lock_path):
        if time.time() - start > timeout:
            raise RuntimeError(
                f"Timed out after {timeout} seconds waiting for another process "
                f"to finish building the OpenEquivariance extension '{name}'. "
                f"If no other build is running, a previous build was likely "
                f"killed partway; delete {lock_path} and retry."
            )
        time.sleep(1)


def load_jit_extension():
    """Compile the SYCL extension at first import. CUDA and HIP use only the
    precompiled stable extension."""
    global \
        BUILT_EXTENSION, \
        BUILT_EXTENSION_ERROR, \
        LINKED_LIBPYTHON, \
        LINKED_LIBPYTHON_ERROR, \
        extension_module

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

        include_dirs, extra_link_args = (["backend"], ["-Wl,--no-as-needed"])
        extra_include_dirs = []

        try:
            import pybind11

            extra_include_dirs.append(pybind11.get_include())
        except Exception as e:
            BUILT_EXTENSION_ERROR = (
                "Could not locate pybind11 include path required for JIT "
                f"OpenEquivariance extension compilation: {e}"
            )
            return

        if LINKED_LIBPYTHON:
            extra_link_args.pop()
            extra_link_args.extend(
                [
                    f"-Wl,--no-as-needed,-rpath,{python_lib_dir}",
                    f"-L{python_lib_dir}",
                    f"-l{python_lib_name}",
                ],
            )
        import shutil

        cxx = os.environ.get("CXX", "")
        if "icpx" not in os.path.basename(cxx):
            if shutil.which("icpx") is None:
                BUILT_EXTENSION_ERROR = (
                    "The SYCL backend requires the oneAPI DPC++ compiler. "
                    "Put 'icpx' on your PATH or set CXX to it."
                )
                return
            os.environ["CXX"] = "icpx"

        extra_cflags.extend(["-fsycl", "-DSYCL_BACKEND"])
        extra_link_args.extend(["-fsycl", "-lsycl", "-ltorch_xpu", "-lc10_xpu"])

        for lib_dir in library_paths("xpu"):
            extra_link_args.append("-Wl,-rpath," + lib_dir)
            extra_link_args.append("-L" + lib_dir)

        torch_sources = [oeq_root + "/extension/" + src for src in torch_sources]
        include_dirs = (
            [oeq_root + "/extension/" + d for d in include_dirs]
            + extra_include_dirs
            + include_paths("xpu")
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                _wait_for_torch_build_lock("libtorch_tp_jit")
                with _maybe_set_max_jobs_env():
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


def load_precompiled_extension():
    global BUILT_EXTENSION, BUILT_EXTENSION_ERROR, LINKED_LIBPYTHON, extension_module
    LINKED_LIBPYTHON = (
        True  # Doesn't actually use libpython, just set this as true anyway
    )
    try:
        extension_module = importlib.import_module(
            f"openequivariance._torch.extlib.{EXTENSION_NAME}"
        )

        torch.ops.load_library(extension_module.__file__)
        BUILT_EXTENSION = True
    except Exception as e:
        BUILT_EXTENSION_ERROR = (
            f"Error loading precompiled OpenEquivariance Extension: {e}"
        )


AOTI_SO_NAME = f"lib{EXTENSION_NAME}_aoti.so"

# CUDA and HIP load only the precompiled stable extension, from a wheel or a
# source build; SYCL is still compiled at first import.
USE_PRECOMPILED_EXTENSION = BACKEND != "sycl"

_SOURCE_BUILD_HINT = (
    "To build the extension for your setup from source, run:\n"
    "    pip install --no-binary openequivariance openequivariance"
)

if not USE_PRECOMPILED_EXTENSION:
    load_jit_extension()
elif Version(torch.__version__) < Version("2.10"):
    BUILT_EXTENSION_ERROR = (
        f"OpenEquivariance requires PyTorch >= 2.10, found {torch.__version__}."
    )
elif not os.path.exists(os.path.join(os.path.dirname(__file__), AOTI_SO_NAME)):
    BUILT_EXTENSION_ERROR = (
        f"No precompiled OpenEquivariance extension {EXTENSION_NAME} for PyTorch "
        f"{torch.__version__}. {_SOURCE_BUILD_HINT}"
    )
else:
    load_precompiled_extension()

if BUILT_EXTENSION_ERROR is not None:
    warnings.warn(str(BUILT_EXTENSION_ERROR), stacklevel=3)


def torch_ext_so_path():
    if not USE_PRECOMPILED_EXTENSION:
        return extension_module.__file__
    else:
        dirname = os.path.dirname(extension_module.__file__)
        return os.path.join(dirname, AOTI_SO_NAME)


sys.modules["oeq_utilities"] = extension_module

if BUILT_EXTENSION:
    from oeq_utilities import (
        DeviceProp,
        GPUTimer,
    )
else:

    def _raise_import_error_helper(import_target: str):
        if not BUILT_EXTENSION:
            raise ImportError(
                f"Could not import {import_target}: {BUILT_EXTENSION_ERROR}"
            )

    def DeviceProp(*args, **kwargs):
        _raise_import_error_helper("DeviceProp")

    def GPUTimer(*args, **kwargs):
        _raise_import_error_helper("GPUTimer")
