# ruff: noqa : F401, E402
import sys
import os
import time
import warnings
import sysconfig
import contextlib
from pathlib import Path
from packaging.version import Version

import torch

from openequivariance.core.logging import getLogger

oeq_root = str(Path(__file__).parent.parent.parent)

BUILT_EXTENSION = False
BUILT_EXTENSION_ERROR = None

LINKED_LIBPYTHON = False
LINKED_LIBPYTHON_ERROR = None

extension_module = None


def _detect_backend():
    """
    Determines which GPU backend this PyTorch build targets.

    Returns one of ``"cuda"``, ``"hip"`` or ``"sycl"``. HIP builds report a
    ``torch.version.cuda`` of ``None``, so HIP must be tested first.

    All three checks are build-time properties of the PyTorch install, not
    runtime device queries, so importing works on a machine with no
    accelerator attached (a CI builder, for instance).
    """
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

# The SYCL backend needs at least the 2.7 APIs (torch.library.register_autocast;
# 2.6 for the XPU device and cpp_extension's SYCL support), but it is only
# tested against the 2.8 floor the rest of the project already requires for
# AOTI and export, so that is what is enforced.
if IS_SYCL and Version(torch.__version__) < Version("2.8"):
    raise RuntimeError(
        f"The SYCL backend requires PyTorch >= 2.8, found {torch.__version__}."
    )

# The torch device type that tensors passed to the kernels must live on.
DEVICE_TYPE = "xpu" if IS_SYCL else "cuda"


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
        if BACKEND == "cuda":
            extra_link_args.extend(["-lcuda", "-lcudart", "-lnvrtc", "-lcublas"])

            try:
                torch_libs, cuda_libs = library_paths("cuda")
                extra_link_args.append("-Wl,-rpath," + torch_libs)
                extra_link_args.append("-L" + cuda_libs)
                if os.path.exists(cuda_libs + "/stubs"):
                    extra_link_args.append("-L" + cuda_libs + "/stubs")
            except Exception as e:
                getLogger().info(str(e))

            extra_cflags.append("-DCUDA_BACKEND")
        elif BACKEND == "hip":
            torch_libs = library_paths("cuda")[0]
            extra_link_args.append("-Wl,-rpath," + torch_libs)
            extra_cflags.append("-DHIP_BACKEND")
        elif BACKEND == "sycl":
            # torch.utils.cpp_extension compiles with $CXX (default c++),
            # which must be the oneAPI DPC++ driver for -fsycl to work.
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

            # SYCL sources must be compiled and linked by the SYCL compiler
            # driver; -fsycl is required on both the compile and link lines.
            extra_cflags.extend(["-fsycl", "-DSYCL_BACKEND"])
            extra_link_args.extend(
                ["-fsycl", "-ltorch_xpu", "-lc10_xpu", "-lmkl_sycl_blas"]
            )

            for lib_dir in library_paths("xpu"):
                extra_link_args.append("-Wl,-rpath," + lib_dir)
                extra_link_args.append("-L" + lib_dir)

            mkl_root = os.environ.get("MKLROOT")
            if mkl_root:
                mkl_lib = os.path.join(mkl_root, "lib")
                extra_link_args.append("-L" + mkl_lib)
                extra_link_args.append("-Wl,-rpath," + mkl_lib)
                extra_include_dirs.append(os.path.join(mkl_root, "include"))

        torch_sources = [oeq_root + "/extension/" + src for src in torch_sources]
        include_dirs = (
            [oeq_root + "/extension/" + d for d in include_dirs]
            + extra_include_dirs
            + include_paths("xpu" if BACKEND == "sycl" else "cuda")
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
        if BACKEND == "cuda":
            import openequivariance._torch.extlib.oeq_stable_cuda as extension_module
        elif BACKEND == "hip":
            import openequivariance._torch.extlib.oeq_stable_hip as extension_module
        elif BACKEND == "sycl":
            import openequivariance._torch.extlib.oeq_stable_sycl as extension_module

        torch.ops.load_library(extension_module.__file__)
        BUILT_EXTENSION = True
    except Exception as e:
        BUILT_EXTENSION_ERROR = (
            f"Error loading precompiled OpenEquivariance Extension: {e}"
        )


USE_PRECOMPILED_EXTENSION = True
WARNING_MESSAGE = ""

if os.getenv("OEQ_JIT_EXTENSION", "0") == "1":
    WARNING_MESSAGE += "Environment variable OEQ_JIT_EXTENSION=1 is set.\n"
    USE_PRECOMPILED_EXTENSION = False

if Version(torch.__version__) <= Version("2.9.9"):
    WARNING_MESSAGE += f"PyTorch version {torch.__version__} is < 2.10, minimum required for precompiled extension. Please upgrade to 2.10.\n"
    USE_PRECOMPILED_EXTENSION = False

if BACKEND == "hip":
    WARNING_MESSAGE += "HIP does not support precompiled extension yet.\n"
    USE_PRECOMPILED_EXTENSION = False

if BACKEND == "sycl":
    WARNING_MESSAGE += "SYCL does not support precompiled extension yet.\n"
    USE_PRECOMPILED_EXTENSION = False

AOTI_SO_NAME = f"liboeq_stable_{BACKEND}_aoti.so"

if not os.path.exists(os.path.join(os.path.dirname(__file__), AOTI_SO_NAME)):
    WARNING_MESSAGE += "Precompiled extension shared object not found.\n"
    USE_PRECOMPILED_EXTENSION = False


if USE_PRECOMPILED_EXTENSION:
    load_precompiled_extension()
else:
    WARNING_MESSAGE += "For these reasons, falling back to JIT compilation of OpenEquivariance extension. If another process holds the build lock, this waits up to 5 minutes, then raises an error naming the lock file to delete.\n"
    warnings.warn(WARNING_MESSAGE, stacklevel=3)
    load_jit_extension()


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
