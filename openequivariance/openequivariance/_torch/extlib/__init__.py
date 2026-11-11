# ruff: noqa : F401, E402
import sys
import os
import time
import warnings
import sysconfig
import contextlib
import importlib
import importlib.util
from pathlib import Path
from packaging.version import Version

import torch

if Version(torch.__version__) < Version("2.10"):
    raise RuntimeError(
        f"OpenEquivariance requires PyTorch >= 2.10, found {torch.__version__}."
    )
oeq_root = str(Path(__file__).parent.parent.parent)

SOURCE_BUILD_HINT = (
    "To build the extension for your setup from source, run:\n"
    "    pip install --no-binary openequivariance openequivariance"
)


class Extension:
    def __init__(self):
        self.built_extension = False
        self.built_extension_error = None
        self.ext_module_python = None
        self.aoti_module_path = None


class JITExtension(Extension):
    """
    Only SYCL uses PyTorch's JIT compilation flow; CUDA and HIP now use
    precompiled extensions shipped with the wheel.
    """

    @staticmethod
    @contextlib.contextmanager
    def maybe_set_max_jobs_env(limit=8):
        if "MAX_JOBS" in os.environ:
            yield
            return
        os.environ["MAX_JOBS"] = str(min(limit, os.cpu_count() or 1))
        try:
            yield
        finally:
            os.environ.pop("MAX_JOBS", None)

    @staticmethod
    def wait_for_torch_build_lock(name, timeout=300):
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

    def __init__(self):
        super().__init__()
        python_lib_dir = sysconfig.get_config_var("LIBDIR") or ""
        python_lib_name = f"python{sys.version_info.major}.{sys.version_info.minor}"
        found_libpython = any(
            os.path.exists(os.path.join(python_lib_dir, f"lib{python_lib_name}{ext}"))
            for ext in (".so", ".a")
        )

        missing = [
            name
            for name in ("setuptools", "ninja")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            self.built_extension_error = (
                f"The SYCL backend compiles its extension at first import, which "
                f"needs {', '.join(missing)}. Install them with:\n"
                "    pip install openequivariance[sycl]"
            )
            return

        try:
            from torch.utils.cpp_extension import library_paths, include_paths

            extra_cflags = ["-O3"]
            torch_sources = ["libtorch_tp_jit.cpp", "json11/json11.cpp"]

            include_dirs, extra_link_args = (["backend"], ["-Wl,--no-as-needed"])

            if found_libpython:
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
                    self.built_extension_error = (
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
            include_dirs = [
                oeq_root + "/extension/" + d for d in include_dirs
            ] + include_paths("xpu")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                try:
                    self.wait_for_torch_build_lock("libtorch_tp_jit")
                    with self.maybe_set_max_jobs_env():
                        self.ext_module_python = torch.utils.cpp_extension.load(
                            "libtorch_tp_jit",
                            torch_sources,
                            extra_cflags=extra_cflags,
                            extra_include_paths=include_dirs,
                            extra_ldflags=extra_link_args,
                        )
                    torch.ops.load_library(self.ext_module_python.__file__)
                    self.built_extension = True
                    self.aoti_module_path = self.ext_module_python.__file__

                except Exception as e:
                    # If compiling torch fails (e.g. low gcc version), we should fall back to the
                    # version that takes integer pointers as args (but is untraceable to PyTorch JIT / export).
                    self.built_extension_error = e
        except Exception as e:
            self.built_extension_error = (
                f"Error JIT-compiling OpenEquivariance Extension: {e}"
            )


class PrecompiledExtension(Extension):
    def __init__(self, backend):
        super().__init__()
        if backend == "cuda":
            ext_module_py_name = f"oeq_stable_cuda{torch.version.cuda.split('.')[0]}"
        else:
            ext_module_py_name = f"oeq_stable_{backend}"

        full_name = f"openequivariance._torch.extlib.{ext_module_py_name}"
        spec = importlib.util.find_spec(full_name)
        aoti_module_path = spec and os.path.join(
            os.path.dirname(spec.origin), f"lib{ext_module_py_name}_aoti.so"
        )
        if spec is None or not os.path.exists(aoti_module_path):
            self.built_extension_error = (
                f"No precompiled OpenEquivariance extension {ext_module_py_name} "
                f"for PyTorch {torch.__version__}. {SOURCE_BUILD_HINT}"
            )
            return

        try:
            self.ext_module_python = importlib.import_module(full_name)
            torch.ops.load_library(self.ext_module_python.__file__)
            self.built_extension = True
            self.aoti_module_path = aoti_module_path
        except Exception as e:
            self.built_extension_error = (
                f"Error loading precompiled OpenEquivariance extension "
                f"{ext_module_py_name}: {e}\n{SOURCE_BUILD_HINT}"
            )


def _detect_backend():
    """HIP builds report a ``torch.version.cuda`` of None, so HIP is tested first."""
    if torch.version.hip:
        return "hip"
    if torch.version.cuda:
        return "cuda"
    if getattr(torch.version, "xpu", None):
        return "sycl"
    raise RuntimeError(
        "Only the CUDA, HIP and XPU (SYCL) backends are supported. "
        "No supported accelerator was detected in this PyTorch build."
    )


BACKEND = _detect_backend()
IS_HIP = BACKEND == "hip"
IS_SYCL = BACKEND == "sycl"
DEVICE_TYPE = "xpu" if IS_SYCL else "cuda"

extension = None
if BACKEND == "sycl":
    extension = JITExtension()
else:
    extension = PrecompiledExtension(BACKEND)
BUILT_EXTENSION = extension.built_extension
BUILT_EXTENSION_ERROR = extension.built_extension_error


if BUILT_EXTENSION:
    sys.modules["oeq_utilities"] = extension.ext_module_python
    from oeq_utilities import (
        DeviceProp,
        GPUTimer,
    )

    def torch_ext_so_path():
        return extension.aoti_module_path

else:
    if BUILT_EXTENSION_ERROR is not None:
        warnings.warn(str(BUILT_EXTENSION_ERROR), stacklevel=3)

    def _raise_import_error_helper(import_target: str):
        if not BUILT_EXTENSION:
            raise ImportError(
                f"Could not import {import_target}: {BUILT_EXTENSION_ERROR}"
            )

    def DeviceProp(*args, **kwargs):
        _raise_import_error_helper("DeviceProp")

    def GPUTimer(*args, **kwargs):
        _raise_import_error_helper("GPUTimer")

    def torch_ext_so_path():
        _raise_import_error_helper("torch_ext_so_path")
