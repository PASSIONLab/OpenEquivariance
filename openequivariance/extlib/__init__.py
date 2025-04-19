import os, warnings, tempfile, warnings
from pathlib import Path

from openequivariance.benchmark.logging_utils import getLogger

oeq_root = str(Path(__file__).parent.parent)

build_ext = True 
TORCH_COMPILE=True
torch_module, generic_module = None, None
postprocess_kernel = lambda kernel: kernel

if not build_ext: 
    from openequivariance.extlib.generic_module import * 
else:
    from setuptools import setup
    from torch.utils.cpp_extension import library_paths, include_paths

    global torch
    import torch

    extra_cflags=["-O3"]
    generic_sources = ['generic_module.cpp']
    torch_sources = ['torch_tp_jit.cpp']

    include_dirs, extra_link_args = ['util'], None 
    if torch.cuda.is_available() and torch.version.cuda:
        extra_link_args = ['-Wl,--no-as-needed', '-lcuda', '-lcudart', '-lnvrtc']

        try:
            cuda_libs = library_paths('cuda')[1]
            extra_link_args.append('-L' + cuda_libs)
            if os.path.exists(cuda_libs + '/stubs'):
                extra_link_args.append('-L' + cuda_libs + '/stubs')
        except Exception as e:
            getLogger().info(str(e))

        extra_cflags.append("-DCUDA_BACKEND")
    elif torch.cuda.is_available() and torch.version.hip:
        extra_link_args = [ '-Wl,--no-as-needed', '-lhiprtc']

        def postprocess(kernel):
            kernel = kernel.replace("__syncwarp();", "__threadfence_block();")
            kernel = kernel.replace("__shfl_down_sync(FULL_MASK,", "__shfl_down(")
            return kernel 
        postprocess_kernel = postprocess

        extra_cflags.append("-DHIP_BACKEND")

    generic_sources = [oeq_root + '/extension/' + src for src in generic_sources]
    torch_sources = [oeq_root + '/extension/' + src for src in torch_sources]
    include_dirs = [oeq_root + '/extension/' + d for d in include_dirs] + include_paths('cuda')

    torch_compile_exception = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            torch_module = torch.utils.cpp_extension.load("torch_tp_jit",
                torch_sources, extra_cflags=extra_cflags, extra_include_paths=include_dirs, extra_ldflags=extra_link_args) 
            torch.ops.load_library(torch_module.__file__)
        except Exception as e:
            # If compiling torch fails (e.g. low gcc version), we should fall back to the
            # version that takes integer pointers as args (but is untraceable to PyTorch JIT / export). 
            TORCH_COMPILE=False
            torch_compile_exception = e

        generic_module = torch.utils.cpp_extension.load("generic_module",
            generic_sources, extra_cflags=extra_cflags, extra_include_paths=include_dirs, extra_ldflags=extra_link_args)

    if not TORCH_COMPILE:
        warnings.warn("Could not compile integrated PyTorch wrapper. Falling back to Pybind11" +
                            f", but JITScript, compile fullgraph, and export will fail.\n {torch_compile_exception}")

from generic_module import *