import os, warnings, tempfile
from pathlib import Path

from openequivariance.benchmark.logging_utils import getLogger

oeq_root = str(Path(__file__).parent.parent)

build_ext = True 
candidates = [f for f in os.listdir(oeq_root + '/extlib') 
                if f.startswith('kernel_wrapper')]

if len(candidates) == 1:
    build_ext = False 

kernel_wrapper = None
torch_wrapper = None

postprocess_kernel = lambda kernel: kernel

if not build_ext: 
    from openequivariance.extlib.kernel_wrapper import * 
else:
    from setuptools import setup
    from torch.utils.cpp_extension import library_paths, include_paths

    global torch
    import torch

    generic_sources = ['kernel_wrapper.cpp']
    aten_sources = ['torch_wrapper.cpp']
    extra_cflags=["-O3"]

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
    aten_sources = [oeq_root + '/extension/' + src for src in aten_sources]

    include_dirs = [oeq_root + '/extension/' + d for d in include_dirs] + include_paths('cuda')

    with warnings.catch_warnings():
#        with tempfile.TemporaryDirectory() as tmpdir: // Might be needed for RocM
        warnings.simplefilter("ignore")
        #kernel_wrapper = torch.utils.cpp_extension.load("kernel_wrapper",
        #    generic_sources,
        #    extra_cflags=extra_cflags,
        #    extra_include_paths=include_dirs,
        #    extra_ldflags=extra_link_args)

        # Need to locate the correct compiler here... 
        torch_wrapper = torch.utils.cpp_extension.load("torch_wrapper",
            aten_sources,
            extra_cflags=extra_cflags,
            extra_include_paths=include_dirs,
            extra_ldflags=extra_link_args)
        torch.ops.load_library(torch_wrapper.__file__)

        print("Torch ops load library succeeded!")

#from kernel_wrapper import *