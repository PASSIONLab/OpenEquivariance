import os, warnings, logging
from pathlib import Path

oeq_root = str(Path(__file__).parent.parent)

build_ext = True 
candidates = [f for f in os.listdir(oeq_root + '/extlib') 
                if f.startswith('kernel_wrapper')]

if len(candidates) == 1:
    build_ext = False 

kernel_wrapper = None

postprocess_kernel = lambda kernel: kernel


if not build_ext: 
    from openequivariance.extlib.kernel_wrapper import * 
else:
    from setuptools import setup
    from torch.utils.cpp_extension import library_paths, include_paths

    global torch
    import torch

    sources, include_dirs, extra_link_args = None, ['util'], None 
    if torch.cuda.is_available() and torch.version.cuda: 
        sources = ['kernel_wrapper.cpp']
        extra_link_args = ['-Wl,--no-as-needed', '-lcuda', '-lcudart', '-lnvrtc']

        try:
            cuda_libs = library_paths('cuda')[1]
            extra_link_args.append('-L' + cuda_libs)
            if os.path.exists(cuda_libs + '/stubs'):
                extra_link_args.append('-L' + cuda_libs + '/stubs')
        except Exception as e:
            logging.info(str(e))
    elif torch.cuda.is_available() and torch.version.hip:
        sources = ['hip_wrapper.cpp']
        extra_link_args = [ '-Wl,--no-as-needed', '-lhiprtc']

        def postprocess(kernel):
            kernel = kernel.replace("__syncwarp();", "")
            kernel = kernel.replace("__shfl_down_sync(FULL_MASK,", "__shfl_down(")
            return kernel 
        postprocess_kernel = postprocess

    sources = [oeq_root + '/extension/' + src for src in sources]
    include_dirs = [oeq_root + '/extension/' + d for d in include_dirs] + include_paths('cuda')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel_wrapper = torch.utils.cpp_extension.load(sources[0].split('.')[0],
            sources,
            extra_cflags=["-O3"],
            extra_include_paths=include_dirs,
            extra_ldflags=extra_link_args)

from kernel_wrapper import *