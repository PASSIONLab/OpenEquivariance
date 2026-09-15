from functools import lru_cache

import numpy as np
from jinja2 import Environment, PackageLoader


def raise_helper(msg):
    raise Exception(msg)


def divide(numerator, denominator):
    return numerator // denominator


def sizeof(dtype):
    if dtype in ["float", "int", "unsigned int"]:
        return 4
    else:
        raise Exception("Provided undefined datatype to sizeof!")


def cpp_scalar_type(dtype):
    """Return the C++ scalar spelling for a supported generated dtype."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return "float"
    if dtype == np.dtype(np.float64):
        return "double"
    raise ValueError("generated kernels support f32 and f64 scalar types")


def cpp_scalar_literal(value, scalar):
    """Return an exact C++ literal for a generated ``float`` or ``double``."""
    if scalar not in ("float", "double"):
        raise ValueError(f"Unsupported generated scalar type: {scalar}")
    suffix = "f" if scalar == "float" else ""
    return f"static_cast<scalar_t>({float(value).hex()}{suffix})"


@lru_cache(maxsize=2)
def get_jinja_environment(is_hip=False):
    env = Environment(
        loader=PackageLoader("openequivariance"), extensions=["jinja2.ext.do"]
    )
    env.globals["raise"] = raise_helper
    env.globals["divide"] = divide
    env.globals["sizeof"] = sizeof
    env.globals["enumerate"] = enumerate
    env.globals["cpp_scalar_literal"] = cpp_scalar_literal

    env.globals["is_hip"] = is_hip
    env.globals["syncwarp"] = (
        '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");__builtin_amdgcn_wave_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");'
        if is_hip
        else "__syncwarp()"
    )
    env.globals["atomic_add"] = "unsafeAtomicAdd" if is_hip else "atomicAdd"

    if is_hip:
        env.globals["shfl_down"] = lambda val, offset: f"__shfl_down( {val}, {offset})"
        env.globals["shfl_down_32"] = lambda val, offset: (
            f"__shfl_down( {val}, {offset}, 32)"
        )
    else:
        env.globals["shfl_down"] = lambda val, offset: (
            f"__shfl_down_sync(FULL_MASK, {val}, {offset})"
        )
        env.globals["shfl_down_32"] = env.globals["shfl_down"]
    return env
