from functools import lru_cache

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


@lru_cache(maxsize=8)
def get_jinja_environment(backend="cuda", warp_size=32):
    """:param warp_size: only consulted by SYCL, which bakes the sub-group size
    into the generated kernel as a compile-time property."""
    if backend not in ("cuda", "hip", "sycl"):
        raise ValueError(f"Unknown kernel backend '{backend}'")

    env = Environment(
        loader=PackageLoader("openequivariance"), extensions=["jinja2.ext.do"]
    )
    env.globals["raise"] = raise_helper
    env.globals["divide"] = divide
    env.globals["sizeof"] = sizeof
    env.globals["enumerate"] = enumerate

    is_hip = backend == "hip"
    is_sycl = backend == "sycl"

    env.globals["is_hip"] = is_hip
    env.globals["is_sycl"] = is_sycl
    env.globals["warp_size"] = warp_size

    if is_sycl:
        env.globals["syncwarp"] = "_sycl_syncwarp()"
        env.globals["atomic_add"] = "_sycl_atomic_add"
        env.globals["shfl_down"] = (
            lambda val, offset: f"_sycl_shfl_down({val}, {offset})"
        )
    elif is_hip:
        env.globals["syncwarp"] = (
            '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");'
            "__builtin_amdgcn_wave_barrier();"
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");'
        )
        env.globals["atomic_add"] = "unsafeAtomicAdd"
        env.globals["shfl_down"] = lambda val, offset: f"__shfl_down( {val}, {offset})"
    else:
        env.globals["syncwarp"] = "__syncwarp()"
        env.globals["atomic_add"] = "atomicAdd"
        env.globals["shfl_down"] = (
            lambda val, offset: f"__shfl_down_sync(FULL_MASK, {val}, {offset})"
        )

    return env
