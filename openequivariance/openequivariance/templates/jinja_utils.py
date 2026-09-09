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


def get_jinja_environment(backend="cuda", warp_size=32):
    """
    Builds the Jinja environment used to render the kernel templates.

    :param backend: one of ``"cuda"``, ``"hip"`` or ``"sycl"``.
    :param warp_size: size of a warp / wavefront / sub-group. Only consulted by
                      the SYCL backend, which must bake the sub-group size into
                      the generated kernel as a compile-time property.
    """
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

    env.globals["backend"] = backend
    env.globals["is_hip"] = is_hip
    env.globals["is_sycl"] = is_sycl
    env.globals["warp_size"] = warp_size

    if is_sycl:
        # Provided by templates/sycl_compat.cuh.
        env.globals["syncwarp"] = "oeq_syncwarp()"
        env.globals["atomic_add"] = "oeq_atomic_add"
        env.globals["shfl_down"] = lambda val, offset: f"oeq_shfl_down({val}, {offset})"
    elif is_hip:
        env.globals["syncwarp"] = "__threadfence_block()"
        env.globals["atomic_add"] = "unsafeAtomicAdd"
        env.globals["shfl_down"] = lambda val, offset: f"__shfl_down( {val}, {offset})"
    else:
        env.globals["syncwarp"] = "__syncwarp()"
        env.globals["atomic_add"] = "atomicAdd"
        env.globals["shfl_down"] = (
            lambda val, offset: f"__shfl_down_sync(FULL_MASK, {val}, {offset})"
        )

    return env
