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


def get_jinja_environment(is_hip=False):
    env = Environment(
        loader=PackageLoader("openequivariance"), extensions=["jinja2.ext.do"]
    )
    env.globals["raise"] = raise_helper
    env.globals["divide"] = divide
    env.globals["sizeof"] = sizeof
    env.globals["enumerate"] = enumerate

    env.globals["is_hip"] = is_hip
    env.globals["syncwarp"] = "__threadfence_block()" if is_hip else "__syncwarp()"
    env.globals["atomic_add"] = "unsafeAtomicAdd" if is_hip else "atomicAdd"

    if is_hip:
        env.globals["shfl_down"] = lambda val, offset: f"__shfl_down( {val}, {offset})"
    else:
        env.globals["shfl_down"] = (
            lambda val, offset: f"__shfl_down_sync(FULL_MASK, {val}, {offset})"
        )
    return env
