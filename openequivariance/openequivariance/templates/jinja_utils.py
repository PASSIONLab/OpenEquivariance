from jinja2 import Environment, PackageLoader, StrictUndefined


def raise_helper(msg):
    raise Exception(msg)


def divide(numerator, denominator):
    return numerator // denominator


def sizeof(dtype):
    if dtype in ["float", "int", "unsigned int"]:
        return 4
    else:
        raise Exception("Provided undefined datatype to sizeof!")


def get_jinja_environment():
    # StrictUndefined: an undefined variable in a template is an error rather
    # than an empty string. Kernels index shared memory with rendered constants,
    # so an empty render silently produces `smem[ + k * 32 + lane_id]`, which
    # still compiles and is only correct when the offset happens to be zero.
    env = Environment(
        loader=PackageLoader("openequivariance"),
        extensions=["jinja2.ext.do"],
        undefined=StrictUndefined,
    )
    env.globals["raise"] = raise_helper
    env.globals["divide"] = divide
    env.globals["sizeof"] = sizeof
    env.globals["enumerate"] = enumerate
    return env
