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


def get_jinja_environment():
    env = Environment(
        loader=PackageLoader("openequivariance"), extensions=["jinja2.ext.do"]
    )
    env.globals["raise"] = raise_helper
    env.globals["divide"] = divide
    env.globals["sizeof"] = sizeof
    env.globals["enumerate"] = enumerate
    return env
