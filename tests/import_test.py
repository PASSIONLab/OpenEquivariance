from importlib.metadata import version


def test_import():
    import openequivariance

    assert openequivariance.__version__ is not None
    assert openequivariance.__version__ != "0.0.0"
    assert openequivariance.__version__ == version("openequivariance")


def test_extension_built():
    from openequivariance import BUILT_EXTENSION, BUILT_EXTENSION_ERROR

    assert BUILT_EXTENSION_ERROR is None
    assert BUILT_EXTENSION


def test_torch_extension_built():
    from openequivariance import TORCH_COMPILE, TORCH_COMPILE_ERROR

    assert TORCH_COMPILE_ERROR is None
    assert TORCH_COMPILE
