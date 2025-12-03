from importlib.metadata import version


def test_import():
    import openequivariance

    assert openequivariance.__version__ is not None
    assert openequivariance.__version__ != "0.0.0"
    assert openequivariance.__version__ == version("openequivariance")
