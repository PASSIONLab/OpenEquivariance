def test_import():
    import openequivariance

    assert openequivariance.__version__ is not None     
    assert openequivariance.__version__ != "0.0.0"