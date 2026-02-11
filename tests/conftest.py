import os
import pytest

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


def pytest_addoption(parser):
    parser.addoption(
        "--jax",
        action="store_true",
        default=False,
        help="Test the JAX frontend instead of PyTorch",
    )
    parser.addoption(
        "--test-production-configs",
        action="store_true",
        default=False,
        help="Run production model configuration tests (opt-in)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "production_configs: opt-in production model configuration tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--test-production-configs"):
        return
    skip_prod = pytest.mark.skip(reason="need --test-production-configs to run")
    for item in items:
        if "production_configs" in item.keywords:
            item.add_marker(skip_prod)
