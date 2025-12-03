import pytest
import os

os.environ["JAX_ENABLE_X64"] = "True"
def pytest_addoption(parser):
    parser.addoption(
        "--jax", action="store", default=False, help="Test the JAX frontend instead of PyTorch"
    )