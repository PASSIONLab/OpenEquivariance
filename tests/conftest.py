import os
import sys

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


@pytest.fixture(scope="session")
def with_jax(request):
    return request.config.getoption("--jax")


def device_type():
    # Called at module scope, before fixtures exist, so --jax is read from the
    # command line rather than the with_jax fixture. This keeps JAX-only runs
    # from importing the torch extension module.
    if "--jax" in sys.argv:
        from openequivariance.jax.extlib import DEVICE_TYPE
    else:
        from openequivariance._torch.extlib import DEVICE_TYPE

    return DEVICE_TYPE


def torch_accelerator():
    import torch

    return getattr(torch, device_type())
