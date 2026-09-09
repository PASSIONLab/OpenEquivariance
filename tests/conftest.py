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


@pytest.fixture(scope="session")
def with_jax(request):
    return request.config.getoption("--jax")


def device_type():
    """
    The torch device type the kernels run on for the detected backend:
    ``"xpu"`` for SYCL, ``"cuda"`` for CUDA and HIP.
    """
    from openequivariance._torch.extlib import DEVICE_TYPE

    return DEVICE_TYPE


def torch_accelerator():
    """The ``torch.cuda`` / ``torch.xpu`` module matching the active backend."""
    import torch

    return getattr(torch, device_type())


@pytest.fixture(scope="session")
def device():
    return device_type()
