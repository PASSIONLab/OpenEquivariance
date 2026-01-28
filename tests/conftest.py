import os

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
