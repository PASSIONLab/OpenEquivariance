# ruff: noqa: F401
import sys
import os
import numpy as np

from pathlib import Path
from importlib.metadata import version

from openequivariance.core.e3nn_lite import (
    TPProblem,
    Irrep,
    Irreps,
    _MulIr,
    Instruction,
)

__version__ = None
try:
    __version__ = version("openequivariance")
except Exception as e:
    print(f"Warning: Could not determine oeq version: {e}", file=sys.stderr)


def _check_package_editable():
    import json
    from importlib.metadata import Distribution

    direct_url = Distribution.from_name("openequivariance").read_text("direct_url.json")
    return json.loads(direct_url).get("dir_info", {}).get("editable", False)


_editable_install_output_path = Path(__file__).parent.parent.parent / "outputs"


def extension_source_path():
    """
    :returns: Path to the source code of the C++ extension.
    """
    return str(Path(__file__).parent / "extension")


if "OEQ_NOTORCH" not in os.environ or os.environ["OEQ_NOTORCH"] != "1":
    import torch

    from openequivariance._torch.TensorProduct import TensorProduct
    from openequivariance._torch.TensorProductConv import TensorProductConv

    from openequivariance._torch.extlib import (
        torch_ext_so_path as torch_ext_so_path_internal,
    )
    from openequivariance.core.utils import torch_to_oeq_dtype

    torch.serialization.add_safe_globals(
        [
            TensorProduct,
            TensorProductConv,
            TPProblem,
            Irrep,
            Irreps,
            _MulIr,
            Instruction,
            np.float32,
            np.float64,
        ]
    )


def torch_ext_so_path():
    """
    :returns: Path to a ``.so`` file that must be linked to use OpenEquivariance
              from the PyTorch C++ Interface.
    """
    try:
        return torch_ext_so_path_internal()
    except NameError:
        return None


jax = None
try:
    import openequivariance_extjax
    import openequivariance.jax as jax
except ImportError:
    pass

__all__ = [
    "TPProblem",
    "Irreps",
    "TensorProduct",
    "TensorProductConv",
    "torch_to_oeq_dtype",
    "_check_package_editable",
    "torch_ext_so_path",
    "jax",
]
