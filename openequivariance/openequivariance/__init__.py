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

TensorProduct, TensorProductConv, torch_ext_so_path, torch_to_oeq_dtype = None, None, None, None

if "OEQ_NOTORCH" not in os.environ or os.environ["OEQ_NOTORCH"] != "1":
    import torch
    from openequivariance.impl_torch.TensorProduct import TensorProduct 
    from openequivariance.impl_torch.TensorProductConv import TensorProductConv

    from openequivariance.impl_torch.extlib import torch_ext_so_path
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

jax = None
try:
    import openequivariance_extjax
    import openequivariance.impl_jax as jax
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
    "jax"
]
