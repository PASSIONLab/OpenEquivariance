# ruff: noqa: F401
import openequivariance.extlib
from pathlib import Path
from importlib.metadata import version

from openequivariance.implementations.e3nn_lite import TPProblem, Irreps
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.implementations.convolution.TensorProductConv import (
    TensorProductConv,
)
from openequivariance.implementations.utils import torch_to_oeq_dtype

__version__ = version("openequivariance")


def _check_package_editable():
    import json
    from importlib.metadata import Distribution

    direct_url = Distribution.from_name("openequivariance").read_text("direct_url.json")
    return json.loads(direct_url).get("dir_info", {}).get("editable", False)


_editable_install_output_path = Path(__file__).parent.parent / "outputs"

def torch_ext_so_path():
    return openequivariance.extlib.torch_module.__file__


__all__ = [
    "TPProblem",
    "Irreps",
    "TensorProduct",
    "TensorProductConv",
    "torch_to_oeq_dtype",
    "_check_package_editable",
    "torch_ext_so_path"
]