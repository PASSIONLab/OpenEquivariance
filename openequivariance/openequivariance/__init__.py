# ruff: noqa: F401
import sys
import os
import numpy as np

from pathlib import Path
import warnings
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


def check_package_editable():
    import json
    from importlib.metadata import Distribution, PackageNotFoundError

    try:
        dist = Distribution.from_name("openequivariance")
    except PackageNotFoundError:
        return False
    direct_url = dist.read_text("direct_url.json") or "{}"
    return json.loads(direct_url).get("dir_info", {}).get("editable", False)


_editable_install_output_path = Path(__file__).parent.parent.parent / "outputs"


def skip_rebuild_without_flock():
    import errno
    import fcntl

    for finder in sys.meta_path:
        build_dir = getattr(finder, "path", None)
        if (
            not getattr(finder, "rebuild_flag", False)
            or "openequivariance" not in getattr(finder, "known_source_files", {})
            or not build_dir
            or not os.path.isdir(build_dir)
        ):
            continue
        fd = os.open(
            os.path.join(build_dir, "editable_rebuild.lock"), os.O_RDWR | os.O_CREAT
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EACCES):
                skip = os.environ.get("SKBUILD_EDITABLE_SKIP", "")
                os.environ["SKBUILD_EDITABLE_SKIP"] = os.pathsep.join((skip, build_dir))
                warnings.warn(f"flock unsupported in {build_dir}, skipping rebuild.")
        finally:
            os.close(fd)


if check_package_editable():
    skip_rebuild_without_flock()

if "OEQ_NOTORCH" not in os.environ or os.environ["OEQ_NOTORCH"] != "1":
    import torch

    from openequivariance._torch.TensorProduct import TensorProduct
    from openequivariance._torch.TensorProductConv import TensorProductConv
    from openequivariance._torch.utils import transpose_irreps

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

    from openequivariance._torch.extlib import BUILT_EXTENSION, BUILT_EXTENSION_ERROR


def torch_ext_so_path():
    """
    :returns: Path to a ``.so`` file that must be linked to use OpenEquivariance
              from the PyTorch C++ Interface, or ``None`` if PyTorch support is
              disabled (``OEQ_NOTORCH=1``).
    :raises ImportError: If the OpenEquivariance extension failed to load.
    """
    try:
        return torch_ext_so_path_internal()
    except NameError:
        return None


def __getattr__(name):
    if name == "jax":
        import openequivariance.jax as jax_module

        return jax_module
    raise AttributeError(f"module 'openequivariance' has no attribute {name!r}")


__all__ = [
    "TPProblem",
    "Irreps",
    "TensorProduct",
    "TensorProductConv",
    "torch_to_oeq_dtype",
    "check_package_editable",
    "torch_ext_so_path",
    "transpose_irreps",
]
