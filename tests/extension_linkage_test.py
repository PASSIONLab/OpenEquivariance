import re
import shutil
import subprocess

import pytest


def test_extension_has_no_direct_blas_dependency():
    if not shutil.which("readelf"):
        pytest.skip("readelf is required to inspect ELF dependencies")
    import openequivariance
    from openequivariance._torch.extlib import extension_module

    paths = {extension_module.__file__, openequivariance.torch_ext_so_path()}
    for path in paths:
        dynamic = subprocess.check_output(["readelf", "--dynamic", path], text=True)
        dependencies = [line for line in dynamic.splitlines() if "(NEEDED)" in line]
        assert not any(
            re.search(r"lib(cublas|hipblas|rocblas)", line) for line in dependencies
        )
        symbols = subprocess.check_output(
            ["readelf", "--dyn-syms", "--wide", path], text=True
        )
        undefined = [line for line in symbols.splitlines() if " UND " in line]
        assert not any(
            re.search(r"\b(cublas|hipblas|rocblas)[A-Za-z_]", line)
            for line in undefined
        )
