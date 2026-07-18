"""
TEMPORARY test for folding `postprocess_kernel` into the Jinja pipeline.

`extlib.postprocess_kernel` (both _torch and jax variants) does three string
replacements on the rendered kernel when running on HIP:

    1. "__syncwarp();"              -> "__threadfence_block();"
    2. "__shfl_down_sync(FULL_MASK," -> "__shfl_down("
    3. "atomicAdd"                   -> "unsafeAtomicAdd"

This test:
  * renders kernels for a matrix of tensor products / convolutions on CPU
    (no GPU required -- rendering is pure Python),
  * characterizes exactly what postprocess_kernel changes (and asserts it
    catches *every* occurrence, e.g. no "__syncwarp()" without a semicolon
    that the string replace would silently miss),
  * pins the CUDA render and the postprocessed HIP render as sha256 goldens
    in kernel_postprocess_goldens.json,
  * once LoopUnrollTP/LoopUnrollConv grow an `is_hip` flag (the "new
    process"), verifies the Jinja-rendered HIP kernel is byte-identical to
    the old postprocessed output.

Regenerate goldens with OEQ_REGEN_GOLDENS=1. Delete this file (and the
goldens) once postprocess_kernel is removed.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

# Rendering needs no GPU. If torch is missing or has no CUDA/HIP backend,
# keep openequivariance/__init__.py from importing its torch extension.
try:
    import torch

    _TORCH_USABLE = bool(torch.version.cuda or torch.version.hip)
except ImportError:
    _TORCH_USABLE = False
if not _TORCH_USABLE:
    os.environ["OEQ_NOTORCH"] = "1"

import numpy as np  # noqa: E402

from openequivariance.core.e3nn_lite import TPProblem  # noqa: E402
from openequivariance.core.LoopUnrollTP import LoopUnrollTP  # noqa: E402
from openequivariance.core.LoopUnrollConv import LoopUnrollConv  # noqa: E402
from openequivariance.benchmark.problems import (  # noqa: E402
    diffdock_problems,
    mace_problems,
)

GOLDEN_PATH = Path(__file__).parent / "kernel_postprocess_goldens.json"


class FakeDeviceProp:
    """Stand-in for extlib.DeviceProp so kernels render without a GPU."""

    def __init__(self, warpsize):
        self.warpsize = warpsize
        self.maxSharedMemPerBlock = 48 * 1024
        self.multiprocessorCount = 108


def reference_hip_postprocess(kernel):
    """Verbatim copy of the HIP branch of extlib.postprocess_kernel."""
    kernel = kernel.replace("__syncwarp();", "__threadfence_block();")
    kernel = kernel.replace("__shfl_down_sync(FULL_MASK,", "__shfl_down(")
    kernel = kernel.replace("atomicAdd", "unsafeAtomicAdd")
    return kernel


def _new_process_available():
    return all(
        "is_hip" in inspect.signature(cls.__init__).parameters
        for cls in (LoopUnrollTP, LoopUnrollConv)
    )


_HAS_IS_HIP = _new_process_available()


def _uvu_f64_problem():
    return TPProblem(
        "32x1e + 8x2e",
        "1x1e + 1x2e",
        "32x1e + 8x2e",
        [(0, 0, 0, "uvu", True), (1, 1, 1, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
        irrep_dtype=np.float64,
        weight_dtype=np.float64,
    )


def _shared_weight_uvw_problem():
    return TPProblem(
        "16x2e",
        "4x2e",
        "16x2e",
        [(0, 0, 0, "uvw", True)],
        shared_weights=True,
        internal_weights=False,
        irrep_dtype=np.float32,
        weight_dtype=np.float32,
    )


def _render(kind, problem, dp, is_hip):
    # Pre-refactor, the third constructor argument was a postprocessing
    # callable; post-refactor it is the is_hip flag itself.
    if _HAS_IS_HIP:
        backend_arg = is_hip
    else:
        backend_arg = reference_hip_postprocess if is_hip else (lambda k: k)

    if kind == "batch":
        return LoopUnrollTP(problem, dp, backend_arg, torch_op=False).jit_kernel
    if kind == "conv_atomic":
        return LoopUnrollConv(
            problem, dp, backend_arg, torch_op=False, deterministic=False
        ).jit_kernel
    if kind == "conv_det":
        return LoopUnrollConv(
            problem, dp, backend_arg, torch_op=False, deterministic=True
        ).jit_kernel
    if kind == "conv_det_kahan":
        return LoopUnrollConv(
            problem, dp, backend_arg, torch_op=False, deterministic=True, kahan=True
        ).jit_kernel
    raise ValueError(kind)


CASES = {
    "batch_mace0_f32": ("batch", lambda: mace_problems()[0]),
    "batch_diffdock0_f32": ("batch", lambda: diffdock_problems()[0]),
    "batch_uvu_f64": ("batch", _uvu_f64_problem),
    "batch_uvw_shared_f32": ("batch", _shared_weight_uvw_problem),
    "conv_atomic_mace0_f32": ("conv_atomic", lambda: mace_problems()[0]),
    "conv_det_mace0_f32": ("conv_det", lambda: mace_problems()[0]),
    "conv_det_kahan_mace0_f32": ("conv_det_kahan", lambda: mace_problems()[0]),
}

WARP_SIZES = [32, 64]  # NVIDIA / AMD CDNA


def _case_params():
    return [
        pytest.param(case_id, warpsize, id=f"{case_id}-w{warpsize}")
        for case_id in CASES
        for warpsize in WARP_SIZES
    ]


def _render_case(case_id, warpsize, is_hip=False):
    kind, problem_fn = CASES[case_id]
    return _render(kind, problem_fn(), FakeDeviceProp(warpsize), is_hip)


def _sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Characterization of the OLD process (passes today).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id,warpsize", _case_params())
def test_postprocess_fully_characterized(case_id, warpsize):
    """The three string replacements catch every relevant token, and nothing
    else in the kernel can be clobbered by them."""
    cuda_kernel = _render_case(case_id, warpsize, is_hip=False)

    # Every __syncwarp appears exactly as "__syncwarp();" -- the string
    # replace would silently miss any other spelling.
    n_syncwarp = cuda_kernel.count("__syncwarp")
    assert n_syncwarp == cuda_kernel.count("__syncwarp();")

    # Every warp shuffle appears exactly as "__shfl_down_sync(FULL_MASK,".
    n_shfl = cuda_kernel.count("__shfl_down")
    assert n_shfl == cuda_kernel.count("__shfl_down_sync(FULL_MASK,")

    # "atomicAdd" is replaced globally: no identifier may contain it as a
    # substring other than the calls themselves ("unsafeAtomicAdd" pre-image
    # would double-replace).
    assert "unsafeAtomicAdd" not in cuda_kernel

    hip_kernel = reference_hip_postprocess(cuda_kernel)

    assert "__syncwarp" not in hip_kernel
    assert "__shfl_down_sync" not in hip_kernel
    n_atomic = cuda_kernel.count("atomicAdd")
    assert hip_kernel.count("unsafeAtomicAdd") == n_atomic
    assert hip_kernel.count("atomicAdd") == 0  # every call became the unsafe one

    assert hip_kernel.count("__threadfence_block();") >= n_syncwarp


@pytest.mark.parametrize("case_id,warpsize", _case_params())
def test_golden_hashes(case_id, warpsize):
    """Pin CUDA render + postprocessed HIP render byte-for-byte, so the Jinja
    refactor can prove it changes nothing. Regen: OEQ_REGEN_GOLDENS=1."""
    goldens = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}
    key = f"{case_id}-w{warpsize}"

    cuda_kernel = _render_case(case_id, warpsize, is_hip=False)
    hip_kernel = reference_hip_postprocess(cuda_kernel)
    entry = {"cuda_sha256": _sha256(cuda_kernel), "hip_sha256": _sha256(hip_kernel)}

    if os.environ.get("OEQ_REGEN_GOLDENS") == "1" or key not in goldens:
        goldens[key] = entry
        GOLDEN_PATH.write_text(json.dumps(goldens, indent=2, sort_keys=True) + "\n")
    else:
        assert goldens[key] == entry, (
            f"Rendered kernel for {key} deviates from the pinned golden. If the "
            "change is intentional, regenerate with OEQ_REGEN_GOLDENS=1."
        )


# ---------------------------------------------------------------------------
# Equivalence of the NEW process (skipped until the Jinja pipeline handles HIP).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_IS_HIP,
    reason="LoopUnrollTP/LoopUnrollConv do not take is_hip yet (new process not implemented)",
)
@pytest.mark.parametrize("case_id,warpsize", _case_params())
def test_jinja_hip_matches_postprocess(case_id, warpsize):
    """The Jinja-rendered HIP kernel must be byte-identical to what the old
    string-replacement postprocess produced, and the CUDA render must be
    unchanged relative to the pre-refactor goldens."""
    cuda_new = _render_case(case_id, warpsize, is_hip=False)
    hip_new = _render_case(case_id, warpsize, is_hip=True)

    assert hip_new == reference_hip_postprocess(cuda_new), (
        "Jinja HIP render differs from old postprocess_kernel output"
    )

    goldens = json.loads(GOLDEN_PATH.read_text())
    key = f"{case_id}-w{warpsize}"
    assert _sha256(cuda_new) == goldens[key]["cuda_sha256"], (
        "CUDA render changed relative to pre-refactor golden"
    )
    assert _sha256(hip_new) == goldens[key]["hip_sha256"], (
        "HIP render changed relative to pre-refactor golden"
    )
