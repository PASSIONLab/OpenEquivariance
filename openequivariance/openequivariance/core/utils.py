import functools
import hashlib
import json
import math
import tempfile
from enum import IntEnum

import numpy as np

from openequivariance.core.e3nn_lite import Instruction, Irreps, TPProblem, wigner_3j


class DTypeEnum(IntEnum):
    """
    The C++ layer storess a copy of this map.
    """

    FLOAT32 = 1
    FLOAT64 = 2
    INT32 = 3
    INT64 = 4
    UINT8 = 5


dtype_to_enum = {
    np.float32: DTypeEnum.FLOAT32,
    np.float64: DTypeEnum.FLOAT64,
    np.int32: DTypeEnum.INT32,
    np.int64: DTypeEnum.INT64,
    np.uint8: DTypeEnum.UINT8,
    np.dtype(np.float32): DTypeEnum.FLOAT32,
    np.dtype(np.float64): DTypeEnum.FLOAT64,
    np.dtype(np.int32): DTypeEnum.INT32,
    np.dtype(np.int64): DTypeEnum.INT64,
    np.dtype(np.uint8): DTypeEnum.UINT8,
}


def sparse_outer_product_work(cg: np.ndarray) -> int:
    return np.sum(np.max(cg != 0, axis=2))


# Nonzeros
@functools.lru_cache(typed=True)
def count_cg_non_zero(l1, l2, l3) -> int:
    return np.count_nonzero(wigner_3j(l1, l2, l3))


def calculate_total_nnz(tpp: TPProblem) -> int:
    """
    To make sure you don't over count repeat CGs which get used multiple times
    """
    nnz_by_l_combo = {}
    for ins in tpp.instructions:  # type : Instruction
        l1 = tpp.irreps_in1[ins.i_in1].ir.l
        l2 = tpp.irreps_in2[ins.i_in2].ir.l
        l3 = tpp.irreps_out[ins.i_out].ir.l
        assert isinstance(l1, int)
        assert isinstance(l2, int)
        assert isinstance(l3, int)
        nnz_by_l_combo[(l1, l2, l3)] = count_cg_non_zero(l1, l2, l3)
    return sum(nnz_by_l_combo.values())


def calc_weight_offsets(tpp: TPProblem) -> list[int]:
    """
    Returns a list of weight offsets for every instruction.
    """
    assert isinstance(tpp, TPProblem)
    offset = 0
    offsets = []
    for ins in tpp.instructions:
        assert isinstance(ins, Instruction)
        offsets.append(offset)
        if ins.has_weight:
            flatsize = math.prod(ins.path_shape)
            offset += flatsize
    return offsets


def filter_and_analyze_problem(problem):
    """
    Centralized function that stops unhandled problem configurations,
    returns a dictionary of useful information about the problem.
    """
    for i, inst in enumerate(problem.instructions):
        assert inst.connection_mode == problem.instructions[0].connection_mode, (
            f"All instructions must have the same connection mode, got {inst.connection_mode} and {problem.instructions[0].connection_mode}"
        )

        assert inst.has_weight, (
            f"All instructions must have trainable weights, got {inst.has_weight} at index {i}"
        )

    assert problem.instructions[0].connection_mode in ["uvu", "uvw"], (
        f"Connection mode must be 'uvu' or 'uvw', got {problem.instructions[0].connection_mode}"
    )

    if problem.layout == "ir_mul":
        assert problem.instructions[0].connection_mode == "uvu", (
            "layout='ir_mul' is only supported for pure 'uvu' problems"
        )

    assert problem.irrep_dtype == problem.weight_dtype, (
        f"irrep_dtype and weight_dtype must be the same, got {problem.irrep_dtype} and {problem.weight_dtype}"
    )

    assert not problem.internal_weights, (
        f"Openequivariance does not support internal weights, got {problem.internal_weights}"
    )

    assert len(problem.instructions) > 0, "Tensor product has no valid instructions!"

    result = {
        "is_uvw": problem.instructions[0].connection_mode == "uvw",
    }
    return result


def torch_to_oeq_dtype(torch_dtype) -> type[np.generic]:
    """
    Convenience function; converts a torch datatype to the corresponding
    numpy datatype for use in TPProblem.

    :param torch_dtype: torch datatype (e.g., torch.float32, torch.float64)
    :return: numpy datatype (e.g., np.float32, np.float64)
    """

    global torch
    import torch

    if torch_dtype == torch.float32:
        return np.float32
    elif torch_dtype == torch.float64:
        return np.float64
    else:
        raise ValueError("Unsupported torch dtype!")


def oeq_to_torch_dtype(oeq_dtype: type[np.generic]):
    global torch
    import torch

    if oeq_dtype == np.float32:
        return torch.float32
    elif oeq_dtype == np.float64:
        return torch.float64
    else:
        raise ValueError("Unsupported numpy dtype!")


def benchmark(func, num_warmup, num_iter, mode="gpu_time", kernel_names=[]):
    """
    mode=gpu_time may include PyTorch overhead
    mode=kernel_time measures runtime for only the specified kernels
    """
    from openequivariance._torch.extlib import GPUTimer

    assert mode in ["gpu_time", "torch_kernel_time"]
    time_millis = np.zeros(num_iter, dtype=np.float32)
    timer = GPUTimer()

    for i in range(num_warmup):
        func()

    if mode == "gpu_time":
        for i in range(num_iter):
            timer.clear_L2_cache()
            timer.start()
            func()
            time_millis[i] = timer.stop_clock_get_elapsed()

    else:
        from torch.profiler import ProfilerActivity, profile, record_function

        trace_file = tempfile.NamedTemporaryFile().name

        for i in range(num_iter):
            timer.clear_L2_cache()
            with profile(
                activities=[ProfilerActivity.CUDA], record_shapes=True
            ) as prof:
                with record_function("profile"):
                    func()

            prof.export_chrome_trace(trace_file)
            with open(trace_file, "r") as f:
                trace = json.load(f)

            kernel_time = 0.0
            for event in trace["traceEvents"]:
                if "args" in event and "stream" in event["args"]:
                    event_time_ms = event["dur"] / 1000

                    add_event = False
                    for kernel_name in kernel_names:
                        if kernel_name in event["name"]:
                            add_event = True

                    if add_event:
                        kernel_time += event_time_ms

            time_millis[i] = kernel_time

    return time_millis


def hash_str_64(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:7], "big")


def transpose_irrep_layout(
    array: np.ndarray,
    irreps: Irreps,
    src_layout: str,
    dst_layout: str,
) -> np.ndarray:
    """
    Transpose irrep-packed feature arrays between `mul_ir` and `ir_mul` layouts.

    Expected input shape is `[..., irreps.dim]`. A new array is returned.
    If `src_layout == dst_layout`, this returns a copy.
    """
    if src_layout not in ("mul_ir", "ir_mul"):
        raise ValueError(f"Unsupported src_layout: {src_layout}")
    if dst_layout not in ("mul_ir", "ir_mul"):
        raise ValueError(f"Unsupported dst_layout: {dst_layout}")

    x = np.asarray(array)
    out = np.empty_like(x)

    if src_layout == dst_layout:
        out[...] = x
        return out

    slices = irreps.slices()
    for ir_idx, mul_ir in enumerate(irreps):
        mul = mul_ir.mul
        dim = mul_ir.ir.dim
        seg = slices[ir_idx]
        block = x[..., seg.start : seg.stop]

        if src_layout == "ir_mul" and dst_layout == "mul_ir":
            out[..., seg.start : seg.stop] = (
                block.reshape(*block.shape[:-1], dim, mul)
                .swapaxes(-1, -2)
                .reshape(*block.shape[:-1], mul * dim)
            )
        elif src_layout == "mul_ir" and dst_layout == "ir_mul":
            out[..., seg.start : seg.stop] = (
                block.reshape(*block.shape[:-1], mul, dim)
                .swapaxes(-1, -2)
                .reshape(*block.shape[:-1], dim * mul)
            )
        else:
            raise ValueError(
                f"Unsupported layout transpose: {src_layout} -> {dst_layout}"
            )

    return out
