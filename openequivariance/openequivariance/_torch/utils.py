import torch
import numpy as np
from types import MappingProxyType
from openequivariance.core.utils import DTypeEnum
from openequivariance.core.e3nn_lite import Irreps


def reorder_helper(schedule, weights_in, direction, has_batch_dim):
    assert direction in ["forward", "backward"]

    specs = schedule.weight_reordering_info(weights_in, has_batch_dim)
    weights_out = torch.zeros_like(weights_in)

    for spec in specs:
        parent_range = spec["parent_range"]
        parent_shape = spec["parent_shape"]
        weights_subrange = spec["weights_subrange"]
        child_range = spec["child_range"]
        transpose_perm = spec["transpose_perm"]

        if direction == "forward":
            reshape_size = spec["reshape_size"]

            sliced_weights = weights_in[parent_range].reshape(parent_shape)[
                weights_subrange
            ]

            weights_out[child_range] = sliced_weights.permute(transpose_perm).reshape(
                reshape_size
            )

        elif direction == "backward":
            transpose_child_shape = spec["transpose_child_shape"]
            child_shape = spec["child_shape"]

            sliced_weights = (
                weights_in[child_range]
                .reshape(transpose_child_shape)
                .permute(transpose_perm)
            )

            weights_out[parent_range].reshape(parent_shape)[weights_subrange] = (
                sliced_weights.flatten().reshape(child_shape)
            )

    return weights_out


def reorder_numpy_helper(schedule, weights_in, direction, has_batch_dim):
    weights_in = torch.from_numpy(weights_in.copy())
    result = reorder_helper(schedule, weights_in, direction, has_batch_dim)
    return result.detach().cpu().numpy().copy()


def reorder_torch(schedule, weights_in, direction, has_batch_dim):
    if isinstance(weights_in, torch.Tensor):
        return reorder_helper(schedule, weights_in, direction, has_batch_dim)
    else:
        return reorder_numpy_helper(schedule, weights_in, direction, has_batch_dim)


enum_to_torch_dtype = MappingProxyType(
    {
        DTypeEnum.FLOAT32: torch.float32,
        DTypeEnum.FLOAT64: torch.float64,
        DTypeEnum.INT32: torch.int32,
        DTypeEnum.INT64: torch.int64,
        DTypeEnum.UINT8: torch.uint8,
    }
)


def string_to_tensor(text: str) -> torch.Tensor:
    bytes_data = text.encode("utf-8")
    np_bytes = np.frombuffer(bytes_data, dtype=np.uint8)
    result = torch.tensor(np_bytes, device="cpu")
    result.requires_grad = False
    return result


def transpose_irreps(
    array: torch.Tensor,
    irreps: Irreps,
    src_layout: str,
    dst_layout: str,
) -> torch.Tensor:
    r"""
    Transpose irrep-packed feature tensors between ``mul_ir`` and ``ir_mul`` layouts.

    The function operates on the trailing feature dimension and preserves all leading
    batch dimensions. It uses only differentiable PyTorch tensor operations, so gradients
    propagate through the transpose.

    :param array: Input feature tensor with shape ``[..., irreps.dim]``.
    :param irreps: Irreps specification describing how the trailing feature dimension
                   is partitioned into irrep blocks.
    :param src_layout: Source layout. Must be either ``"mul_ir"`` or ``"ir_mul"``.
    :param dst_layout: Destination layout. Must be either ``"mul_ir"`` or ``"ir_mul"``.


    :returns: Tensor in ``dst_layout`` with the same shape, dtype, and device as ``array``.
              If ``src_layout == dst_layout``, returns a clone of ``array``.


    :raises TypeError: If ``array`` is not a ``torch.Tensor``.
    :raises ValueError: If ``src_layout`` or ``dst_layout`` is not one of
                        ``"mul_ir"`` or ``"ir_mul"``.
    """
    if src_layout not in ("mul_ir", "ir_mul"):
        raise ValueError(f"Unsupported src_layout: {src_layout}")
    if dst_layout not in ("mul_ir", "ir_mul"):
        raise ValueError(f"Unsupported dst_layout: {dst_layout}")

    if not isinstance(array, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(array)}")

    out = torch.empty_like(array)

    if src_layout == dst_layout:
        out.copy_(array)
        return out

    slices = irreps.slices()
    for ir_idx, mul_ir in enumerate(irreps):
        mul = mul_ir.mul
        dim = mul_ir.ir.dim
        seg = slices[ir_idx]
        block = array[..., seg.start : seg.stop]

        if src_layout == "ir_mul" and dst_layout == "mul_ir":
            out[..., seg.start : seg.stop] = (
                block.reshape(*block.shape[:-1], dim, mul)
                .transpose(-1, -2)
                .reshape(*block.shape[:-1], mul * dim)
            )
        elif src_layout == "mul_ir" and dst_layout == "ir_mul":
            out[..., seg.start : seg.stop] = (
                block.reshape(*block.shape[:-1], mul, dim)
                .transpose(-1, -2)
                .reshape(*block.shape[:-1], dim * mul)
            )
        else:
            raise ValueError(
                f"Unsupported layout transpose: {src_layout} -> {dst_layout}"
            )

    return out
