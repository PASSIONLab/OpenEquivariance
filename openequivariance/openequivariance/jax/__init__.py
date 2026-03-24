import jax
import jax.numpy as jnp

from openequivariance.core.e3nn_lite import Irreps
from openequivariance.jax.TensorProduct import TensorProduct as TensorProduct
from openequivariance.jax.TensorProductConv import (
    TensorProductConv as TensorProductConv,
)


def transpose_irreps(
    array: jax.Array,
    irreps: Irreps,
    src_layout: str,
    dst_layout: str,
) -> jax.Array:
    r"""
    Transpose irrep-packed feature arrays between ``mul_ir`` and ``ir_mul`` layouts.

    The function operates on the trailing feature dimension and preserves all leading
    batch dimensions. It uses differentiable JAX operations, so gradients propagate
    through the transpose.

    :param array: Input feature array with shape ``[..., irreps.dim]``.
    :type array: jax.Array
    :param irreps: Irreps specification describing how the trailing feature dimension
                   is partitioned into irrep blocks.
    :type irreps: Irreps
    :param src_layout: Source layout. Must be either ``"mul_ir"`` or ``"ir_mul"``.
    :type src_layout: str
    :param dst_layout: Destination layout. Must be either ``"mul_ir"`` or ``"ir_mul"``.
    :type dst_layout: str

    :returns: Array in ``dst_layout`` with the same shape, dtype, and device as ``array``.
              If ``src_layout == dst_layout``, returns a copy of ``array``.
    :rtype: jax.Array

    :raises ValueError: If ``src_layout`` or ``dst_layout`` is not one of
                        ``"mul_ir"`` or ``"ir_mul"``.
    """
    if src_layout not in ("mul_ir", "ir_mul"):
        raise ValueError(f"Unsupported src_layout: {src_layout}")
    if dst_layout not in ("mul_ir", "ir_mul"):
        raise ValueError(f"Unsupported dst_layout: {dst_layout}")

    x = jnp.asarray(array)
    if src_layout == dst_layout:
        return jnp.array(x, copy=True)

    out = jnp.empty_like(x)
    slices = irreps.slices()

    for ir_idx, mul_ir in enumerate(irreps):
        mul = mul_ir.mul
        dim = mul_ir.ir.dim
        seg = slices[ir_idx]
        block = x[..., seg.start : seg.stop]

        if src_layout == "ir_mul" and dst_layout == "mul_ir":
            transposed = (
                block.reshape(*block.shape[:-1], dim, mul)
                .swapaxes(-1, -2)
                .reshape(*block.shape[:-1], mul * dim)
            )
        elif src_layout == "mul_ir" and dst_layout == "ir_mul":
            transposed = (
                block.reshape(*block.shape[:-1], mul, dim)
                .swapaxes(-1, -2)
                .reshape(*block.shape[:-1], dim * mul)
            )
        else:
            raise ValueError(
                f"Unsupported layout transpose: {src_layout} -> {dst_layout}"
            )

        out = out.at[..., seg.start : seg.stop].set(transposed)

    return out


__all__ = ["TensorProduct", "TensorProductConv", "transpose_irreps"]
