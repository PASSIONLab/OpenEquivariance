"""Public JAX tensor-product convolution selection."""

from typing import Optional

import jax
import numpy as np

from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.utils import transpose_irrep_layout
from openequivariance.jax.LoopUnrollTensorProductConv import (
    LoopUnrollTensorProductConv,
)
from openequivariance.jax.StreamingTensorProductConv import (
    StreamingTensorProductConv,
    StreamingUnavailableError,
    streaming_support,
)


class TensorProductConv:
    r"""Apply a tensor-product convolution to a directed graph.

    The mode selects between the general standard convolution, which supports
    all tensor-product operations, and a more efficient streaming convolution
    for UVU operations. The streaming implementation is based on the
    computational schedule proposed by Chorošajev and Bény [CB2026]_.
    ``mode="auto"`` uses streaming when supported and otherwise selects the
    standard convolution.

    :param config: Specification of the tensor product.
    :param deterministic: Request deterministic aggregation. This selects the
        standard implementation in automatic mode and is unavailable in
        streaming mode.
    :param kahan: Request Kahan summation. This selects the standard
        implementation in automatic mode and is unavailable in streaming mode.
    :param requires_jvp: Enable JVP support for the standard implementation.
    :param mode: One of ``"auto"``, ``"streaming"``, or ``"standard"``.

    .. [CB2026] Chorošajev and Bény, *Sobek: Streaming Equivariant Tensor
        Product Convolutions*, arXiv (2026).
        https://doi.org/10.48550/arXiv.2607.18074
    """

    _MODES = frozenset(("auto", "streaming", "standard"))

    def __init__(
        self,
        config: TPProblem,
        deterministic: bool = False,
        kahan: bool = False,
        requires_jvp: bool = True,
        mode: str = "standard",
    ):
        """Create a convolution and select one static implementation."""
        if mode not in self._MODES:
            choices = ", ".join(sorted(self._MODES))
            raise ValueError(f"mode must be one of {choices}. Received {mode!r}")

        self.config = config
        self.mode = mode
        self.deterministic = deterministic
        self.kahan = kahan
        self.requires_jvp = requires_jvp
        self.weight_numel = config.weight_numel
        self._standard_layout_adapter = False
        supports_streaming, reason = streaming_support(config)
        if deterministic:
            supports_streaming = False
            reason = (
                "deterministic aggregation is provided by the standard implementation"
            )
        elif kahan:
            supports_streaming = False
            reason = "Kahan summation is provided by the standard implementation"

        if mode == "streaming" and not supports_streaming:
            raise StreamingUnavailableError(
                f"mode='streaming' is unavailable: {reason}."
            )
        if mode != "standard" and supports_streaming:
            self._impl: StreamingTensorProductConv | LoopUnrollTensorProductConv = (
                StreamingTensorProductConv(config)
            )
        else:
            standard_config = config
            if config.layout == "ir_mul":
                # The established loop-unroll kernels are native ``mul_ir``
                # kernels.  Preserve the public layout at this boundary
                # instead of making their long-standing schedule interpret a
                # different memory order.
                standard_config = config.clone()
                standard_config.layout = "mul_ir"
                self._standard_layout_adapter = True
            self._impl = LoopUnrollTensorProductConv(
                standard_config,
                deterministic=deterministic,
                kahan=kahan,
                requires_jvp=requires_jvp,
            )

    @property
    def uses_streaming_kernel(self) -> bool:
        """Return whether this object selected receiver-row streaming."""
        return isinstance(self._impl, StreamingTensorProductConv)

    @property
    def implementation(self) -> str:
        """Return ``"streaming"`` or ``"standard"`` for this object."""
        return "streaming" if self.uses_streaming_kernel else "standard"

    @property
    def L3_dim(self) -> int:
        """Return the flattened output feature dimension."""
        return self.config.irreps_out.dim

    def reorder_weights_from_e3nn(self, weights, has_batch_dim: bool = True):
        """Convert canonical e3nn weights to the selected OEQ weight order."""
        return self._impl.reorder_weights_from_e3nn(weights, has_batch_dim)

    def reorder_weights_to_e3nn(self, weights, has_batch_dim: bool = True):
        """Convert selected OEQ-order weights to canonical e3nn order."""
        return self._impl.reorder_weights_to_e3nn(weights, has_batch_dim)

    def forward(
        self,
        X: jax.Array,
        Y: jax.Array,
        W: jax.Array,
        rows: Optional[jax.Array] = None,
        cols: Optional[jax.Array] = None,
        sender_perm: Optional[jax.Array] = None,
        *,
        indices_are_sorted: bool = False,
        row_ptr: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Apply the selected convolution implementation.

        When ``indices_are_sorted`` is true, ``rows``, ``cols``, ``Y``, and
        ``W`` must already share receiver-row order. Streaming then skips its
        topology sort and operand gathers. The standard implementation simply
        consumes that aligned order. A deterministic ``sender_perm`` must use
        the same order as the supplied edge operands. For streaming, valid
        edges must form a nondecreasing receiver-row prefix and invalid padding
        must be a tail. Under JIT this is a trusted contract, so it adds no
        runtime validation. The standard implementation requires every
        endpoint to be valid. ``row_ptr`` optionally reuses the cached receiver
        offsets for a sorted streaming call.
        """
        if not isinstance(indices_are_sorted, bool):
            raise TypeError("indices_are_sorted must be a Python bool")
        if row_ptr is not None and not indices_are_sorted:
            raise ValueError("row_ptr requires indices_are_sorted=True")
        if rows is None or cols is None:
            raise ValueError("rows and cols are required for convolution")
        if self.uses_streaming_kernel:
            if sender_perm is not None:
                raise StreamingUnavailableError(
                    "Receiver streaming does not accept sender_perm. Select "
                    "mode='standard' for deterministic aggregation."
                )
            assert isinstance(self._impl, StreamingTensorProductConv)
            return self._impl.forward(
                X,
                Y,
                W,
                rows,
                cols,
                indices_are_sorted=indices_are_sorted,
                row_ptr=row_ptr,
            )
        assert isinstance(self._impl, LoopUnrollTensorProductConv)
        if getattr(self, "_standard_layout_adapter", False):
            from openequivariance.jax import transpose_irreps

            X = transpose_irreps(X, self.config.irreps_in1, "ir_mul", "mul_ir")
            Y = transpose_irreps(Y, self.config.irreps_in2, "ir_mul", "mul_ir")
        output = self._impl.forward(
            X,
            Y,
            W,
            rows,
            cols,
            sender_perm,
            indices_are_sorted=indices_are_sorted,
        )
        if getattr(self, "_standard_layout_adapter", False):
            output = transpose_irreps(
                output, self.config.irreps_out, "mul_ir", "ir_mul"
            )
        return output

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        """Run the standard CPU helper when the standard implementation is selected."""
        if self.uses_streaming_kernel:
            raise StreamingUnavailableError(
                "Receiver streaming does not provide the CPU helper API."
            )
        assert isinstance(self._impl, LoopUnrollTensorProductConv)
        if not self._standard_layout_adapter:
            return self._impl.forward_cpu(L1_in, L2_in, weights, L3_out, graph)
        native_output = np.empty_like(L3_out)
        self._impl.forward_cpu(
            transpose_irrep_layout(L1_in, self.config.irreps_in1, "ir_mul", "mul_ir"),
            transpose_irrep_layout(L2_in, self.config.irreps_in2, "ir_mul", "mul_ir"),
            weights,
            native_output,
            graph,
        )
        L3_out[...] = transpose_irrep_layout(
            native_output, self.config.irreps_out, "mul_ir", "ir_mul"
        )

    def backward_cpu(
        self,
        L1_in,
        L1_grad,
        L2_in,
        L2_grad,
        L3_grad,
        weights,
        weights_grad,
        graph,
    ):
        """Run the standard backward CPU helper when it is selected."""
        if self.uses_streaming_kernel:
            raise StreamingUnavailableError(
                "Receiver streaming does not provide the CPU helper API."
            )
        assert isinstance(self._impl, LoopUnrollTensorProductConv)
        if not self._standard_layout_adapter:
            return self._impl.backward_cpu(
                L1_in, L1_grad, L2_in, L2_grad, L3_grad, weights, weights_grad, graph
            )
        native_l1_grad = np.empty_like(L1_grad)
        native_l2_grad = np.empty_like(L2_grad)
        self._impl.backward_cpu(
            transpose_irrep_layout(L1_in, self.config.irreps_in1, "ir_mul", "mul_ir"),
            native_l1_grad,
            transpose_irrep_layout(L2_in, self.config.irreps_in2, "ir_mul", "mul_ir"),
            native_l2_grad,
            transpose_irrep_layout(L3_grad, self.config.irreps_out, "ir_mul", "mul_ir"),
            weights,
            weights_grad,
            graph,
        )
        L1_grad[...] = transpose_irrep_layout(
            native_l1_grad, self.config.irreps_in1, "mul_ir", "ir_mul"
        )
        L2_grad[...] = transpose_irrep_layout(
            native_l2_grad, self.config.irreps_in2, "mul_ir", "ir_mul"
        )

    def double_backward_cpu(
        self, in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad, graph
    ):
        """Run the standard double-backward CPU helper when it is selected."""
        if self.uses_streaming_kernel:
            raise StreamingUnavailableError(
                "Receiver streaming does not provide the CPU helper API."
            )
        assert isinstance(self._impl, LoopUnrollTensorProductConv)
        if not self._standard_layout_adapter:
            return self._impl.double_backward_cpu(
                in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad, graph
            )
        in1_grad, in2_grad, weights_grad, out_dgrad = self._impl.double_backward_cpu(
            transpose_irrep_layout(in1, self.config.irreps_in1, "ir_mul", "mul_ir"),
            transpose_irrep_layout(in2, self.config.irreps_in2, "ir_mul", "mul_ir"),
            transpose_irrep_layout(
                out_grad, self.config.irreps_out, "ir_mul", "mul_ir"
            ),
            weights,
            weights_dgrad,
            transpose_irrep_layout(
                in1_dgrad, self.config.irreps_in1, "ir_mul", "mul_ir"
            ),
            transpose_irrep_layout(
                in2_dgrad, self.config.irreps_in2, "ir_mul", "mul_ir"
            ),
            graph,
        )
        return (
            transpose_irrep_layout(
                in1_grad, self.config.irreps_in1, "mul_ir", "ir_mul"
            ),
            transpose_irrep_layout(
                in2_grad, self.config.irreps_in2, "mul_ir", "ir_mul"
            ),
            weights_grad,
            transpose_irrep_layout(
                out_dgrad, self.config.irreps_out, "mul_ir", "ir_mul"
            ),
        )

    __call__ = forward


__all__ = ["TensorProductConv"]
