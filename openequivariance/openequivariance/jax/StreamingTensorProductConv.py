"""Receiver-streaming tensor-product convolution implementation."""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.FactorizedComputationSchedule import (
    factorized_schedule_from_problem,
)
from openequivariance.jax.utils import reorder_jax


class StreamingUnavailableError(ValueError):
    """Raised when a convolution cannot use receiver-row streaming."""


class StreamingConvTopology(NamedTuple):
    """Store receiver-sorted topology that can be reused by streaming calls.

    Invalid padded edges are sorted after valid rows and assigned a sentinel
    receiver. Forward traversal excludes the sentinel suffix, while reverse
    kernels discard it before reading floating-point operands.
    """

    order: jax.Array
    receivers: jax.Array
    senders: jax.Array
    row_ptr: jax.Array
    valid_edges: jax.Array


def streaming_support(config: TPProblem) -> tuple[bool, str]:
    """Report whether ``config`` can use streaming without constructing a kernel.

    The predicate only inspects immutable problem metadata and lowers the
    lightweight static schedule. It has no CUDA, JAX compilation, or other runtime
    side effects.
    """
    if config.shared_weights:
        return False, "receiver streaming requires unshared per-edge weights"
    if config.internal_weights:
        return False, "receiver streaming requires externally supplied weights"
    if config.layout not in ("mul_ir", "ir_mul"):
        return False, "receiver streaming supports only mul_ir and ir_mul layouts"
    if config.irrep_dtype != config.weight_dtype:
        return False, "receiver streaming requires matching irrep and weight dtypes"
    if config.irrep_dtype not in (np.float32, np.float64):
        return False, "receiver streaming supports only float32 and float64"
    native_config = config.clone()
    native_config.layout = "mul_ir"
    try:
        factorized_schedule_from_problem(native_config)
    except ValueError as error:
        return False, str(error)
    return True, ""


class StreamingTensorProductConv:
    """Apply a full external-weight convolution with receiver-owned accumulation.

    The operator always accepts ``X[N, input_dim]``, ``Y[E, edge_dim]``, and
    native-order ``W[E, weight_numel]``. The public ``TensorProductConv``
    selects this implementation only for supported configurations.

    Feature layouts ``"mul_ir"`` and ``"ir_mul"`` are both accepted. Both
    implementations execute in ``"mul_ir"`` and use differentiable boundary
    transposes when the public problem uses ``"ir_mul"``.
    """

    def __init__(self, config: TPProblem):
        """Create a streaming convolution for a previously validated problem."""
        if config.shared_weights:
            raise StreamingUnavailableError(
                "The streaming schedule requires unshared per-edge weights."
            )
        if config.internal_weights:
            raise StreamingUnavailableError(
                "The streaming schedule requires externally supplied weights."
            )
        if config.layout not in ("mul_ir", "ir_mul"):
            raise StreamingUnavailableError(
                "The streaming schedule supports only mul_ir and ir_mul layouts."
            )
        if config.irrep_dtype != config.weight_dtype:
            raise StreamingUnavailableError(
                "The streaming schedule requires matching irrep and weight dtypes."
            )
        if config.irrep_dtype not in (np.float32, np.float64):
            raise StreamingUnavailableError(
                "The streaming schedule supports only float32 and float64."
            )

        self.config = config
        self.weight_numel = config.weight_numel
        self.public_layout = config.layout
        native_config = config.clone()
        native_config.layout = "mul_ir"
        self._native_config = native_config
        try:
            self.schedule = factorized_schedule_from_problem(native_config)
        except ValueError as error:
            raise StreamingUnavailableError(str(error)) from error

    @property
    def uses_streaming_kernel(self) -> bool:
        """Return whether this object uses the receiver-streaming kernel."""
        return True

    @property
    def implementation(self) -> str:
        """Return the implementation name for this static problem."""
        return "streaming"

    def reorder_weights_from_e3nn(
        self, weights: jax.Array, has_batch_dim: bool = True
    ) -> jax.Array:
        """Convert canonical e3nn weights to the selected OEQ weight order."""
        weights = jnp.asarray(weights)
        if weights.ndim < 1 or weights.shape[-1] != self.weight_numel:
            raise ValueError(f"weights must end in dimension {self.weight_numel}")
        return reorder_jax(self.schedule, weights, "forward", has_batch_dim)

    def reorder_weights_to_e3nn(
        self, weights: jax.Array, has_batch_dim: bool = True
    ) -> jax.Array:
        """Convert selected OEQ-order weights to canonical e3nn order."""
        weights = jnp.asarray(weights)
        if weights.ndim < 1 or weights.shape[-1] != self.weight_numel:
            raise ValueError(f"weights must end in dimension {self.weight_numel}")
        return reorder_jax(self.schedule, weights, "backward", has_batch_dim)

    @staticmethod
    def _prepare_topology(
        rows: jax.Array,
        cols: jax.Array,
        num_nodes: int,
        *,
        indices_are_sorted: bool = False,
    ) -> StreamingConvTopology:
        """Prepare receiver rows and offsets for one streaming call.

        When the caller has already sorted the edge operands by receiver, this
        method reuses their order and only builds the offsets.  Otherwise it
        creates the stable permutation that places valid edges first in
        nondecreasing receiver order.
        """
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative")
        if num_nodes == 0 and rows.shape[0] != 0:
            raise ValueError("N=0 requires E=0 for streaming convolution")
        if rows.ndim != 1 or cols.ndim != 1 or rows.shape != cols.shape:
            raise ValueError("rows and cols must be equal-length vectors")
        if rows.dtype != jnp.int32 or cols.dtype != jnp.int32:
            raise ValueError("streaming convolution topology must use int32 indices")
        valid_edges = (
            (rows >= 0) & (rows < num_nodes) & (cols >= 0) & (cols < num_nodes)
        )
        receivers = jnp.where(valid_edges, rows, num_nodes)
        order = (
            jnp.arange(rows.shape[0], dtype=jnp.int32)
            if indices_are_sorted
            else jnp.argsort(receivers, stable=True).astype(jnp.int32)
        )
        edges_per_receiver = jnp.bincount(
            jnp.where(valid_edges, rows, 0),
            weights=valid_edges.astype(rows.dtype),
            length=num_nodes,
        ).astype(rows.dtype)
        row_ptr = jnp.concatenate(
            (jnp.zeros((1,), dtype=rows.dtype), jnp.cumsum(edges_per_receiver))
        )
        return StreamingConvTopology(
            order=order,
            receivers=receivers[order],
            senders=cols[order],
            row_ptr=row_ptr,
            valid_edges=valid_edges[order],
        )

    @staticmethod
    def _validate_row_ptr(
        row_ptr: jax.Array, rows: jax.Array, cols: jax.Array, num_nodes: int
    ) -> None:
        """Validate the shape and dtype of cached receiver offsets."""
        del rows, cols
        if row_ptr.dtype != jnp.int32 or row_ptr.shape != (num_nodes + 1,):
            raise ValueError("row_ptr must be an int32 vector with shape [N + 1]")

    def _validate_native_operands(
        self,
        X: jax.Array,
        Y: jax.Array,
        W: jax.Array,
        rows: jax.Array,
        cols: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Validate the public receiver-streaming operands."""
        X, Y, W = map(jnp.asarray, (X, Y, W))
        if X.ndim != 2 or X.shape[1] != self.schedule.input_dim:
            raise ValueError(f"X must have shape [N, {self.schedule.input_dim}]")
        if Y.ndim != 2 or Y.shape[1] != self.schedule.edge_dim:
            raise ValueError(f"Y must have shape [E, {self.schedule.edge_dim}]")
        if W.ndim != 2 or W.shape != (Y.shape[0], self.weight_numel):
            raise ValueError(f"W must have shape [E, {self.weight_numel}]")
        if X.dtype != Y.dtype or X.dtype != W.dtype:
            raise ValueError("X, Y, and W must have matching dtypes")
        expected_dtype = jnp.dtype(self.config.irrep_dtype)
        if X.dtype != expected_dtype:
            raise ValueError(
                f"X, Y, and W must have the configured dtype {expected_dtype}"
            )
        if X.shape[0] == 0 and Y.shape[0] != 0:
            raise ValueError("N=0 requires E=0 for streaming convolution")
        if rows.ndim != 1 or cols.ndim != 1 or rows.shape != cols.shape:
            raise ValueError("rows and cols must be equal-length vectors")
        if rows.shape[0] != Y.shape[0]:
            raise ValueError("topology and edge feature counts must agree")
        if rows.dtype != jnp.int32 or cols.dtype != jnp.int32:
            raise ValueError("streaming convolution topology must use int32 indices")
        return X, Y, W

    def forward(
        self,
        X: jax.Array,
        Y: jax.Array,
        W: jax.Array,
        rows: Optional[jax.Array] = None,
        cols: Optional[jax.Array] = None,
        *,
        indices_are_sorted: bool = False,
        row_ptr: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Apply receiver streaming with the public sorted-index contract."""
        if not isinstance(indices_are_sorted, bool):
            raise TypeError("indices_are_sorted must be a Python bool")
        if row_ptr is not None and not indices_are_sorted:
            raise ValueError("row_ptr requires indices_are_sorted=True")
        X, Y, W = self._validate_native_operands(X, Y, W, rows, cols)
        if Y.shape[0] == 0:
            return jnp.zeros((X.shape[0], self.schedule.output_dim), dtype=X.dtype)
        topology = self._prepare_topology(
            rows, cols, X.shape[0], indices_are_sorted=indices_are_sorted
        )
        if row_ptr is not None:
            self._validate_row_ptr(row_ptr, rows, cols, X.shape[0])
            topology = topology._replace(row_ptr=row_ptr)
        if not indices_are_sorted:
            Y, W = Y[topology.order], W[topology.order]
        if self.public_layout == "ir_mul":
            from openequivariance.jax import transpose_irreps

            X = transpose_irreps(X, self.config.irreps_in1, "ir_mul", "mul_ir")
            Y = transpose_irreps(Y, self.config.irreps_in2, "ir_mul", "mul_ir")
        from openequivariance.jax.jvp.factorized_projected_prim import (
            factorized_projected,
        )

        output = factorized_projected(
            self.schedule,
            X,
            Y,
            W,
            topology.senders,
            topology.receivers,
            topology.row_ptr,
        )
        if self.public_layout == "ir_mul":
            output = transpose_irreps(
                output, self.config.irreps_out, "mul_ir", "ir_mul"
            )
        return output

    __call__ = forward


__all__ = [
    "StreamingUnavailableError",
    "StreamingTensorProductConv",
    "streaming_support",
]
