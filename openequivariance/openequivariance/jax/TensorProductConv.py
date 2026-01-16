import numpy as np
from functools import partial
from typing import Optional
from openequivariance.jax import extlib

from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.core.LoopUnrollConv import LoopUnrollConv
from openequivariance.core.utils import hash_attributes
from openequivariance.jax.utils import reorder_jax

import jax
import jax.numpy as jnp

from openequivariance.benchmark.logging_utils import getLogger

logger = getLogger()


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9))
def forward(X, Y, W, rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs):
    forward_call = jax.ffi.ffi_call(
        "conv_forward", jax.ShapeDtypeStruct((X.shape[0], L3_dim), irrep_dtype)
    )
    return forward_call(X, Y, W, rows, cols, workspace, sender_perm, **attrs)


def forward_with_inputs(
    X, Y, W, rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs
):
    return forward(
        X, Y, W, rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs
    ), (X, Y, W, rows, cols, sender_perm, workspace)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def backward(X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs):
    backward_call = jax.ffi.ffi_call(
        "conv_backward",
        (
            jax.ShapeDtypeStruct(X.shape, irrep_dtype),
            jax.ShapeDtypeStruct(Y.shape, irrep_dtype),
            jax.ShapeDtypeStruct(W.shape, irrep_dtype),
        ),
    )
    return backward_call(X, Y, W, dZ, rows, cols, workspace, sender_perm, **attrs)


def backward_with_inputs(
    X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs
):
    return backward(
        X, Y, W, dZ, rows, cols, workspace, sender_perm, irrep_dtype, attrs
    ), (X, Y, W, dZ)  # rows, cols, sender_perm, workspace)


def double_backward(
    rows, cols, workspace, sender_perm, irrep_dtype, attrs, inputs, derivatives
):
    double_backward_call = jax.ffi.ffi_call(
        "conv_double_backward",
        (
            jax.ShapeDtypeStruct(inputs[0].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[1].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[2].shape, irrep_dtype),
            jax.ShapeDtypeStruct(inputs[3].shape, irrep_dtype),
        ),
    )
    return double_backward_call(
        *inputs, *derivatives, rows, cols, workspace, sender_perm, **attrs
    )


def backward_autograd(
    rows, cols, workspace, sender_perm, L3_dim, irrep_dtype, attrs, inputs, dZ
):
    return backward(
        inputs[0],
        inputs[1],
        inputs[2],
        dZ,
        rows,
        cols,
        workspace,
        sender_perm,
        irrep_dtype,
        attrs,
    )


forward.defvjp(forward_with_inputs, backward_autograd)
backward.defvjp(backward_with_inputs, double_backward)


class TensorProductConv(LoopUnrollConv):
    r"""
    Identical to ``oeq.torch.TensorProductConv`` with functionality in JAX, with one
    key difference: integer arrays passed to this function must have dtype
    ``np.int32`` (as opposed to ``np.int64`` in the PyTorch version).

    :param problem: Specification of the tensor product.
    :param deterministic: if ``False``, uses atomics for the convolution. If ``True``, uses a deterministic
           fixup-based algorithm. `Default`: ``False``.
    :param kahan: If ``True``, uses Kahan summation to improve accuracy during aggregation. To use this option,
           the input tensors must be in float32 precision AND you must set ``deterministic=True``. *Default*: ``False``.
    """

    def __init__(
        self, config: TPProblem, deterministic: bool = False, kahan: bool = False
    ):
        dp = extlib.DeviceProp(0)
        super().__init__(
            config,
            dp,
            extlib.postprocess_kernel,
            idx_dtype=np.int32,  # N.B. this is distinct from the PyTorch version
            torch_op=False,
            deterministic=deterministic,
            kahan=kahan,
        )

        self.attrs = {
            "kernel": self.jit_kernel,
            "forward_config": vars(self.forward_schedule.launch_config),
            "backward_config": vars(self.backward_schedule.launch_config),
            "double_backward_config": vars(self.double_backward_schedule.launch_config),
            "kernel_prop": self.kernel_prop,
        }
        hash_attributes(self.attrs)

        self.weight_numel = config.weight_numel
        self.L3_dim = self.config.irreps_out.dim

        self.workspace = jnp.zeros((self.workspace_size,), dtype=jnp.uint8)
        logger.info(
            f"Convolution requires {self.workspace_size // (2**20)}MB of workspace."
        )
        self.dummy_transpose_perm = jnp.zeros((1,), dtype=jnp.int32)

    def forward(
        self,
        X: jax.numpy.ndarray,
        Y: jax.numpy.ndarray,
        W: jax.numpy.ndarray,
        rows: jax.numpy.ndarray,
        cols: jax.numpy.ndarray,
        sender_perm: Optional[jax.numpy.ndarray] = None,
    ) -> jax.numpy.ndarray:
        r"""
        Computes the fused CG tensor product + convolution.

        :param X: Tensor of shape ``[|V|, problem.irreps_in1.dim()]``, datatype ``problem.irrep_dtype``.
        :param Y: Tensor of shape ``[|E|, problem.irreps_in1.dim()]``, datatype ``problem.irrep_dtype``.
        :param W: Tensor of datatype ``problem.weight_dtype`` and shape

            * ``[|E|, problem.weight_numel]`` if ``problem.shared_weights=False``
            * ``[problem.weight_numel]`` if ``problem.shared_weights=True``

        :param rows: Tensor of shape ``[|E|]`` with row indices for each nonzero in the adjacency matrix,
                datatype ``np.int32``. Must be row-major sorted along with ``cols`` when ``deterministic=True``.
        :param cols: Tensor of shape ``[|E|]`` with column indices for each nonzero in the adjacency matrix,
                datatype ``np.int32``.
        :param sender_perm: Tensor of shape ``[|E|]`` and ``np.int32`` datatype containing a
                permutation that transposes the adjacency matrix nonzeros from row-major to column-major order.
                Must be provided when ``deterministic=True``.

        :return: Tensor of shape ``[|V|, problem.irreps_out.dim()]``, datatype ``problem.irrep_dtype``.
        """
        if not self.deterministic:
            sender_perm = self.dummy_transpose_perm
        else:
            assert sender_perm is not None, (
                "Must provide sender_perm for deterministic convolutions."
            )

        return forward(
            X,
            Y,
            W,
            rows,
            cols,
            self.workspace,
            sender_perm,
            self.L3_dim,
            self.config.irrep_dtype,
            self.attrs,
        )

    def __call__(
        self,
        X: jax.numpy.ndarray,
        Y: jax.numpy.ndarray,
        W: jax.numpy.ndarray,
        rows: jax.numpy.ndarray,
        cols: jax.numpy.ndarray,
        sender_perm: Optional[jax.numpy.ndarray] = None,
    ) -> jax.numpy.ndarray:
        return self.forward(X, Y, W, rows, cols, sender_perm)

    def reorder_weights_from_e3nn(self, weights, has_batch_dim=True):
        return reorder_jax(self.forward_schedule, weights, "forward", has_batch_dim)

    def reorder_weights_to_e3nn(self, weights, has_batch_dim=True):
        return reorder_jax(self.forward_schedule, weights, "backward", has_batch_dim)

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        rows = graph.rows.astype(np.int32)
        cols = graph.cols.astype(np.int32)
        sender_perm = graph.transpose_perm.astype(np.int32)
        weights = self.reorder_weights_from_e3nn(
            weights, has_batch_dim=not self.config.shared_weights
        )
        result = self.forward(
            jax.numpy.asarray(L1_in),
            jax.numpy.asarray(L2_in),
            jax.numpy.asarray(weights),
            jax.numpy.asarray(rows),
            jax.numpy.asarray(cols),
            jax.numpy.asarray(sender_perm),
        )
        L3_out[:] = np.asarray(result)

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
        rows = graph.rows.astype(np.int32)
        cols = graph.cols.astype(np.int32)
        sender_perm = graph.transpose_perm.astype(np.int32)
        weights = self.reorder_weights_from_e3nn(
            weights, has_batch_dim=not self.config.shared_weights
        )

        backward_fn = jax.vjp(
            lambda X, Y, W: self.forward(
                X,
                Y,
                W,
                jax.numpy.asarray(rows),
                jax.numpy.asarray(cols),
                jax.numpy.asarray(sender_perm),
            ),
            jax.numpy.asarray(L1_in),
            jax.numpy.asarray(L2_in),
            jax.numpy.asarray(weights),
        )[1]
        L1_grad_jax, L2_grad_jax, weights_grad_jax = backward_fn(
            jax.numpy.asarray(L3_grad)
        )
        L1_grad[:] = np.asarray(L1_grad_jax)
        L2_grad[:] = np.asarray(L2_grad_jax)
        weights_grad[:] = np.asarray(weights_grad_jax)
        weights_grad[:] = self.reorder_weights_to_e3nn(
            weights_grad, has_batch_dim=not self.config.shared_weights
        )

    def double_backward_cpu(
        self, in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad, graph
    ):
        in1_jax = jax.numpy.asarray(in1)
        in2_jax = jax.numpy.asarray(in2)
        weights_jax = jax.numpy.asarray(weights)
        out_grad_jax = jax.numpy.asarray(out_grad)
        in1_dgrad_jax = jax.numpy.asarray(in1_dgrad)
        in2_dgrad_jax = jax.numpy.asarray(in2_dgrad)
        weights_dgrad_jax = jax.numpy.asarray(weights_dgrad)

        rows_jax = jax.numpy.asarray(graph.rows.astype(self.idx_dtype))
        cols_jax = jax.numpy.asarray(graph.cols.astype(self.idx_dtype))
        sender_perm_jax = jax.numpy.asarray(graph.transpose_perm.astype(self.idx_dtype))

        in1_grad, in2_grad, weights_grad, out_dgrad = jax.vjp(
            lambda x, y, w, o: jax.vjp(
                lambda a, b, c: self.forward(
                    a, b, c, rows_jax, cols_jax, sender_perm_jax
                ),
                x,
                y,
                w,
            )[1](o),
            in1_jax,
            in2_jax,
            weights_jax,
            out_grad_jax,
        )[1]((in1_dgrad_jax, in2_dgrad_jax, weights_dgrad_jax))

        return (
            np.asarray(in1_grad),
            np.asarray(in2_grad),
            np.asarray(weights_grad),
            np.asarray(out_dgrad),
        )
